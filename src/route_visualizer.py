from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from database_interfaces.board_lib_interface import BoardLibInterface


ROLE_COLORS = {
	"start": "#00FF00",
	"middle": "#00FFFF",
	"finish": "#FF00FF",
	"foot": "#FFA500",
}


def _project_root() -> Path:
	return Path(__file__).resolve().parents[1]


def _default_db_path() -> Path:
	return _project_root() / "data" / "raw" / "kilter_database.sqlite"


def _default_images_root() -> Path:
	return _project_root() / "data" / "raw" / "kilter_images"


def _get_climb_row_by_name(db: BoardLibInterface, climb_name: str) -> tuple | None:
	return db.execute(
		"""
		SELECT uuid, name, layout_id, created_at
		FROM climbs
		WHERE name = ?
		ORDER BY created_at DESC
		LIMIT 1
		""",
		[climb_name],
	).fetchone()


def _decode_hsm_set_ids(db: BoardLibInterface, layout_id: int, hsm: int) -> list[int]:
	"""Decode the climb's hold-set mask into set_ids for the given layout."""
	rows = db.fetchall(
		"""
		SELECT DISTINCT set_id
		FROM product_sizes_layouts_sets
		WHERE layout_id = ?
		ORDER BY set_id
		""",
		[layout_id],
	)
	available_set_ids = [int(row[0]) for row in rows]
	return [
		set_id
		for bit_index, set_id in enumerate(available_set_ids)
		if int(hsm or 0) & (1 << bit_index)
	]


def _select_product_size(
	db: BoardLibInterface,
	layout_id: int,
	set_ids: list[int],
	climb_edges: tuple[int, int, int, int],
	product_size_id: Optional[int] = None,
) -> tuple[int, tuple[int, int, int, int]]:
	"""Choose a board size that contains the climb and has images for all needed sets."""
	climb_left, climb_right, climb_bottom, climb_top = climb_edges
	set_placeholders = ",".join(["?"] * len(set_ids))

	if product_size_id is not None:
		row = db.execute(
			f"""
			SELECT ps.id, ps.edge_left, ps.edge_right, ps.edge_bottom, ps.edge_top,
			       COUNT(DISTINCT psls.set_id) AS matched_sets
			FROM product_sizes ps
			LEFT JOIN product_sizes_layouts_sets psls
				ON psls.product_size_id = ps.id
				AND psls.layout_id = ?
				AND psls.set_id IN ({set_placeholders})
			WHERE ps.id = ?
			GROUP BY ps.id, ps.edge_left, ps.edge_right, ps.edge_bottom, ps.edge_top
			""",
			[layout_id, *set_ids, product_size_id],
		).fetchone()
		if row is None:
			raise ValueError(f"Unknown product_size_id={product_size_id}.")

		selected_id = int(row[0])
		board_edges = (
			int(row[1]),
			int(row[2]),
			int(row[3]),
			int(row[4]),
		)
		matched_sets = int(row[5])
		fits_climb = (
			board_edges[0] <= climb_left
			and board_edges[1] >= climb_right
			and board_edges[2] <= climb_bottom
			and board_edges[3] >= climb_top
		)
		if matched_sets != len(set_ids):
			raise ValueError(
				f"product_size_id={product_size_id} does not have all required images for layout_id={layout_id}."
			)
		if not fits_climb:
			raise ValueError(
				f"product_size_id={product_size_id} does not fit climb bounds {climb_edges}."
			)
		return selected_id, board_edges

	row = db.execute(
		f"""
		SELECT ps.id, ps.edge_left, ps.edge_right, ps.edge_bottom, ps.edge_top,
		       ((ps.edge_right - ps.edge_left) * (ps.edge_top - ps.edge_bottom)) AS board_area,
		       COUNT(DISTINCT psls.set_id) AS matched_sets
		FROM product_sizes ps
		JOIN product_sizes_layouts_sets psls
			ON psls.product_size_id = ps.id
			AND psls.layout_id = ?
			AND psls.set_id IN ({set_placeholders})
		WHERE ps.edge_left <= ?
		  AND ps.edge_right >= ?
		  AND ps.edge_bottom <= ?
		  AND ps.edge_top >= ?
		GROUP BY ps.id, ps.edge_left, ps.edge_right, ps.edge_bottom, ps.edge_top
		HAVING matched_sets = ?
		ORDER BY board_area ASC, ps.id ASC
		LIMIT 1
		""",
		[layout_id, *set_ids, climb_left, climb_right, climb_bottom, climb_top, len(set_ids)],
	).fetchone()
	if row is None:
		raise ValueError(
			"No product_size_id can fit this climb's edge bounds while providing all required set images."
		)

	selected_id = int(row[0])
	board_edges = (
		int(row[1]),
		int(row[2]),
		int(row[3]),
		int(row[4]),
	)
	return selected_id, board_edges


def _resolve_images_for_climb(
	db: BoardLibInterface,
	climb_uuid: str,
	layout_id: int,
	images_root: Path,
	product_size_id: Optional[int] = None,
) -> tuple[list[Path], tuple[int, int, int, int]]:
	"""Return compositable image layers and board bounds for the climb.

	The required sets are decoded from the climb's ``hsm`` bitmask within the
	context of its ``layout_id``. If ``product_size_id`` is omitted, selects the
	smallest board whose edge bounds still contain the climb.
	"""

	climb = db.get_climb_by_uuid(climb_uuid)
	if not climb or not climb.get("frames"):
		raise ValueError("Climb has no frame data.")

	set_ids = _decode_hsm_set_ids(db, layout_id, int(climb.get("hsm") or 0))
	if not set_ids:
		raise ValueError(f"Could not decode hold sets from hsm for layout_id={layout_id}.")

	climb_edges = (
		int(climb.get("edge_left") or 0),
		int(climb.get("edge_right") or 0),
		int(climb.get("edge_bottom") or 0),
		int(climb.get("edge_top") or 0),
	)
	product_size_id, board_edges = _select_product_size(
		db,
		layout_id,
		set_ids,
		climb_edges,
		product_size_id=product_size_id,
	)

	set_placeholders = ",".join(["?"] * len(set_ids))

	rows = db.fetchall(
		f"""
		SELECT set_id, image_filename
		FROM product_sizes_layouts_sets
		WHERE layout_id = ? AND product_size_id = ? AND set_id IN ({set_placeholders})
		ORDER BY set_id
		""",
		[layout_id, product_size_id, *set_ids],
	)
	if not rows:
		raise ValueError(
			f"No images found for layout_id={layout_id}, "
			f"product_size_id={product_size_id}, set_ids={set_ids}."
		)

	image_paths: list[Path] = []
	for _set_id, image_filename in rows:
		image_path = images_root / str(image_filename)
		if not image_path.exists():
			raise FileNotFoundError(f"Image file not found: {image_path}")
		image_paths.append(image_path)

	if len(image_paths) != len(set_ids):
		raise ValueError(
			f"Missing one or more image layers for layout_id={layout_id}, product_size_id={product_size_id}."
		)

	return image_paths, board_edges


def _composite_images(image_paths: list[Path]) -> np.ndarray:
	"""Alpha-composite all PNG layers into a single RGBA array."""
	base = Image.open(image_paths[0]).convert("RGBA")
	for path in image_paths[1:]:
		overlay = Image.open(path).convert("RGBA")
		if overlay.size != base.size:
			overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
		base = Image.alpha_composite(base, overlay)
	return np.array(base)


def visualize_route(climb_name: str, product_size_id: Optional[int] = None) -> None:
	"""Display a climb on the correct board image with holds circled by role color."""

	db_path = _default_db_path()
	images_root = _default_images_root()

	with BoardLibInterface(db_path) as db:
		climb_row = _get_climb_row_by_name(db, climb_name)
		if climb_row is None:
			raise ValueError(f"No climb found with name: {climb_name}")

		climb_uuid, resolved_name, layout_id, _ = climb_row
		holds = db.get_hold_positions_for_climb(climb_uuid)
		if not holds:
			raise ValueError(f"No hold data found for climb: {resolved_name}")

		image_paths, board_edges = _resolve_images_for_climb(
			db,
			climb_uuid,
			int(layout_id),
			images_root,
			product_size_id=product_size_id,
		)
		board_left, board_right, board_bottom, board_top = board_edges

	img = _composite_images(image_paths)
	img_h, img_w = img.shape[0], img.shape[1]

	def to_px(x: int, y: int) -> tuple[float, float]:
		x_norm = (x - board_left) / max(board_right - board_left, 1)
		y_norm = (y - board_bottom) / max(board_top - board_bottom, 1)
		x_px = x_norm * (img_w - 1)
		# Invert y to match image coords (top-left origin).
		y_px = (1.0 - y_norm) * (img_h - 1)
		return x_px, y_px

	fig, ax = plt.subplots(figsize=(10, 8))
	ax.imshow(img)
	ax.set_title(f"{resolved_name}")
	ax.axis("off")

	for hold in holds:
		if hold.x is None or hold.y is None:
			continue
		x_px, y_px = to_px(int(hold.x), int(hold.y))
		color = ROLE_COLORS.get((hold.role_name or "").lower(), "#FFFF00")
		circle = Circle((x_px, y_px), radius=25, fill=False, linewidth=2.2, edgecolor=color)
		ax.add_patch(circle)

	plt.tight_layout()
	plt.show()
 
 
if __name__ == "__main__":
	visualize_route("just a day")