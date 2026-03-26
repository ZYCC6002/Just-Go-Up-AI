from __future__ import annotations

import argparse
from pathlib import Path

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


def _composite_images(image_paths: list[Path]) -> np.ndarray:
	"""Alpha-composite all PNG layers into a single RGBA array."""
	base = Image.open(image_paths[0]).convert("RGBA")
	for path in image_paths[1:]:
		overlay = Image.open(path).convert("RGBA")
		if overlay.size != base.size:
			overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
		base = Image.alpha_composite(base, overlay)
	return np.array(base)


def visualize_route(climb_name: str) -> None:
	"""Display a climb on the correct board image with holds circled by role color."""

	db_path = _default_db_path()
	images_root = _default_images_root()

	with BoardLibInterface(db_path) as db:
		climb = db.get_climb_by_name(climb_name)
		if climb is None:
			raise ValueError(f"No climb found with name: {climb_name}")
		holds = db.get_hold_positions_for_climb(climb)
		if not holds:
			raise ValueError(f"No hold data found for climb: {climb.name}")

		image_paths, board_edges = db.resolve_image_paths_for_climb(
			climb,
			images_root,
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
	ax.set_title(f"{climb.name}")
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


def main() -> None:
	parser = argparse.ArgumentParser(description="Visualize a Kilter climb route on board image layers.")
	parser.add_argument("--climb-name", type=str, default="Alberts dream", help="Exact climb name to visualize.")
	args = parser.parse_args()
	visualize_route(args.climb_name)


if __name__ == "__main__":
	main()