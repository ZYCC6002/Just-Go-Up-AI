"""Interface helpers for BoardLib SQLite databases (e.g., kilter_database.sqlite)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import sqlite3


@dataclass(frozen=True)
class HoldPlacement:
	placement_id: int
	role_id: int
	hole_id: Optional[int]
	x: Optional[int]
	y: Optional[int]
	role_name: Optional[str]


class BoardLibInterface:
	"""Thin wrapper around BoardLib SQLite DBs.

	Example DBs:
	  - kilter_database.sqlite
	  - tension_database.sqlite
	"""

	def __init__(self, db_path: str | Path) -> None:
		self.db_path = Path(db_path)
		self._conn: Optional[sqlite3.Connection] = None

	def connect(self) -> sqlite3.Connection:
		if self._conn is None:
			self._conn = sqlite3.connect(str(self.db_path))
		return self._conn

	def close(self) -> None:
		if self._conn is not None:
			self._conn.close()
			self._conn = None

	def __enter__(self) -> "BoardLibInterface":
		self.connect()
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()

	def execute(self, query: str, params: Sequence[Any] | None = None) -> sqlite3.Cursor:
		conn = self.connect()
		cur = conn.cursor()
		cur.execute(query, params or [])
		return cur

	def fetchall(self, query: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]:
		return self.execute(query, params).fetchall()

	def list_tables(self) -> list[str]:
		rows = self.fetchall("SELECT name FROM sqlite_master WHERE type='table';")
		return [r[0] for r in rows]

	def get_table_schema(self, table: str) -> list[tuple[Any, ...]]:
		return self.fetchall(f"PRAGMA table_info({table});")

	def get_climb_by_uuid(self, climb_uuid: str) -> Optional[dict[str, Any]]:
		row = self.execute(
			"SELECT uuid, layout_id, name, frames, hsm, edge_left, edge_right, edge_bottom, edge_top, angle FROM climbs WHERE uuid = ?",
			[climb_uuid],
		).fetchone()
		if row is None:
			return None
		keys = ["uuid", "layout_id", "name", "frames", "hsm", "edge_left", "edge_right", "edge_bottom", "edge_top", "angle"]
		return dict(zip(keys, row))

	def get_climb_row_by_name(self, climb_name: str) -> tuple[Any, ...] | None:
		return self.execute(
			"""
			SELECT uuid, name, layout_id, created_at
			FROM climbs
			WHERE name = ?
			ORDER BY created_at DESC
			LIMIT 1
			""",
			[climb_name],
		).fetchone()

	def decode_hsm_set_ids(self, layout_id: int, hsm: int) -> list[int]:
		"""Decode a layout-specific hold-set mask into concrete set_ids."""
		rows = self.fetchall(
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

	def select_product_size(
		self,
		layout_id: int,
		set_ids: list[int],
		climb_edges: tuple[int, int, int, int],
		product_size_id: Optional[int] = None,
	) -> tuple[int, tuple[int, int, int, int]]:
		"""Choose a board size that fits the climb and has images for all required sets."""
		climb_left, climb_right, climb_bottom, climb_top = climb_edges
		set_placeholders = ",".join(["?"] * len(set_ids))

		if product_size_id is not None:
			row = self.execute(
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

		row = self.execute(
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

	def resolve_image_paths_for_climb(
		self,
		climb_uuid: str,
		images_root: Path,
		product_size_id: Optional[int] = None,
	) -> tuple[list[Path], tuple[int, int, int, int]]:
		"""Return compositable image layers and board bounds for a climb."""
		climb = self.get_climb_by_uuid(climb_uuid)
		if not climb or not climb.get("frames"):
			raise ValueError("Climb has no frame data.")

		layout_id = int(climb["layout_id"])
		set_ids = self.decode_hsm_set_ids(layout_id, int(climb.get("hsm") or 0))
		if not set_ids:
			raise ValueError(f"Could not decode hold sets from hsm for layout_id={layout_id}.")

		climb_edges = (
			int(climb.get("edge_left") or 0),
			int(climb.get("edge_right") or 0),
			int(climb.get("edge_bottom") or 0),
			int(climb.get("edge_top") or 0),
		)
		selected_product_size_id, board_edges = self.select_product_size(
			layout_id,
			set_ids,
			climb_edges,
			product_size_id=product_size_id,
		)

		set_placeholders = ",".join(["?"] * len(set_ids))
		rows = self.fetchall(
			f"""
			SELECT set_id, image_filename
			FROM product_sizes_layouts_sets
			WHERE layout_id = ? AND product_size_id = ? AND set_id IN ({set_placeholders})
			ORDER BY set_id
			""",
			[layout_id, selected_product_size_id, *set_ids],
		)
		if not rows:
			raise ValueError(
				f"No images found for layout_id={layout_id}, "
				f"product_size_id={selected_product_size_id}, set_ids={set_ids}."
			)

		image_paths: list[Path] = []
		for _set_id, image_filename in rows:
			image_path = images_root / str(image_filename)
			if not image_path.exists():
				raise FileNotFoundError(f"Image file not found: {image_path}")
			image_paths.append(image_path)

		if len(image_paths) != len(set_ids):
			raise ValueError(
				f"Missing one or more image layers for layout_id={layout_id}, product_size_id={selected_product_size_id}."
			)

		return image_paths, board_edges

	@staticmethod
	def parse_frames(frames: str) -> list[tuple[int, int]]:
		"""Parse frames string into (placement_id, role_id)."""
		if not frames:
			return []
		out: list[tuple[int, int]] = []
		i = 0
		while i < len(frames):
			if frames[i] != "p":
				i += 1
				continue
			i += 1
			start = i
			while i < len(frames) and frames[i].isdigit():
				i += 1
			placement_id = int(frames[start:i]) if i > start else 0
			if i < len(frames) and frames[i] == "r":
				i += 1
				start = i
				while i < len(frames) and frames[i].isdigit():
					i += 1
				role_id = int(frames[start:i]) if i > start else 0
				out.append((placement_id, role_id))
		return out

	def _role_map(self) -> dict[int, str]:
		rows = self.fetchall("SELECT id, name FROM placement_roles;")
		return {int(r[0]): str(r[1]) for r in rows}

	def get_hold_positions_for_climb(self, climb_uuid: str) -> list[HoldPlacement]:
		climb = self.get_climb_by_uuid(climb_uuid)
		if climb is None:
			return []

		frames = climb.get("frames") or ""
		pairs = self.parse_frames(frames)
		if not pairs:
			return []

		placement_ids = sorted({pid for pid, _ in pairs if pid})
		placeholders = ",".join(["?"] * len(placement_ids))
		placement_rows = self.fetchall(
			f"SELECT id, hole_id FROM placements WHERE id IN ({placeholders});",
			placement_ids,
		)
		placement_to_hole = {int(pid): int(hole_id) for pid, hole_id in placement_rows}

		hole_ids = sorted({hole_id for hole_id in placement_to_hole.values()})
		placeholders = ",".join(["?"] * len(hole_ids))
		hole_rows = self.fetchall(
			f"SELECT id, x, y FROM holes WHERE id IN ({placeholders});",
			hole_ids,
		)
		hole_to_xy = {int(hid): (int(x), int(y)) for hid, x, y in hole_rows}

		role_map = self._role_map()

		holds: list[HoldPlacement] = []
		for placement_id, role_id in pairs:
			hole_id = placement_to_hole.get(placement_id)
			coords = hole_to_xy.get(hole_id) if hole_id is not None else None
			x, y = coords if coords else (None, None)
			holds.append(
				HoldPlacement(
					placement_id=placement_id,
					role_id=role_id,
					hole_id=hole_id,
					x=x,
					y=y,
					role_name=role_map.get(role_id),
				)
			)
		return holds

	def iter_climb_hold_counts(self, climb_uuids: Iterable[str]) -> dict[str, int]:
		counts: dict[str, int] = {}
		for climb_uuid in climb_uuids:
			climb = self.get_climb_by_uuid(climb_uuid)
			if not climb:
				continue
			pairs = self.parse_frames(climb.get("frames") or "")
			counts[climb_uuid] = len(pairs)
		return counts


__all__ = ["BoardLibInterface", "HoldPlacement"]
