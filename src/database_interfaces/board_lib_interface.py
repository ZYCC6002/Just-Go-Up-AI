"""Interface helpers for BoardLib SQLite databases (e.g., kilter_database.sqlite)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import sqlite3

from database_interfaces.hold_metadata_importer import import_kilter_board_csv_metadata


@dataclass(frozen=True)
class HoldPlacement:
	placement_id: int
	role_id: int
	hole_id: Optional[int]
	x: Optional[int]
	y: Optional[int]
	role_name: Optional[str]
	metadata: Optional["ExternalHoldMetadata"] = None


@dataclass(frozen=True)
class ExternalHoldMetadata:
	hole_id: int
	external_hold_id: int
	type: Optional[str]
	function: Optional[str]
	depth: Optional[int]
	orientation: Optional[int]
	texture: Optional[int]
	size: Optional[int]
	source: str


@dataclass
class Climb:
	uuid: str
	layout_id: int
	name: str
	frames: str
	hsm: int
	edge_left: int
	edge_right: int
	edge_bottom: int
	edge_top: int
	angle: float
	product_id: Optional[int] = None
	holds: Optional[list[HoldPlacement]] = None

	@property
	def edges(self) -> tuple[int, int, int, int]:
		return (self.edge_left, self.edge_right, self.edge_bottom, self.edge_top)


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

	def _infer_product_id_from_frames(self, frames: str) -> Optional[int]:
		pairs = self.parse_frames(frames)
		placement_ids = sorted({pid for pid, _rid in pairs if pid})
		if not placement_ids:
			return None

		placeholders = ",".join(["?"] * len(placement_ids))
		rows = self.fetchall(
			f"""
			SELECT DISTINCT h.product_id
			FROM placements p
			JOIN holes h ON h.id = p.hole_id
			WHERE p.id IN ({placeholders})
			""",
			placement_ids,
		)
		product_ids = [int(r[0]) for r in rows if r and r[0] is not None]
		return product_ids[0] if product_ids else None

	def get_climb(self, climb_uuid: str) -> Optional[Climb]:
		row = self.execute(
			"SELECT uuid, layout_id, name, frames, hsm, edge_left, edge_right, edge_bottom, edge_top, angle FROM climbs WHERE uuid = ?",
			[climb_uuid],
		).fetchone()
		if row is None:
			return None

		frames = str(row[3] or "")
		product_id = self._infer_product_id_from_frames(frames)
		return Climb(
			uuid=str(row[0]),
			layout_id=int(row[1]),
			name=str(row[2]),
			frames=frames,
			hsm=int(row[4] or 0),
			edge_left=int(row[5] or 0),
			edge_right=int(row[6] or 0),
			edge_bottom=int(row[7] or 0),
			edge_top=int(row[8] or 0),
			angle=float(row[9] or 0.0),
			product_id=product_id,
		)

	def get_climb_by_name(self, climb_name: str) -> Optional[Climb]:
		row = self.execute(
			"""
			SELECT uuid
			FROM climbs
			WHERE name = ?
			ORDER BY created_at DESC
			LIMIT 1
			""",
			[climb_name],
		).fetchone()
		if row is None:
			return None
		return self.get_climb(str(row[0]))

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
		product_id: int,
	) -> tuple[int, tuple[int, int, int, int]]:
		"""Choose the smallest board size (for a product) that fits climb edges and required sets."""
		climb_left, climb_right, climb_bottom, climb_top = climb_edges
		set_placeholders = ",".join(["?"] * len(set_ids))

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
			WHERE ps.product_id = ?
			  AND ps.edge_left < ?
			  AND ps.edge_right > ?
			  AND ps.edge_bottom < ?
			  AND ps.edge_top > ?
			GROUP BY ps.id, ps.edge_left, ps.edge_right, ps.edge_bottom, ps.edge_top
			HAVING matched_sets = ?
			ORDER BY board_area ASC, ps.id ASC
			LIMIT 1
			""",
			[layout_id, *set_ids, product_id, climb_left, climb_right, climb_bottom, climb_top, len(set_ids)],
		).fetchone()
		if row is None:
			raise ValueError(
				"No product_size_id for the provided product_id can fit this climb's edge bounds while providing all required set images."
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
		climb: Climb,
		images_root: Path,
	) -> tuple[list[Path], tuple[int, int, int, int]]:
		"""Return compositable image layers and board bounds for a climb."""
		if not climb.frames:
			raise ValueError("Climb has no frame data.")

		layout_id = int(climb.layout_id)
		set_ids = self.decode_hsm_set_ids(layout_id, int(climb.hsm or 0))
		if not set_ids:
			raise ValueError(f"Could not decode hold sets from hsm for layout_id={layout_id}.")

		climb_edges = climb.edges

		if climb.product_id is None:
			raise ValueError("Climb object is missing product_id.")
		product_id = int(climb.product_id)
		selected_product_size_id, board_edges = self.select_product_size(
			layout_id,
			set_ids,
			climb_edges,
			product_id=product_id,
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

	def _table_exists(self, table_name: str) -> bool:
		row = self.execute(
			"SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
			[table_name],
		).fetchone()
		return row is not None

	def _fetch_external_hold_metadata(
		self,
		hole_ids: list[int],
		*,
		source: str,
	) -> dict[int, ExternalHoldMetadata]:
		if not hole_ids or not self._table_exists("external_hold_metadata"):
			return {}

		placeholders = ",".join(["?"] * len(hole_ids))
		rows = self.fetchall(
			f"""
			SELECT hole_id, external_hold_id, type, function, depth, orientation, texture, size, source
			FROM external_hold_metadata
			WHERE source = ? AND hole_id IN ({placeholders})
			""",
			[source, *hole_ids],
		)

		metadata_by_hole: dict[int, ExternalHoldMetadata] = {}
		for hole_id, external_hold_id, hold_type, hold_function, depth, orientation, texture, size, src in rows:
			metadata_by_hole[int(hole_id)] = ExternalHoldMetadata(
				hole_id=int(hole_id),
				external_hold_id=int(external_hold_id),
				type=str(hold_type) if hold_type is not None else None,
				function=str(hold_function) if hold_function is not None else None,
				depth=int(depth) if depth is not None else None,
				orientation=int(orientation) if orientation is not None else None,
				texture=int(texture) if texture is not None else None,
				size=int(size) if size is not None else None,
				source=str(src),
			)
		return metadata_by_hole

	def get_hold_positions_for_climb(
		self,
		climb_uuid: str | Climb,
		*,
		include_metadata: bool = True,
		metadata_source: str = "kilter_board_csv",
		metadata_product_id: int = 1,
	) -> list[HoldPlacement]:
		climb_obj = climb_uuid if isinstance(climb_uuid, Climb) else self.get_climb(climb_uuid)
		if climb_obj is None:
			return []

		frames = climb_obj.frames or ""
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
			f"SELECT id, x, y, product_id FROM holes WHERE id IN ({placeholders});",
			hole_ids,
		)
		hole_to_xy = {int(hid): (int(x), int(y)) for hid, x, y, _product_id in hole_rows}
		hole_to_product_id = {int(hid): int(product_id) for hid, _x, _y, product_id in hole_rows}

		metadata_by_hole: dict[int, ExternalHoldMetadata] = {}
		if include_metadata and hole_ids:
			all_match_product = all(
				hole_to_product_id.get(hole_id) == metadata_product_id
				for hole_id in hole_ids
			)
			if all_match_product:
				metadata_by_hole = self._fetch_external_hold_metadata(
					hole_ids,
					source=metadata_source,
				)

		role_map = self._role_map()

		holds: list[HoldPlacement] = []
		for placement_id, role_id in pairs:
			hole_id = placement_to_hole.get(placement_id)
			coords = hole_to_xy.get(hole_id) if hole_id is not None else None
			x, y = coords if coords else (None, None)
			metadata = metadata_by_hole.get(hole_id) if hole_id is not None else None
			holds.append(
				HoldPlacement(
					placement_id=placement_id,
					role_id=role_id,
					hole_id=hole_id,
					x=x,
					y=y,
					role_name=role_map.get(role_id),
					metadata=metadata,
				)
			)

		if isinstance(climb_uuid, Climb):
			climb_uuid.holds = holds
		return holds

	def get_climb_stats(self, climb_uuid: str) -> dict[str, Any]:
		"""Return quality_average, ascensionist_count, and FA info for a climb UUID.

		Returns a dict with keys:
		  quality_average     – float in [0, 4] or None
		  ascensionist_count  – int or None
		  fa_username         – str or None (first-ascent setter)
		  fa_at               – str or None (first-ascent timestamp)
		"""
		rows = self.fetchall(
			"SELECT quality_average, ascensionist_count, fa_username, fa_at, difficulty_average "
			"FROM climb_stats WHERE climb_uuid = ?",
			[climb_uuid],
		)
		if not rows:
			return {
				"quality_average": None,
				"ascensionist_count": None,
				"fa_username": None,
				"fa_at": None,
				"difficulty_average": None,
			}
		quality, ascents, fa_user, fa_at, difficulty = rows[0]
		return {
			"quality_average": float(quality) if quality is not None else None,
			"ascensionist_count": int(ascents) if ascents is not None else None,
			"fa_username": str(fa_user) if fa_user else None,
			"fa_at": str(fa_at) if fa_at else None,
			"difficulty_average": float(difficulty) if difficulty is not None else None,
		}

	def import_external_hold_metadata(
		self,
		csv_path: str | Path,
		*,
		source: str = "kilter_board_csv",
		product_id: int = 1,
		create_table: bool = True,
	) -> dict[str, Any]:
		"""Import external hold metadata CSV and map rows to ``holes.id``."""
		conn = self.connect()
		return import_kilter_board_csv_metadata(
			conn,
			csv_path,
			source=source,
			product_id=product_id,
			create_table=create_table,
		)


__all__ = ["BoardLibInterface", "Climb", "HoldPlacement", "ExternalHoldMetadata"]