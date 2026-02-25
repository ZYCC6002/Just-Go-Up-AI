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
			"SELECT uuid, name, frames, edge_left, edge_right, edge_bottom, edge_top FROM climbs WHERE uuid = ?",
			[climb_uuid],
		).fetchone()
		if row is None:
			return None
		keys = ["uuid", "name", "frames", "edge_left", "edge_right", "edge_bottom", "edge_top"]
		return dict(zip(keys, row))

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
