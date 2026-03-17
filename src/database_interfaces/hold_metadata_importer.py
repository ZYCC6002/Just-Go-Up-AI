"""Import external hold metadata and map it to BoardLib hole IDs."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any


def _ensure_external_table(conn: sqlite3.Connection) -> None:
	conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS external_hold_metadata (
			source TEXT NOT NULL,
			external_hold_id INTEGER NOT NULL,
			hole_id INTEGER NOT NULL,
			type TEXT,
			function TEXT,
			depth INTEGER,
			orientation INTEGER,
			texture INTEGER,
			size INTEGER,
			PRIMARY KEY (source, hole_id),
			FOREIGN KEY (hole_id) REFERENCES holes(id)
		)
		"""
	)


def import_kilter_board_csv_metadata(
	conn: sqlite3.Connection,
	csv_path: str | Path,
	*,
	source: str = "kilter_board_csv",
	product_id: int = 1,
	create_table: bool = True,
) -> dict[str, Any]:
	"""Import external Kilter board hold metadata from CSV into DB.

	Mapping used:
	- db_x = x_coordinate * 4 + 4
	- db_y = y_coordinate * 4 + 4

	Returns a summary dict with inserted row counts and mapping diagnostics.
	"""
	if create_table:
		_ensure_external_table(conn)

	hole_rows = conn.execute(
		"SELECT id, x, y FROM holes WHERE product_id = ?",
		[product_id],
	).fetchall()
	coord_to_hole_id = {(int(x), int(y)): int(hole_id) for hole_id, x, y in hole_rows}

	resolved_rows: list[tuple[Any, ...]] = []
	unmatched: list[tuple[int, int, int]] = []
	used_hole_ids: set[int] = set()
	duplicate_hole_ids: set[int] = set()

	path = Path(csv_path)
	with path.open("r", newline="", encoding="utf-8") as fp:
		reader = csv.DictReader(fp)
		required = {
			"hold_id",
			"x_coordinate",
			"y_coordinate",
			"type",
			"function",
			"depth",
			"orientation",
			"texture",
			"size",
		}
		if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
			raise ValueError(
				"CSV missing required columns. "
				f"Expected at least: {sorted(required)}; got: {reader.fieldnames}"
			)

		for row in reader:
			external_hold_id = int(row["hold_id"])
			x_coordinate = int(row["x_coordinate"])
			y_coordinate = int(row["y_coordinate"])

			db_x = x_coordinate * 4 + 4
			db_y = y_coordinate * 4 + 4
			hole_id = coord_to_hole_id.get((db_x, db_y))

			if hole_id is None:
				unmatched.append((external_hold_id, db_x, db_y))
				continue

			if hole_id in used_hole_ids:
				duplicate_hole_ids.add(hole_id)
			used_hole_ids.add(hole_id)

			resolved_rows.append(
				(
					source,
					external_hold_id,
					hole_id,
					row["type"],
					row["function"],
					int(row["depth"]),
					int(row["orientation"]),
					int(row["texture"]),
					int(row["size"]),
				)
			)

	if unmatched:
		example = unmatched[:5]
		raise ValueError(
			f"{len(unmatched)} CSV rows could not be mapped to holes for product_id={product_id}. "
			f"Example (external_hold_id, mapped_x, mapped_y): {example}"
		)

	if duplicate_hole_ids:
		example = sorted(duplicate_hole_ids)[:10]
		raise ValueError(
			f"Duplicate hole mappings detected for {len(duplicate_hole_ids)} hole_id values. "
			f"Example hole_ids: {example}"
		)

	conn.executemany(
		"""
		INSERT OR REPLACE INTO external_hold_metadata
		(source, external_hold_id, hole_id, type, function, depth, orientation, texture, size)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		""",
		resolved_rows,
	)
	conn.commit()

	return {
		"source": source,
		"csv_path": str(path),
		"csv_rows": len(resolved_rows),
		"inserted_or_replaced": len(resolved_rows),
		"product_id": product_id,
	}


__all__ = ["import_kilter_board_csv_metadata"]
