from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import math
import random

import torch

from database_interfaces.board_lib_interface import BoardLibInterface, HoldPlacement
from model_training.route_transformer_encoder import RouteTransformerConfig


UNKNOWN_TOKEN = "<UNK>"


@dataclass
class CategoricalVocab:
	item_to_idx: dict[Any, int]
	idx_to_item: list[Any]

	@property
	def size(self) -> int:
		return len(self.idx_to_item)

	def encode(self, item: Any) -> int:
		if item in self.item_to_idx:
			return self.item_to_idx[item]
		if UNKNOWN_TOKEN in self.item_to_idx:
			return self.item_to_idx[UNKNOWN_TOKEN]
		return 0


@dataclass
class RouteVocabBundle:
	type_vocab: CategoricalVocab
	function_vocab: CategoricalVocab
	role_vocab: CategoricalVocab
	hole_vocab: CategoricalVocab

	def to_transformer_config(self, **overrides: Any) -> RouteTransformerConfig:
		base = dict(
			type_vocab_size=self.type_vocab.size,
			function_vocab_size=self.function_vocab.size,
			role_vocab_size=self.role_vocab.size,
			hole_id_vocab_size=self.hole_vocab.size,
		)
		base.update(overrides)
		return RouteTransformerConfig(**base)


@dataclass
class RawHoldToken:
	role_id: int
	hole_id: int
	x: float
	y: float
	type_name: str
	function_name: str
	depth: float
	orientation_deg: float
	size: float
	metadata_available: bool


@dataclass
class RouteSample:
	uuid: str
	name: str
	angle: float
	grade: float | None
	layout_id: int
	num_holds: int
	metadata_coverage: float
	tokens: dict[str, torch.Tensor]


def _build_vocab(items: Iterable[Any], *, include_unknown: bool) -> CategoricalVocab:
	unique_items = sorted(set(items))
	idx_to_item: list[Any] = []
	if include_unknown:
		idx_to_item.append(UNKNOWN_TOKEN)
	idx_to_item.extend(unique_items)
	item_to_idx = {item: idx for idx, item in enumerate(idx_to_item)}
	return CategoricalVocab(item_to_idx=item_to_idx, idx_to_item=idx_to_item)


def _extract_hold_token(hold: HoldPlacement) -> RawHoldToken | None:
	if hold.hole_id is None or hold.x is None or hold.y is None:
		return None

	if hold.metadata is None:
		return RawHoldToken(
			role_id=int(hold.role_id),
			hole_id=int(hold.hole_id),
			x=float(hold.x),
			y=float(hold.y),
			type_name=UNKNOWN_TOKEN,
			function_name=UNKNOWN_TOKEN,
			depth=0.0,
			orientation_deg=0.0,
			size=2.0,
			metadata_available=False,
		)

	md = hold.metadata
	return RawHoldToken(
		role_id=int(hold.role_id),
		hole_id=int(hold.hole_id),
		x=float(hold.x),
		y=float(hold.y),
		type_name=str(md.type) if md.type else UNKNOWN_TOKEN,
		function_name=str(md.function) if md.function else UNKNOWN_TOKEN,
		depth=float(md.depth if md.depth is not None else 0),
		orientation_deg=float(md.orientation if md.orientation is not None else 0),
		size=float(md.size if md.size is not None else 2),
		metadata_available=True,
	)


def _orientation_to_sin_cos(angle_deg: float) -> tuple[float, float]:
	radians = math.radians(angle_deg)
	return math.sin(radians), math.cos(radians)


def _encode_route_tokens(route_tokens: list[RawHoldToken], vocabs: RouteVocabBundle) -> dict[str, torch.Tensor]:
	type_ids = [vocabs.type_vocab.encode(tok.type_name) for tok in route_tokens]
	function_ids = [vocabs.function_vocab.encode(tok.function_name) for tok in route_tokens]
	role_ids = [vocabs.role_vocab.encode(tok.role_id) for tok in route_tokens]
	hole_ids = [vocabs.hole_vocab.encode(tok.hole_id) for tok in route_tokens]

	orientation_pairs = [_orientation_to_sin_cos(tok.orientation_deg) for tok in route_tokens]
	orientation_sin = [p[0] for p in orientation_pairs]
	orientation_cos = [p[1] for p in orientation_pairs]

	return {
		"type_id": torch.tensor(type_ids, dtype=torch.long),
		"function_id": torch.tensor(function_ids, dtype=torch.long),
		"role_id": torch.tensor(role_ids, dtype=torch.long),
		"hole_id": torch.tensor(hole_ids, dtype=torch.long),
		"x": torch.tensor([tok.x for tok in route_tokens], dtype=torch.float32),
		"y": torch.tensor([tok.y for tok in route_tokens], dtype=torch.float32),
		"depth": torch.tensor([tok.depth for tok in route_tokens], dtype=torch.float32),
		"orientation_sin": torch.tensor(orientation_sin, dtype=torch.float32),
		"orientation_cos": torch.tensor(orientation_cos, dtype=torch.float32),
		"size": torch.tensor([tok.size for tok in route_tokens], dtype=torch.float32),
	}


def _load_raw_routes(
	db: BoardLibInterface,
	*,
	require_full_metadata: bool,
	metadata_source: str,
	metadata_product_id: int,
	max_routes: int | None,
	min_holds: int,
) -> list[tuple[dict[str, Any], list[RawHoldToken], float]]:
	"""Load climbs and convert each hold to raw token records.

	Returns tuples of:
	- climb row dict
	- raw hold tokens
	- metadata coverage ratio [0,1]
	"""

	rows = db.fetchall(
		"""
		SELECT c.uuid, c.name, c.layout_id, c.angle, cs.difficulty_average
		FROM climbs c
		LEFT JOIN climb_stats cs ON c.uuid = cs.climb_uuid
		WHERE c.frames IS NOT NULL AND c.frames != ''
		"""
	)

	raw_routes: list[tuple[dict[str, Any], list[RawHoldToken], float]] = []
	for uuid, name, layout_id, angle, difficulty_average in rows:
		holds = db.get_hold_positions_for_climb(
			str(uuid),
			include_metadata=True,
			metadata_source=metadata_source,
			metadata_product_id=metadata_product_id,
		)
		tokens = [tok for tok in (_extract_hold_token(h) for h in holds) if tok is not None]
		if len(tokens) < min_holds:
			continue

		meta_count = sum(1 for tok in tokens if tok.metadata_available)
		coverage = meta_count / len(tokens) if tokens else 0.0
		if require_full_metadata and coverage < 1.0:
			continue

		climb_info = {
			"uuid": str(uuid),
			"name": str(name),
			"layout_id": int(layout_id),
			"angle": float(angle) if angle is not None else 0.0,
			"grade": float(difficulty_average) if difficulty_average is not None else None,
		}
		raw_routes.append((climb_info, tokens, coverage))

		if max_routes is not None and len(raw_routes) >= max_routes:
			break

	return raw_routes


def build_training_samples_from_db(
	db_path: str,
	*,
	require_full_metadata: bool = False,
	metadata_source: str = "kilter_board_csv",
	metadata_product_id: int = 1,
	max_routes: int | None = None,
	min_holds: int = 1,
) -> tuple[list[RouteSample], RouteVocabBundle]:
	"""Prepare train-ready route samples and vocabularies from SQLite.

	- Handles missing metadata (with UNKNOWN defaults) unless full metadata is required.
	- Converts orientation degrees to explicit sin/cos inputs.
	- Returns token tensors ready for `collate_hold_token_batch`.
	"""

	with BoardLibInterface(db_path) as db:
		raw_routes = _load_raw_routes(
			db,
			require_full_metadata=require_full_metadata,
			metadata_source=metadata_source,
			metadata_product_id=metadata_product_id,
			max_routes=max_routes,
			min_holds=min_holds,
		)

	if not raw_routes:
		raise ValueError("No routes matched preprocessing filters.")

	all_types: list[str] = []
	all_functions: list[str] = []
	all_roles: list[int] = []
	all_holes: list[int] = []
	for _climb, tokens, _coverage in raw_routes:
		all_types.extend(tok.type_name for tok in tokens)
		all_functions.extend(tok.function_name for tok in tokens)
		all_roles.extend(tok.role_id for tok in tokens)
		all_holes.extend(tok.hole_id for tok in tokens)

	vocabs = RouteVocabBundle(
		type_vocab=_build_vocab(all_types, include_unknown=True),
		function_vocab=_build_vocab(all_functions, include_unknown=True),
		role_vocab=_build_vocab(all_roles, include_unknown=False),
		hole_vocab=_build_vocab(all_holes, include_unknown=False),
	)

	samples: list[RouteSample] = []
	for climb_info, tokens, coverage in raw_routes:
		encoded = _encode_route_tokens(tokens, vocabs)
		samples.append(
			RouteSample(
				uuid=climb_info["uuid"],
				name=climb_info["name"],
				angle=climb_info["angle"],
				grade=climb_info["grade"],
				layout_id=climb_info["layout_id"],
				num_holds=len(tokens),
				metadata_coverage=coverage,
				tokens=encoded,
			)
		)

	return samples, vocabs


def split_route_samples(
	samples: list[RouteSample],
	*,
	train_ratio: float = 0.8,
	val_ratio: float = 0.1,
	seed: int = 42,
) -> tuple[list[RouteSample], list[RouteSample], list[RouteSample]]:
	if not 0 < train_ratio < 1:
		raise ValueError("train_ratio must be in (0, 1).")
	if not 0 <= val_ratio < 1:
		raise ValueError("val_ratio must be in [0, 1).")
	if train_ratio + val_ratio >= 1:
		raise ValueError("train_ratio + val_ratio must be < 1.")

	indices = list(range(len(samples)))
	rng = random.Random(seed)
	rng.shuffle(indices)

	n = len(indices)
	n_train = int(n * train_ratio)
	n_val = int(n * val_ratio)

	train_idx = indices[:n_train]
	val_idx = indices[n_train:n_train + n_val]
	test_idx = indices[n_train + n_val:]

	train = [samples[i] for i in train_idx]
	val = [samples[i] for i in val_idx]
	test = [samples[i] for i in test_idx]
	return train, val, test


__all__ = [
	"CategoricalVocab",
	"RouteVocabBundle",
	"RawHoldToken",
	"RouteSample",
	"build_training_samples_from_db",
	"split_route_samples",
]
