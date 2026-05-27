from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import math
import random

import numpy as np
import torch

from database_interfaces.board_lib_interface import BoardLibInterface, HoldPlacement
from model_training.route_vae_encoder import RouteVAEEncoderConfig


UNKNOWN_TOKEN = "<UNK>"

# ---------------------------------------------------------------------------
# Grip category mapping
# ---------------------------------------------------------------------------
# Maps the 5 Kilter hold types to grip-feel categories. Each type gets its
# own category (jug and sloper are NOT merged) so the model can distinguish
# juggy pump routes from sloper balance routes.
_GRIP_CATEGORY: dict[str, str] = {
    "jug":    "jug",        # open-hand, pulling-dominant
    "sloper": "sloper",     # open-hand, friction/balance-dominant
    "crimp":  "crimp",
    "pinch":  "pinch",
    "foot":   "foot_type",  # hold physically designed for feet (vs. role=foot)
}
_GRIP_CATEGORY_DEFAULT = "unknown"


def _extract_grip_category(type_name: str) -> str:
    return _GRIP_CATEGORY.get(type_name, _GRIP_CATEGORY_DEFAULT)


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
    grip_category_vocab: CategoricalVocab

    def to_transformer_config(self, **overrides: Any) -> RouteVAEEncoderConfig:
        grip_vocab = getattr(self, "grip_category_vocab", None)
        base = dict(
            type_vocab_size=self.type_vocab.size,
            function_vocab_size=self.function_vocab.size,
            role_vocab_size=self.role_vocab.size,
            hole_id_vocab_size=self.hole_vocab.size,
            grip_category_vocab_size=grip_vocab.size if grip_vocab is not None else 0,
        )
        base.update(overrides)
        return RouteVAEEncoderConfig(**base)


@dataclass
class RawHoldToken:
    role_id: int
    role_name: str | None
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
            role_name=hold.role_name,
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
        role_name=hold.role_name,
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


def _is_foot_hold(tok: RawHoldToken) -> bool:
    """True if this hold is placed for feet (by role name)."""
    return "foot" in (tok.role_name or "").lower()


def _encode_route_tokens(route_tokens: list[RawHoldToken], vocabs: RouteVocabBundle) -> dict[str, torch.Tensor]:
    # --- Sort holds bottom-to-top (ascending y = canonical climbing sequence) ---
    route_tokens = sorted(route_tokens, key=lambda t: t.y)

    # Categorical features
    type_ids = [vocabs.type_vocab.encode(tok.type_name) for tok in route_tokens]
    function_ids = [vocabs.function_vocab.encode(tok.function_name) for tok in route_tokens]
    role_ids = [vocabs.role_vocab.encode(tok.role_id) for tok in route_tokens]
    hole_ids = [vocabs.hole_vocab.encode(tok.hole_id) for tok in route_tokens]

    # Grip category (per-hold)
    grip_vocab = getattr(vocabs, "grip_category_vocab", None)
    grip_category_ids = (
        [grip_vocab.encode(_extract_grip_category(tok.type_name)) for tok in route_tokens]
        if grip_vocab is not None
        else []
    )

    orientation_pairs = [_orientation_to_sin_cos(tok.orientation_deg) for tok in route_tokens]
    orientation_sin = [p[0] for p in orientation_pairs]
    orientation_cos = [p[1] for p in orientation_pairs]

    xs = [tok.x for tok in route_tokens]
    ys = [tok.y for tok in route_tokens]

    # --- Delta features (move from previous hold in y-sorted sequence) ---
    delta_x_prev = [0.0] + [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    delta_y_prev = [0.0] + [ys[i] - ys[i - 1] for i in range(1, len(ys))]

    # --- Nearest-neighbour distance (encoder-only; requires full sequence) ---
    if len(xs) >= 2:
        coords = np.array(list(zip(xs, ys)), dtype=np.float32)
        diffs = coords[:, None, :] - coords[None, :, :]   # [N, N, 2]
        dists_mat = np.sqrt((diffs ** 2).sum(axis=-1))     # [N, N]
        np.fill_diagonal(dists_mat, np.inf)
        dist_to_nearest = dists_mat.min(axis=1).tolist()
    else:
        dist_to_nearest = [0.0] * len(route_tokens)

    # --- Route-level shape descriptor [9] ---
    n_total = max(len(route_tokens), 1)
    xs_arr = np.array(xs, dtype=np.float32)
    ys_arr = np.array(ys, dtype=np.float32)
    x_std = float(xs_arr.std()) if len(xs_arr) > 1 else 0.0
    y_std = float(ys_arr.std()) if len(ys_arr) > 1 else 0.0

    # Use role_name to count foot holds (role is how the hold is used in this route)
    n_foot = sum(1 for t in route_tokens if _is_foot_hold(t))
    n_hand = n_total - n_foot
    foot_fraction   = n_foot / n_total
    hand_count_norm = min(n_hand / 20.0, 1.0)  # 20 hand holds ≈ "many" (endurance signal)

    # Per-type fractions — separate jug vs sloper so juggy and sloper routes are distinguishable
    jug_frac    = sum(1 for t in route_tokens if t.type_name == "jug")    / n_total
    sloper_frac = sum(1 for t in route_tokens if t.type_name == "sloper") / n_total
    crimp_frac  = sum(1 for t in route_tokens if t.type_name == "crimp")  / n_total
    pinch_frac  = sum(1 for t in route_tokens if t.type_name == "pinch")  / n_total

    # Mean pairwise distance between hand holds (deadpoint/reachy vs technical)
    hand_coords = np.array(
        [(t.x, t.y) for t in route_tokens if not _is_foot_hold(t)],
        dtype=np.float32,
    )
    if len(hand_coords) >= 2:
        hc_diffs = hand_coords[:, None, :] - hand_coords[None, :, :]
        hc_pairwise = np.sqrt((hc_diffs ** 2).sum(axis=-1))
        mean_move = float(hc_pairwise[np.triu_indices(len(hand_coords), k=1)].mean())
    else:
        mean_move = 0.0
    mean_move_norm = min(mean_move / 100.0, 1.0)  # 100 board units ≈ "very large move"

    shape_desc = torch.tensor(
        [x_std, y_std, foot_fraction, hand_count_norm,
         jug_frac, sloper_frac, crimp_frac, pinch_frac, mean_move_norm],
        dtype=torch.float32,
    )  # shape [9]

    result: dict[str, torch.Tensor] = {
        "type_encoded_id":   torch.tensor(type_ids, dtype=torch.long),
        "function_encoded_id": torch.tensor(function_ids, dtype=torch.long),
        "role_encoded_id":   torch.tensor(role_ids, dtype=torch.long),
        "hole_encoded_id":   torch.tensor(hole_ids, dtype=torch.long),
        "x":                 torch.tensor(xs, dtype=torch.float32),
        "y":                 torch.tensor(ys, dtype=torch.float32),
        "depth":             torch.tensor([tok.depth for tok in route_tokens], dtype=torch.float32),
        "orientation_sin":   torch.tensor(orientation_sin, dtype=torch.float32),
        "orientation_cos":   torch.tensor(orientation_cos, dtype=torch.float32),
        "size":              torch.tensor([tok.size for tok in route_tokens], dtype=torch.float32),
        # New per-hold features
        "delta_x_prev":      torch.tensor(delta_x_prev, dtype=torch.float32),
        "delta_y_prev":      torch.tensor(delta_y_prev, dtype=torch.float32),
        "dist_to_nearest":   torch.tensor(dist_to_nearest, dtype=torch.float32),
        # Route-level descriptor (special: shape [9], not [L])
        "shape_desc":        shape_desc,
    }
    if grip_category_ids:
        result["grip_category_id"] = torch.tensor(grip_category_ids, dtype=torch.long)

    return result


def _load_raw_routes(
    db: BoardLibInterface,
    *,
    require_full_metadata: bool,
    metadata_source: str,
    metadata_product_id: int,
    max_routes: int | None,
    min_holds: int,
    min_quality_average: float,
    min_ascensionist_count: int,
    public_only: bool,
) -> list[tuple[dict[str, Any], list[RawHoldToken], float]]:
    """Load climbs and convert each hold to raw token records.

    Returns tuples of:
    - climb row dict
    - raw hold tokens
    - metadata coverage ratio [0,1]
    - only climbs on product_id=1 (Kilter Board Original) are kept
    """
    required_product_id = 1

    rows = db.fetchall(
        """
        SELECT
            c.uuid,
            c.name,
            c.layout_id,
            cs.angle,
            c.is_listed,
            cs.difficulty_average,
            cs.quality_average,
            cs.ascensionist_count
        FROM climbs c
        LEFT JOIN climb_stats cs ON c.uuid = cs.climb_uuid
        WHERE c.frames IS NOT NULL AND c.frames != ''
        """
    )

    raw_routes: list[tuple[dict[str, Any], list[RawHoldToken], float]] = []
    for uuid, name, layout_id, angle, is_listed, difficulty_average, quality_average, ascensionist_count in rows:
        # Require conditioning features to be present.
        # Do not coerce missing angle/grade to defaults.
        if public_only and int(is_listed or 0) != 1:
            continue
        if angle is None or difficulty_average is None:
            continue
        if quality_average is None or float(quality_average) < float(min_quality_average):
            continue
        if ascensionist_count is None or int(ascensionist_count) < int(min_ascensionist_count):
            continue

        climb = db.get_climb(str(uuid))
        if climb is None or climb.product_id != required_product_id:
            continue

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
            "angle": float(angle),
            "grade": float(difficulty_average),
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
    min_quality_average: float = 2.8,
    min_ascensionist_count: int = 100,
    public_only: bool = True,
) -> tuple[list[RouteSample], RouteVocabBundle]:
    """Prepare train-ready route samples and vocabularies from SQLite.

    - Handles missing metadata (with UNKNOWN defaults) unless full metadata is required.
    - Converts orientation degrees to explicit sin/cos inputs.
    - Filters low-signal climbs by `quality_average` and `ascensionist_count`.
    - Keeps only climbs on product_id=1 (Kilter Board Original).
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
            min_quality_average=min_quality_average,
            min_ascensionist_count=min_ascensionist_count,
            public_only=public_only,
        )

    if not raw_routes:
        raise ValueError("No routes matched preprocessing filters.")

    all_types: list[str] = []
    all_functions: list[str] = []
    all_roles: list[int] = []
    all_holes: list[int] = []
    all_grip_categories: list[str] = []
    for _climb, tokens, _coverage in raw_routes:
        all_types.extend(tok.type_name for tok in tokens)
        all_functions.extend(tok.function_name for tok in tokens)
        all_roles.extend(tok.role_id for tok in tokens)
        all_holes.extend(tok.hole_id for tok in tokens)
        all_grip_categories.extend(_extract_grip_category(tok.type_name) for tok in tokens)

    vocabs = RouteVocabBundle(
        type_vocab=_build_vocab(all_types, include_unknown=True),
        function_vocab=_build_vocab(all_functions, include_unknown=True),
        role_vocab=_build_vocab(all_roles, include_unknown=False),
        hole_vocab=_build_vocab(all_holes, include_unknown=False),
        grip_category_vocab=_build_vocab(all_grip_categories, include_unknown=True),
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
