from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import math
import random

import numpy as np
import torch

from database_interfaces.board_lib_interface import BoardLibInterface, Climb, HoldPlacement
from model_training.route_vae_encoder import RouteVAEEncoderConfig


UNKNOWN_TOKEN = "<UNK>"

# Fallback coordinate spans used when select_product_size cannot determine the board
# size for a specific route (e.g. empty set_ids or no fitting product_size row).
# Measured from the holes table for product_id=1 (Kilter Board Original):
#   SELECT MIN(x), MAX(x), MIN(y), MAX(y) FROM holes WHERE product_id=1
#   → x ∈ [-20, 164] (span 184), y ∈ [4, 176] (span 172).
_FALLBACK_BOARD_X_SPAN: float = 184.0
_FALLBACK_BOARD_Y_SPAN: float = 172.0


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

    def to_transformer_config(self, **overrides: Any) -> RouteVAEEncoderConfig:
        base = dict(
            type_vocab_size=self.type_vocab.size,
            function_vocab_size=self.function_vocab.size,
            role_vocab_size=self.role_vocab.size,
            hole_id_vocab_size=self.hole_vocab.size,
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
    # All (angle → difficulty_average) pairs from climb_stats that passed quality/
    # ascensionist filters for this route.  Used by the per-angle grade adversary to
    # supply supervision signals at every angle where community grade data exists.
    angle_grades: dict[float, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.angle_grades is None:
            self.angle_grades = {}


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


def _encode_route_tokens(
    route_tokens: list[RawHoldToken],
    vocabs: RouteVocabBundle,
    *,
    board_x_span: float = _FALLBACK_BOARD_X_SPAN,
    board_y_span: float = _FALLBACK_BOARD_Y_SPAN,
) -> dict[str, torch.Tensor]:
    # --- Sort holds bottom-to-top (ascending y = canonical climbing sequence) ---
    route_tokens = sorted(route_tokens, key=lambda t: t.y)

    # Categorical features
    type_ids = [vocabs.type_vocab.encode(tok.type_name) for tok in route_tokens]
    function_ids = [vocabs.function_vocab.encode(tok.function_name) for tok in route_tokens]
    role_ids = [vocabs.role_vocab.encode(tok.role_id) for tok in route_tokens]
    hole_ids = [vocabs.hole_vocab.encode(tok.hole_id) for tok in route_tokens]

    orientation_pairs = [_orientation_to_sin_cos(tok.orientation_deg) for tok in route_tokens]
    orientation_sin = [p[0] for p in orientation_pairs]
    orientation_cos = [p[1] for p in orientation_pairs]

    xs = [tok.x for tok in route_tokens]
    ys = [tok.y for tok in route_tokens]

    # Board diagonal — used as the reference scale for distance-based features.
    # Half-diagonal (~"cross-board move") serves as the upper bound for mean_move_norm.
    board_diag = math.sqrt(board_x_span ** 2 + board_y_span ** 2)

    # --- Delta features (move from previous hold in y-sorted sequence) ---
    # Normalised by board span so values are in roughly [-1, 1] / [0, 1] regardless
    # of which physical board size a route was set on.
    delta_x_raw = [0.0] + [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    delta_y_raw = [0.0] + [ys[i] - ys[i - 1] for i in range(1, len(ys))]
    delta_x_prev = [dx / board_x_span for dx in delta_x_raw]
    delta_y_prev = [dy / board_y_span for dy in delta_y_raw]

    # --- Nearest-neighbour distance (encoder-only; requires full sequence) ---
    # Normalised by board diagonal → [0, 1] for any realistic board size.
    if len(xs) >= 2:
        coords = np.array(list(zip(xs, ys)), dtype=np.float32)
        diffs = coords[:, None, :] - coords[None, :, :]   # [N, N, 2]
        dists_mat = np.sqrt((diffs ** 2).sum(axis=-1))     # [N, N]
        np.fill_diagonal(dists_mat, np.inf)
        dist_to_nearest = (dists_mat.min(axis=1) / board_diag).tolist()
    else:
        dist_to_nearest = [0.0] * len(route_tokens)

    # --- Route-level shape descriptor [9] ---
    n_total = max(len(route_tokens), 1)
    xs_arr = np.array(xs, dtype=np.float32)
    ys_arr = np.array(ys, dtype=np.float32)
    # Normalise by the board coordinate span so shape descriptor features are
    # on the same [0, ~0.5] scale as the fraction-based features.
    # board_x_span / board_y_span come from select_product_size (the smallest
    # product_size that fits this route's hold extents), falling back to the
    # global hole span for product_id=1 if the lookup fails.
    x_std = (float(xs_arr.std()) if len(xs_arr) > 1 else 0.0) / board_x_span
    y_std = (float(ys_arr.std()) if len(ys_arr) > 1 else 0.0) / board_y_span

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
    # Normalise by half the board diagonal: a move spanning half the diagonal is
    # already a very large reach regardless of board size.  Clamp at 1.0.
    mean_move_norm = min(mean_move / (board_diag * 0.5), 1.0)

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

    The DB join (climbs × climb_stats) produces one row per (UUID, angle).
    Since the encoder does not see angle, all angle variants of the same route
    produce identical z vectors — keeping them inflates training data with
    redundant examples and biases gradient updates toward popular routes.
    This function deduplicates to one entry per UUID by retaining the
    (UUID, angle) pair with the highest quality_average × ascensionist_count.
    Hold tokens are loaded only for the selected row, not for every angle
    variant, so this is also more efficient than loading all duplicates.
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

    # Pass 1 — row-level filters and UUID deduplication (no hold loading).
    # Cache Climb objects so each UUID is fetched from the DB at most once
    # (used for product_id check here and for board-span lookup in Pass 2).
    _climb_cache: dict[str, Climb | None] = {}
    best_per_uuid: dict[str, tuple] = {}           # uuid_str -> (score, row_tuple)
    angle_grades_per_uuid: dict[str, dict[float, float]] = {}  # uuid_str -> {angle: grade}

    for row in rows:
        uuid, name, layout_id, angle, is_listed, difficulty_average, quality_average, ascensionist_count = row
        if public_only and int(is_listed or 0) != 1:
            continue
        if angle is None or difficulty_average is None:
            continue
        if quality_average is None or float(quality_average) < float(min_quality_average):
            continue
        if ascensionist_count is None or int(ascensionist_count) < int(min_ascensionist_count):
            continue

        uuid_str = str(uuid)
        if uuid_str not in _climb_cache:
            _climb_cache[uuid_str] = db.get_climb(uuid_str)
        climb = _climb_cache[uuid_str]
        if climb is None or climb.product_id != required_product_id:
            continue

        # Collect all (angle → grade) pairs that pass quality filters.
        # The per-angle grade adversary uses these as supervision targets.
        if uuid_str not in angle_grades_per_uuid:
            angle_grades_per_uuid[uuid_str] = {}
        angle_grades_per_uuid[uuid_str][float(angle)] = float(difficulty_average)

        score = float(quality_average) * int(ascensionist_count)
        if uuid_str not in best_per_uuid or score > best_per_uuid[uuid_str][0]:
            best_per_uuid[uuid_str] = (score, row)

    # Pass 2 — load hold tokens only for the selected (best) row per UUID.
    raw_routes: list[tuple[dict[str, Any], list[RawHoldToken], float]] = []
    for uuid_str, (_score, row) in best_per_uuid.items():
        uuid, name, layout_id, angle, is_listed, difficulty_average, quality_average, ascensionist_count = row

        holds = db.get_hold_positions_for_climb(
            uuid_str,
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

        # Resolve the board size that fits this route's hold extents.
        # select_product_size chooses the smallest product_size whose edges
        # fully contain the route and that has images for all required sets.
        board_x_span = _FALLBACK_BOARD_X_SPAN
        board_y_span = _FALLBACK_BOARD_Y_SPAN
        climb = _climb_cache.get(uuid_str)
        if climb is not None and climb.product_id is not None:
            try:
                set_ids = db.decode_hsm_set_ids(int(climb.layout_id), int(climb.hsm or 0))
                if set_ids:
                    _, board_edges = db.select_product_size(
                        int(climb.layout_id),
                        set_ids,
                        climb.edges,
                        product_id=int(climb.product_id),
                    )
                    # board_edges = (edge_left, edge_right, edge_bottom, edge_top)
                    board_x_span = float(board_edges[1] - board_edges[0])
                    board_y_span = float(board_edges[3] - board_edges[2])
            except (ValueError, Exception):
                pass  # keep fallback constants; not worth skipping the route

        climb_info = {
            "uuid": uuid_str,
            "name": str(name),
            "layout_id": int(layout_id),
            "angle": float(angle),
            "grade": float(difficulty_average),
            "angle_grades": angle_grades_per_uuid.get(uuid_str, {}),
            "board_x_span": board_x_span,
            "board_y_span": board_y_span,
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
        encoded = _encode_route_tokens(
            tokens,
            vocabs,
            board_x_span=climb_info.get("board_x_span", _FALLBACK_BOARD_X_SPAN),
            board_y_span=climb_info.get("board_y_span", _FALLBACK_BOARD_Y_SPAN),
        )
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
                angle_grades=climb_info.get("angle_grades", {}),
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
