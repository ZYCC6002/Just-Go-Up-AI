"""Validate latent-space neighbor relationships against geometric ground truth.

`routes_cluster.py` produces a cache of standardised latent vectors (`z`); this
script asks the question "if two routes are close in `z`-space, are they
*actually* similar climbs?" It uses Earth Mover's Distance (Wasserstein-1)
between routes' hold-position point clouds as a permutation-invariant,
geometrically grounded similarity measure — independent of clustering method,
projection choice, or hand-picked aggregate features — and checks whether `z`
neighbors are also EMD neighbors.

Usage:
    uv run src/data_analysis/route_neighbor_analysis.py \
        --cluster-cache-path data/routes_new_model_kmeans.pt \
        --climb-name "undermine" --k 5

    uv run src/data_analysis/route_neighbor_analysis.py \
        --cluster-cache-path data/routes_new_model_kmeans.pt \
        --num-random-queries 5 --k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wasserstein_distance_nd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_analysis.style_analysis import FEATURE_LABELS, extract_features, load_cache

# Board spans used to normalise hold coordinates (x in [-20, 164], y in [4, 176];
# see CLAUDE.md "Per-hold token projection dims" / training normalisation notes).
_BOARD_X_SPAN = 184.0
_BOARD_Y_SPAN = 172.0

# Subset of interpretable style features worth cross-checking (skips grade,
# num_holds, max_move, knn_move_dist — less relevant or noisier for "style").
_STYLE_FEAT_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13]


def hold_cloud(sample) -> tuple[np.ndarray, np.ndarray]:
    """Return a route's holds as a uniform-weight point cloud over normalised (x, y)."""
    tok = sample.tokens
    x = tok["x"].numpy().astype(np.float64) / _BOARD_X_SPAN
    y = tok["y"].numpy().astype(np.float64) / _BOARD_Y_SPAN
    points = np.stack([x, y], axis=1)
    weights = np.full(len(points), 1.0 / len(points))
    return points, weights


def route_emd(a, b) -> float:
    """Earth Mover's Distance between two routes' hold-position distributions.

    Permutation-invariant (hold order doesn't matter) and hold-count-invariant
    (each route's mass sums to 1) — a direct "how differently is the wall used"
    measure, complementary to aggregate stats like x_spread/y_spread that only
    capture marginal moments, not the joint spatial layout.
    """
    pa, wa = hold_cloud(a)
    pb, wb = hold_cloud(b)
    return float(wasserstein_distance_nd(pa, pb, wa, wb))


def _resolve_named_indices(samples, names: list[str]) -> list[int]:
    idx = []
    for name in names:
        matches = [i for i, s in enumerate(samples) if name.lower() in s.name.lower()]
        if not matches:
            print(f"  Warning: no climb name matches '{name}'; skipping.")
            continue
        if len(matches) > 1:
            print(f"  '{name}' matched {len(matches)} climbs; using first: '{samples[matches[0]].name}'")
        idx.append(matches[0])
    return idx


def _feat_diff(features: np.ndarray, i: int, j: int) -> np.ndarray:
    return np.abs(features[i, _STYLE_FEAT_IDX] - features[j, _STYLE_FEAT_IDX])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate z-space neighbor relationships using Earth Mover's Distance "
                    "between routes' hold-position distributions."
    )
    parser.add_argument("--cluster-cache-path", type=str,
                        default=str(PROJECT_ROOT / "data/routes_clustered.pt"))
    parser.add_argument("--climb-name", type=str, action="append", default=None,
                        help="Query climb by (sub)name; repeatable. If omitted, picks random queries.")
    parser.add_argument("--num-random-queries", type=int, default=5)
    parser.add_argument("--k", type=int, default=3, help="Number of near/far neighbors to inspect per query.")
    parser.add_argument("--max-routes", type=int, default=1500,
                        help="Subsample the cache to this many routes (EMD is pairwise; keeps runtime reasonable).")
    parser.add_argument("--n-pairs", type=int, default=200,
                        help="Number of random pairs for the aggregate correlation check.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading cache: {args.cluster_cache_path}")
    samples, latent_matrix, _cluster_ids, _payload = load_cache(args.cluster_cache_path)

    # Resolve named queries against the FULL set first, so subsampling can't drop them.
    named_idx = _resolve_named_indices(samples, args.climb_name) if args.climb_name else []

    if args.max_routes is not None and len(samples) > args.max_routes:
        keep = set(rng.choice(len(samples), size=args.max_routes, replace=False).tolist())
        keep.update(named_idx)  # force-include explicitly requested queries
        keep = sorted(keep)
        old_to_new = {old: new for new, old in enumerate(keep)}
        named_idx = [old_to_new[i] for i in named_idx]
        samples = [samples[i] for i in keep]
        latent_matrix = latent_matrix[keep]

    n = len(samples)
    print(f"  {n} routes, {latent_matrix.shape[1]}D latent space")

    print("Extracting interpretable style features...")
    features = extract_features(samples)
    feat_labels = [FEATURE_LABELS[i] for i in _STYLE_FEAT_IDX]

    print("Computing z-space pairwise distances...")
    Zd = cdist(latent_matrix, latent_matrix)

    if named_idx:
        query_idx = named_idx
    else:
        query_idx = rng.choice(n, size=min(args.num_random_queries, n), replace=False).tolist()
    if not query_idx:
        raise SystemExit("No valid query routes resolved.")

    k = args.k
    print()
    print(f"=== Per-query: z-space neighbors vs. EMD ground truth (k={k}) ===")
    for qi in query_idx:
        order = np.argsort(Zd[qi])
        nearest = order[1:k + 1]
        farthest = order[-k:]
        random_other = rng.choice([i for i in range(n) if i != qi], size=k, replace=False)

        def _summarize(group, label):
            emds = np.array([route_emd(samples[qi], samples[j]) for j in group])
            fdiffs = np.array([_feat_diff(features, qi, j) for j in group])
            print(f"  {label:<8s}  z-dist={Zd[qi, group].round(3)}  "
                  f"EMD={emds.round(4)} (mean {emds.mean():.4f})  "
                  f"mean |Δfeature|={fdiffs.mean():.3f}")
            return emds

        print(f"--- Query: \"{samples[qi].name}\" ({samples[qi].num_holds} holds) ---")
        emd_near = _summarize(nearest, "nearest")
        emd_rand = _summarize(random_other, "random")
        emd_far = _summarize(farthest, "farthest")
        verdict = "✓ z-neighbors are geometrically closer" if emd_near.mean() < emd_rand.mean() < emd_far.mean() \
            else "✗ ordering not monotonic — inspect manually"
        print(f"  {verdict}")
        print()

    print(f"=== Aggregate: z-distance vs. EMD correlation over {args.n_pairs} random pairs ===")
    pairs = rng.choice(n, size=(args.n_pairs, 2), replace=True)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    zd = Zd[pairs[:, 0], pairs[:, 1]]
    emds = np.array([route_emd(samples[i], samples[j]) for i, j in pairs])
    rho, p = spearmanr(zd, emds)
    print(f"  Spearman(z-distance, EMD): rho={rho:.3f}, p={p:.2e}  (n={len(pairs)} pairs)")

    print()
    print(f"=== k-NN agreement: do z-space neighbors match EMD-space neighbors? ===")
    print(f"  (fraction of each query's top-{k} z-neighbors that also rank in its top-{k} EMD-neighbors)")
    overlaps = []
    for qi in query_idx:
        z_top = set(np.argsort(Zd[qi])[1:k + 1].tolist())
        emds_all = np.array([route_emd(samples[qi], samples[j]) if j != qi else np.inf for j in range(n)])
        emd_top = set(np.argsort(emds_all)[:k].tolist())
        overlap = len(z_top & emd_top) / k
        overlaps.append(overlap)
        print(f"  \"{samples[qi].name}\": {len(z_top & emd_top)}/{k} overlap")
    print(f"  mean overlap: {np.mean(overlaps):.2f}")
    print()
    print(f"Style features compared: {feat_labels}")


if __name__ == "__main__":
    main()
