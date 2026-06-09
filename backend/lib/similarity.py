"""Latent-space similarity search."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from data_analysis.clustered_analysis_utils import load_cluster_cache


def find_similar(cache_path: str, climb_name: str, k: int = 20) -> list[dict]:
    """Return the k most similar routes to `climb_name` ranked by z-distance.

    Returns a list of dicts with: rank, name, grade, angle, cluster, z_distance.
    Raises ValueError if the climb name has no match in the cache.
    """
    from scipy.spatial.distance import cdist

    samples, latent_matrix, cluster_ids, _ = load_cluster_cache(cache_path)

    # Substring match (case-insensitive)
    matches = [i for i, s in enumerate(samples) if climb_name.lower() in s.name.lower()]
    if not matches:
        raise ValueError(f"No route matching {climb_name!r} found in cache.")
    qi = matches[0]

    dists = cdist(latent_matrix[qi : qi + 1], latent_matrix)[0]  # [N]
    order = np.argsort(dists)

    results = []
    rank = 0
    for idx in order:
        if int(idx) == qi:
            continue
        rank += 1
        s = samples[idx]
        results.append(
            {
                "rank": rank,
                "name": s.name,
                "grade": float(s.grade) if s.grade is not None else None,
                "angle": float(s.angle) if s.angle is not None else None,
                "cluster": int(cluster_ids[idx]),
                "z_distance": float(dists[idx]),
            }
        )
        if rank >= k:
            break

    return results
