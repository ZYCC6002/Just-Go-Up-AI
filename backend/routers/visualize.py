"""Projection endpoint: GET /api/visualize."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from backend.lib.cluster import get_cache_path

router = APIRouter(prefix="/api/visualize", tags=["visualize"])


@router.get("")
def get_visualization(
    cache_key: str,
    method: str = Query("umap", pattern="^(pca|umap|tsne)$"),
    n_dims: int = Query(2, ge=2, le=3),
    max_routes: int = Query(2000, ge=1, le=10000),
    seed: int = 42,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
):
    from data_analysis.clustered_analysis_utils import (
        load_cluster_cache,
        reduce_dimensions,
        subset_cluster_cache,
    )

    cache_path = get_cache_path(cache_key)
    if not cache_path.exists():
        raise HTTPException(status_code=404, detail=f"Cluster cache {cache_key!r} not found")

    samples, latent_matrix, cluster_ids, payload = load_cluster_cache(str(cache_path))
    samples, latent_matrix, cluster_ids = subset_cluster_cache(
        samples, latent_matrix, cluster_ids, max_routes=max_routes
    )

    if method == "pca":
        from sklearn.decomposition import PCA
        projected = PCA(n_components=n_dims, random_state=seed).fit_transform(latent_matrix)
        axis_labels = [f"PC{i + 1}" for i in range(n_dims)]

    elif method == "umap":
        # Always project from the original latent space for display — running UMAP
        # on top of the clustering pre-reduced matrix (UMAP-on-UMAP) creates filaments.
        projected = reduce_dimensions(
            latent_matrix,
            method="umap",
            n_components=n_dims,
            seed=seed,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
        )
        axis_labels = [f"UMAP-{i + 1}" for i in range(n_dims)]

    elif method == "tsne":
        from sklearn.manifold import TSNE
        if tsne_perplexity >= len(samples):
            raise HTTPException(
                status_code=400,
                detail=f"tsne_perplexity ({tsne_perplexity}) must be < n_routes ({len(samples)})",
            )
        projected = TSNE(
            n_components=n_dims,
            perplexity=tsne_perplexity,
            learning_rate="auto",
            init="pca",
            random_state=seed,
        ).fit_transform(latent_matrix)
        axis_labels = [f"t-SNE-{i + 1}" for i in range(n_dims)]

    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method!r}")

    _KILTER_GRADE_LABELS: dict[int, str] = {
        10: "V0", 11: "V1", 12: "V1", 13: "V2", 14: "V2",
        15: "V3", 16: "V3", 17: "V4", 18: "V4", 19: "V5",
        20: "V5", 21: "V6", 22: "V6", 23: "V7", 24: "V8",
        25: "V9", 26: "V10", 27: "V11", 28: "V12", 29: "V13",
        30: "V14", 31: "V15", 32: "V16", 33: "V17",
    }

    def grade_label(grade):
        if grade is None:
            return None
        v = _KILTER_GRADE_LABELS.get(int(grade), "")
        return f"{grade:.1f} ({v})" if v else f"{grade:.1f}"

    points = []
    for i, s in enumerate(samples):
        pt: dict = {
            "x": float(projected[i, 0]),
            "y": float(projected[i, 1]),
            "name": s.name,
            "grade": float(s.grade) if s.grade is not None else None,
            "grade_label": grade_label(s.grade),
            "angle": float(s.angle) if s.angle is not None else None,
            "cluster": int(cluster_ids[i]),
            "uuid": s.uuid,
        }
        if n_dims == 3:
            pt["z"] = float(projected[i, 2])
        points.append(pt)

    return {
        "points": points,
        "axis_labels": axis_labels,
        "n_clusters": int(payload.get("n_clusters", len(set(cluster_ids.tolist())))),
        "method": method,
        "n_dims": n_dims,
    }
