"""Cluster cache management: cache-key generation and run_analysis wrapper."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from backend.lib.startup import project_root

_PROJECT_ROOT = project_root()
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

_MODELS_DIR = _PROJECT_ROOT / "data" / "models"
_CHECKPOINT_PATH = str(_MODELS_DIR / "analysis_model.pt")
_DB_PATH = str(_PROJECT_ROOT / "data" / "raw" / "kilter_database.sqlite")
_CACHE_PATH = str(_PROJECT_ROOT / "data" / "preprocessed_routes_cache.pt")

_MAX_CACHES = 10

# Pre-built cache: kmeans k=6, max_routes=5000, umap pre-reduce (dims=5, n_neighbors=15, min_dist=0.0), seed=42
DEFAULT_CACHE_KEY = "19e1e1c68143"


def _evict_oldest_caches() -> None:
    """Delete the oldest cluster_*.pt files if more than _MAX_CACHES exist."""
    caches = sorted(
        _MODELS_DIR.glob("cluster_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    for old in caches[:-_MAX_CACHES]:
        old.unlink(missing_ok=True)


def get_cache_key(params: dict) -> str:
    serialized = json.dumps(params, sort_keys=True)
    return hashlib.sha1(serialized.encode()).hexdigest()[:12]


def get_cache_path(cache_key: str) -> Path:
    return _MODELS_DIR / f"cluster_{cache_key}.pt"


def run_cluster(params: dict) -> None:
    """Call routes_cluster.run_analysis with the given params dict."""
    from data_analysis.routes_cluster import run_analysis

    cache_key = get_cache_key(params)
    cache_path = get_cache_path(cache_key)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    run_analysis(
        db_path=_DB_PATH,
        cache_path=_CACHE_PATH,
        checkpoint_path=_CHECKPOINT_PATH,
        cluster_cache_path=str(cache_path),
        metadata_source="kilter",
        metadata_product_id=1,
        max_routes=params.get("max_routes", 2000),
        method=params.get("method", "kmeans"),
        n_clusters=params.get("n_clusters", 6),
        hdbscan_min_cluster_size=params.get("hdbscan_min_cluster_size", 50),
        hdbscan_min_samples=params.get("hdbscan_min_samples"),
        pre_reduce=params.get("pre_reduce", True),
        pre_reduce_method=params.get("pre_reduce_method", "umap"),
        pre_reduce_dims=params.get("pre_reduce_dims", 5),
        umap_n_neighbors=params.get("umap_n_neighbors", 15),
        umap_min_dist=params.get("umap_min_dist", 0.0),
        batch_size=32,
        latent_dim_override=None,
        min_grade=params.get("min_grade"),
        max_grade=params.get("max_grade"),
        min_angle=params.get("min_angle"),
        max_angle=params.get("max_angle"),
        include_ungraded=False,
        seed=params.get("seed", 42),
        no_noise=True,
    )
    _evict_oldest_caches()
