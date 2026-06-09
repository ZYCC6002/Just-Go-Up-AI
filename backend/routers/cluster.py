"""Cluster job endpoints: POST /api/cluster, GET /api/cluster/{job_id}."""
from __future__ import annotations

import threading
from typing import Any

import torch
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from backend.lib.cluster import DEFAULT_CACHE_KEY, get_cache_key, get_cache_path, run_cluster

router = APIRouter(prefix="/api/cluster", tags=["cluster"])

# In-memory job store (single-process; fine for HF Spaces single-instance deployment)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


class ClusterParams(BaseModel):
    method: str = "kmeans"
    n_clusters: int = 6
    max_routes: int = 2000
    pre_reduce: bool = True
    pre_reduce_method: str = "umap"
    pre_reduce_dims: int = 5
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 50
    hdbscan_min_samples: int | None = None
    min_grade: float | None = None
    max_grade: float | None = None
    min_angle: float | None = None
    max_angle: float | None = None
    seed: int = 42


def _run_cluster_job(job_id: str, params: dict) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        run_cluster(params)
        cache_path = get_cache_path(job_id)
        payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        n_routes = len(payload.get("samples", []))
        n_clusters = int(payload.get("n_clusters", 0))
        with _jobs_lock:
            _jobs[job_id].update(
                {
                    "status": "done",
                    "cache_key": job_id,
                    "n_routes": n_routes,
                    "n_clusters": n_clusters,
                }
            )
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update({"status": "error", "error": str(exc)})
        raise


@router.get("/default")
def get_default_cluster():
    """Return the pre-built default cluster cache if it exists."""
    cache_path = get_cache_path(DEFAULT_CACHE_KEY)
    if not cache_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Default cache not found")
    try:
        payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        return {
            "job_id": DEFAULT_CACHE_KEY,
            "status": "done",
            "cache_key": DEFAULT_CACHE_KEY,
            "n_routes": len(payload.get("samples", [])),
            "n_clusters": int(payload.get("n_clusters", 0)),
        }
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("")
def start_cluster(params: ClusterParams, background_tasks: BackgroundTasks):
    params_dict = params.model_dump()
    cache_key = get_cache_key(params_dict)
    cache_path = get_cache_path(cache_key)

    if cache_path.exists():
        try:
            payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
            return {
                "job_id": cache_key,
                "status": "done",
                "cache_key": cache_key,
                "n_routes": len(payload.get("samples", [])),
                "n_clusters": int(payload.get("n_clusters", 0)),
            }
        except Exception:
            pass  # corrupt cache — re-run

    with _jobs_lock:
        existing = _jobs.get(cache_key)
        if existing and existing["status"] in ("pending", "running"):
            return {"job_id": cache_key, **existing}
        _jobs[cache_key] = {"status": "pending", "cache_key": None, "error": None}

    background_tasks.add_task(_run_cluster_job, cache_key, params_dict)
    return {"job_id": cache_key, "status": "pending"}


@router.get("/{job_id}")
def get_cluster_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        cache_path = get_cache_path(job_id)
        if cache_path.exists():
            try:
                payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
                return {
                    "status": "done",
                    "cache_key": job_id,
                    "n_routes": len(payload.get("samples", [])),
                    "n_clusters": int(payload.get("n_clusters", 0)),
                }
            except Exception:
                pass
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    return {"job_id": job_id, **job}
