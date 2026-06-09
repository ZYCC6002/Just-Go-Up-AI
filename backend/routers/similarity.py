"""Similarity search endpoint: GET /api/similarity."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.lib.cluster import get_cache_path
from backend.lib.similarity import find_similar

router = APIRouter(prefix="/api/similarity", tags=["similarity"])


@router.get("")
def get_similarity(
    cache_key: str,
    name: str,
    k: int = Query(20, ge=1, le=100),
):
    cache_path = get_cache_path(cache_key)
    if not cache_path.exists():
        raise HTTPException(status_code=404, detail=f"Cluster cache {cache_key!r} not found")
    try:
        results = find_similar(str(cache_path), name, k=k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"query": name, "results": results}
