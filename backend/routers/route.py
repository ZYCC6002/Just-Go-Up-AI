"""Route board image endpoint: GET /api/route/image."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.lib.route_image import get_route_info, render_route_image

router = APIRouter(prefix="/api/route", tags=["route"])


@router.get("/image")
def get_route_image(name: str):
    try:
        png_bytes = render_route_image(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return Response(content=png_bytes, media_type="image/png")


@router.get("/info")
def get_route_info_endpoint(name: str):
    try:
        return get_route_info(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
