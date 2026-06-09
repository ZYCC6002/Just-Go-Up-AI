"""Render a route board image to PNG bytes using PIL (full native resolution)."""
from __future__ import annotations

import io
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from database_interfaces.board_lib_interface import BoardLibInterface

_DB_PATH = _PROJECT_ROOT / "data" / "raw" / "kilter_database.sqlite"
_IMAGES_ROOT = _PROJECT_ROOT / "data" / "raw" / "kilter_images"

ROLE_COLORS = {
    "start": "#00FF00",
    "middle": "#00FFFF",
    "finish": "#FF00FF",
    "foot": "#FFA500",
}

_KILTER_GRADE_LABELS: dict[int, str] = {
    10: "V0", 11: "V1", 12: "V1", 13: "V2", 14: "V2",
    15: "V3", 16: "V3", 17: "V4", 18: "V4", 19: "V5",
    20: "V5", 21: "V6", 22: "V6", 23: "V7", 24: "V8",
    25: "V9", 26: "V10", 27: "V11", 28: "V12", 29: "V13",
    30: "V14", 31: "V15", 32: "V16", 33: "V17",
}


def _grade_label(grade: float | None) -> str:
    if grade is None:
        return "ungraded"
    v = _KILTER_GRADE_LABELS.get(int(grade), "")
    return f"{grade:.1f}  ({v})" if v else f"{grade:.1f}"


def _composite_pil(image_paths: list[Path]) -> Image.Image:
    base = Image.open(image_paths[0]).convert("RGBA")
    for path in image_paths[1:]:
        overlay = Image.open(path).convert("RGBA")
        if overlay.size != base.size:
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        base = Image.alpha_composite(base, overlay)
    return base


def _flatten_white(image: Image.Image) -> Image.Image:
    """Composite RGBA content onto a white background before saving/displaying."""
    if image.mode != "RGBA":
        return image.convert("RGB")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def render_route_image(climb_name: str) -> bytes:
    """Render the board image at native resolution (no text sidebar)."""
    with BoardLibInterface(_DB_PATH) as db:
        climb = db.get_climb_by_name(climb_name)
        if climb is None:
            raise ValueError(f"No climb found: {climb_name!r}")
        holds = db.get_hold_positions_for_climb(climb)
        if not holds:
            raise ValueError(f"No holds found for: {climb_name!r}")
        image_paths, board_edges = db.resolve_image_paths_for_climb(climb, _IMAGES_ROOT)
        board_left, board_right, board_bottom, board_top = board_edges

    base = _composite_pil(image_paths)
    img_w, img_h = base.size

    # 2× supersample for smooth anti-aliased circle edges
    SCALE = 2
    large = base.resize((img_w * SCALE, img_h * SCALE), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(large)

    def to_px(x: int, y: int) -> tuple[float, float]:
        x_norm = (x - board_left) / max(board_right - board_left, 1)
        y_norm = (y - board_bottom) / max(board_top - board_bottom, 1)
        return x_norm * (img_w * SCALE - 1), (1.0 - y_norm) * (img_h * SCALE - 1)

    r = 25 * SCALE
    lw = round(2.5 * SCALE)
    for hold in holds:
        if hold.x is None or hold.y is None:
            continue
        x_px, y_px = to_px(int(hold.x), int(hold.y))
        color = ROLE_COLORS.get((hold.role_name or "").lower(), "#FFFF00")
        draw.ellipse(
            [(x_px - r, y_px - r), (x_px + r, y_px + r)],
            outline=color,
            width=lw,
        )

    out = _flatten_white(large.resize((img_w, img_h), Image.Resampling.LANCZOS))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def get_route_info(climb_name: str) -> dict:
    """Return route metadata as a dict for the frontend info panel."""
    with BoardLibInterface(_DB_PATH) as db:
        climb = db.get_climb_by_name(climb_name)
        if climb is None:
            raise ValueError(f"No climb found: {climb_name!r}")
        holds = db.get_hold_positions_for_climb(climb)
        stats = db.get_climb_stats(climb.uuid)

    grade = stats.get("difficulty_average")
    angle = getattr(climb, "angle", None)
    role_counts = Counter((h.role_name or "unknown").lower() for h in holds)
    type_counts = Counter(
        h.metadata.type for h in holds if h.metadata is not None and h.metadata.type
    )
    return {
        "name": climb.name,
        "grade": grade,
        "grade_label": _grade_label(grade),
        "angle": float(angle) if angle is not None else None,
        "quality": stats.get("quality_average"),
        "ascents": stats.get("ascensionist_count"),
        "n_holds": len(holds),
        "role_counts": {r: role_counts.get(r, 0) for r in ("start", "middle", "finish", "foot")},
        "type_counts": dict(type_counts.most_common()),
    }
