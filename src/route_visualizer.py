from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from database_interfaces.board_lib_interface import BoardLibInterface

# Lazy import so this module can be imported without the full training stack.
try:
    from data_preprocessing.route_preprocessing import RouteSample
except ImportError:
    RouteSample = None  # type: ignore[assignment,misc]


ROLE_COLORS = {
    "start": "#00FF00",
    "middle": "#00FFFF",
    "finish": "#FF00FF",
    "foot": "#FFA500",
}

# Kilter difficulty_grades integer → V-grade label (grades 10–33 cover V0–V16).
_KILTER_GRADE_LABELS: dict[int, str] = {
    10: "V0",
    11: "V1",
    12: "V1",
    13: "V2",
    14: "V2",
    15: "V3",
    16: "V3",
    17: "V4",
    18: "V4",
    19: "V5",
    20: "V5",
    21: "V6",
    22: "V6",
    23: "V7",
    24: "V8",
    25: "V9",
    26: "V10",
    27: "V11",
    28: "V12",
    29: "V13",
    30: "V14",
    31: "V15",
    32: "V16",
    33: "V17",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_db_path() -> Path:
    return _project_root() / "data" / "raw" / "kilter_database.sqlite"


def _default_images_root() -> Path:
    return _project_root() / "data" / "raw" / "kilter_images"


def _composite_images(image_paths: list[Path]) -> np.ndarray:
    """Alpha-composite all PNG layers into a single RGBA array."""
    base = Image.open(image_paths[0]).convert("RGBA")
    for path in image_paths[1:]:
        overlay = Image.open(path).convert("RGBA")
        if overlay.size != base.size:
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        base = Image.alpha_composite(base, overlay)
    return np.array(base)


def _grade_label(grade: float | None) -> str:
    """Return a human-readable grade string, e.g. '22.0 (V6)'."""
    if grade is None:
        return "ungraded"
    v = _KILTER_GRADE_LABELS.get(int(grade), "")
    suffix = f"  ({v})" if v else ""
    return f"{grade:.1f}{suffix}"


def _build_metrics_text(
    *,
    climb,
    holds,
    stats: dict,
    sample,  # RouteSample | None
) -> str:
    """Compose the metrics string for the info panel."""
    lines: list[str] = []
    sep = "─" * 22

    # --- Header ---
    lines += ["Route Metrics", sep]

    # Name (wrap at ~24 chars)
    name = climb.name or "<unnamed>"
    if len(name) > 24:
        name = name[:21] + "..."
    lines.append(f"Name:  {name}")
    lines.append(f"UUID:  {climb.uuid[:8]}...")
    lines.append("")

    # --- Difficulty ---
    lines.append("Difficulty")
    # Prefer values from RouteSample when available — it uses climb_stats.angle
    # and climb_stats.difficulty_average, which may differ from the climbs table.
    # (climb_stats rows are per-angle; climbs.angle is the route's default angle.)
    grade = sample.grade if sample is not None else stats.get("difficulty_average")
    angle = sample.angle if sample is not None else climb.angle
    lines.append(f"  Grade:  {_grade_label(grade)}")
    lines.append(f"  Angle:  {angle:.0f}°")
    lines.append("")

    # --- Community ---
    lines.append("Community")
    quality = stats.get("quality_average")
    ascents = stats.get("ascensionist_count")
    fa = stats.get("fa_username")
    lines.append(f"  Quality: {f'{quality:.2f} / 4.0' if quality is not None else 'n/a'}")
    lines.append(f"  Ascents: {f'{ascents:,}' if ascents is not None else 'n/a'}")
    if fa:
        fa_display = fa[:18] + "..." if len(fa) > 18 else fa
        lines.append(f"  Setter:  {fa_display}")
    lines.append("")

    # --- Holds ---
    lines.append("Holds")
    total_holds = len(holds)
    lines.append(f"  Total:    {total_holds}")
    if sample is not None:
        lines.append(f"  Coverage: {sample.metadata_coverage * 100:.1f}%")
    role_counts = Counter(
        (h.role_name or "unknown").lower() for h in holds
    )
    for role in ("start", "middle", "finish", "foot"):
        cnt = role_counts.get(role, 0)
        if cnt:
            lines.append(f"  {role.capitalize():<8}{cnt}")
    lines.append("")

    # --- Hold types ---
    type_counts = Counter(
        h.metadata.type
        for h in holds
        if h.metadata is not None and h.metadata.type
    )
    if type_counts:
        lines.append("Hold Types")
        for hold_type, cnt in type_counts.most_common():
            label = hold_type[:14] if hold_type else "unknown"
            lines.append(f"  {label:<14} {cnt}")
        lines.append("")

    # --- Geometry (hand holds only — excludes foot holds) ---
    _HAND_ROLES = {"start", "middle", "finish"}
    coords = np.array(
        [
            (h.x, h.y)
            for h in holds
            if h.x is not None
            and h.y is not None
            and (h.role_name or "").lower() in _HAND_ROLES
        ],
        dtype=np.float32,
    )
    if len(coords) >= 2:
        diffs = coords[:, None, :] - coords[None, :, :]   # [N, N, 2]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))          # [N, N]
        pairwise = dists[np.triu_indices(len(coords), k=1)]
        lines.append("Geometry (hands only)")
        lines.append(f"  Mean dist:  {pairwise.mean():.1f}")
        lines.append(f"  Std dist:   {pairwise.std():.1f}")

    return "\n".join(lines)


def visualize_route(climb_name: str, *, sample=None) -> None:
    """Display a climb on the correct board image with holds circled by role color.

    Args:
        climb_name: Exact climb name to look up in the database.
        sample: Optional RouteSample from the cluster cache. When provided,
                ``num_holds``, ``metadata_coverage``, and ``grade`` from the
                cached sample are included in the metrics panel.
    """
    db_path = _default_db_path()
    images_root = _default_images_root()

    with BoardLibInterface(db_path) as db:
        climb = db.get_climb_by_name(climb_name)
        if climb is None:
            raise ValueError(f"No climb found with name: {climb_name}")
        holds = db.get_hold_positions_for_climb(climb)
        if not holds:
            raise ValueError(f"No hold data found for climb: {climb.name}")

        image_paths, board_edges = db.resolve_image_paths_for_climb(
            climb,
            images_root,
        )
        board_left, board_right, board_bottom, board_top = board_edges

        stats = db.get_climb_stats(climb.uuid)

    img = _composite_images(image_paths)
    img_h, img_w = img.shape[0], img.shape[1]

    def to_px(x: int, y: int) -> tuple[float, float]:
        x_norm = (x - board_left) / max(board_right - board_left, 1)
        y_norm = (y - board_bottom) / max(board_top - board_bottom, 1)
        x_px = x_norm * (img_w - 1)
        # Invert y to match image coords (top-left origin).
        y_px = (1.0 - y_norm) * (img_h - 1)
        return x_px, y_px

    fig, (ax_img, ax_info) = plt.subplots(
        1, 2,
        figsize=(14, 8),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    ax_img.imshow(img)
    ax_img.set_title(climb.name)
    ax_img.axis("off")

    for hold in holds:
        if hold.x is None or hold.y is None:
            continue
        x_px, y_px = to_px(int(hold.x), int(hold.y))
        color = ROLE_COLORS.get((hold.role_name or "").lower(), "#FFFF00")
        circle = Circle((x_px, y_px), radius=25, fill=False, linewidth=2.2, edgecolor=color)
        ax_img.add_patch(circle)

    metrics_text = _build_metrics_text(
        climb=climb,
        holds=holds,
        stats=stats,
        sample=sample,
    )
    ax_info.text(
        0.05, 0.97,
        metrics_text,
        transform=ax_info.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=8.5,
        linespacing=1.55,
    )
    ax_info.axis("off")

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a Kilter climb route on board image layers.")
    parser.add_argument("--climb-name", type=str, default="Alberts dream", help="Exact climb name to visualize.")
    args = parser.parse_args()
    visualize_route(args.climb_name)


if __name__ == "__main__":
    main()
