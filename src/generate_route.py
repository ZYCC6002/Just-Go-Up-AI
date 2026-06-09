"""Sample a climbing route from the VAE latent space and visualize it.

Usage:
    uv run src/generate_route.py --checkpoint-path data/models/gen_model.pt
    uv run src/generate_route.py --checkpoint-path data/models/gen_model.pt --n-holds 18 --seed 42
    uv run src/generate_route.py --checkpoint-path data/models/gen_model.pt --output-path data/generated_route.png
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from database_interfaces.board_lib_interface import BoardLibInterface
from model_training.model_utils import build_model_from_checkpoint
from route_visualizer import ROLE_COLORS, _composite_images

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_DB   = PROJECT_ROOT / "data" / "raw" / "kilter_database.sqlite"
_DEFAULT_CKPT = PROJECT_ROOT / "data" / "models" / "gen_model.pt"
_IMAGES_ROOT  = PROJECT_ROOT / "data" / "raw" / "kilter_images"

_KILTER_LAYOUT_ID  = 1
_KILTER_PRODUCT_ID = 1


# ── Generation ────────────────────────────────────────────────────────────────

def decode_z_to_holds(
    model,
    vocabs,
    z: torch.Tensor,
    n_holds: int,
    device: torch.device,
) -> list[dict]:
    """Run the decoder on z [1, latent_dim] and return hold dicts (EOS filtered).

    Each hold dict has keys: hole_id, role_id, type_name, x_pred, y_pred.
    """
    # Dummy batch — all token features are irrelevant because the masked decoder
    # replaces every non-padded position with mask_token + pos_embedding.
    dummy = {
        "type_encoded_id": torch.zeros(1, n_holds, dtype=torch.long, device=device),
        "role_encoded_id": torch.zeros(1, n_holds, dtype=torch.long, device=device),
        "hole_encoded_id": torch.zeros(1, n_holds, dtype=torch.long, device=device),
        "x":               torch.zeros(1, n_holds, device=device),
        "y":               torch.zeros(1, n_holds, device=device),
        "orientation_sin": torch.zeros(1, n_holds, device=device),
        "orientation_cos": torch.zeros(1, n_holds, device=device),
        "size":            torch.zeros(1, n_holds, device=device),
        "padding_mask":    torch.zeros(1, n_holds, dtype=torch.bool, device=device),
    }

    with torch.no_grad():
        out = model.decoder(dummy, z=z)

    type_logits = out["type_logits"][0]  # [n_holds, type_vocab+1]
    role_logits = out["role_logits"][0]
    hole_logits = out["hole_logits"][0]
    x_pred      = out["x_pred"][0]       # [n_holds], normalized [0,1]
    y_pred      = out["y_pred"][0]

    type_eos = model.decoder.type_eos_id
    hole_eos = model.decoder.hole_eos_id

    holds = []
    for i in range(n_holds):
        type_idx = int(type_logits[i].argmax())
        if type_idx == type_eos:
            continue

        hole_idx = int(hole_logits[i].argmax())
        if hole_idx >= hole_eos:
            continue

        role_idx = int(role_logits[i].argmax())

        type_name = (
            vocabs.type_vocab.idx_to_item[type_idx]
            if type_idx < len(vocabs.type_vocab.idx_to_item)
            else "unknown"
        )
        role_id = int(
            vocabs.role_vocab.idx_to_item[role_idx]
            if role_idx < len(vocabs.role_vocab.idx_to_item)
            else 13  # default to middle
        )
        hole_id = vocabs.hole_vocab.idx_to_item[hole_idx]

        holds.append({
            "hole_id":   hole_id,
            "role_id":   role_id,   # raw integer; resolved to name via db.role_map()
            "type_name": type_name,
            "x_pred":    float(x_pred[i]),
            "y_pred":    float(y_pred[i]),
        })

    return holds


def sample_route(
    model,
    vocabs,
    *,
    n_holds: int,
    device: torch.device,
    seed: int | None = None,
) -> list[dict]:
    """Sample z ~ N(0,I), decode, return list of hold dicts."""
    if seed is not None:
        torch.manual_seed(seed)
    latent_dim = model.bottleneck.to_mu.out_features
    z = torch.randn(1, latent_dim, device=device)
    return decode_z_to_holds(model, vocabs, z, n_holds, device)


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_generated_route(
    holds: list[dict],
    *,
    db_path: Path,
    images_root: Path,
    title: str = "Generated Route",
    output_path: Path | None = None,
) -> None:
    """Draw the generated holds on the Kilter board image.

    Board size is selected based on the bounding box of the predicted holds,
    so holds are never clipped regardless of where they fall on the board.
    """
    with BoardLibInterface(db_path) as db:
        positions = db.get_hole_positions([h["hole_id"] for h in holds])
        roles     = db.role_map()

        # Compute hold bounds and select the smallest board that contains them.
        xs = [xy[0] for xy in positions.values()]
        ys = [xy[1] for xy in positions.values()]
        if xs:
            climb_edges = (min(xs), max(xs), min(ys), max(ys))
        else:
            climb_edges = (0, 144, 0, 156)  # full board fallback

        set_ids = db.fetchall(
            "SELECT DISTINCT set_id FROM product_sizes_layouts_sets WHERE layout_id=? ORDER BY set_id",
            [_KILTER_LAYOUT_ID],
        )
        set_ids = [int(r[0]) for r in set_ids]

        image_paths, board_edges = db.resolve_image_paths(
            _KILTER_LAYOUT_ID, set_ids, climb_edges, _KILTER_PRODUCT_ID, images_root,
        )
        board_left, board_right, board_bottom, board_top = board_edges

    img = _composite_images(image_paths)
    img_h, img_w = img.shape[:2]

    def to_px(x: int, y: int) -> tuple[float, float]:
        x_norm = (x - board_left) / max(board_right - board_left, 1)
        y_norm = (y - board_bottom) / max(board_top - board_bottom, 1)
        return x_norm * (img_w - 1), (1.0 - y_norm) * (img_h - 1)

    fig, (ax_img, ax_info) = plt.subplots(
        1, 2, figsize=(14, 8),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    ax_img.imshow(img, interpolation="nearest")
    ax_img.set_title(title)
    ax_img.set_facecolor("white")
    ax_img.axis("off")

    placed: list[dict] = []
    for hold in holds:
        xy = positions.get(hold["hole_id"])
        if xy is None:
            continue
        role_name = roles.get(hold["role_id"], "middle")
        color = ROLE_COLORS.get(role_name.lower(), "#FFFF00")
        ax_img.add_patch(Circle(to_px(*xy), radius=25, fill=False, linewidth=2.2, edgecolor=color))
        placed.append({**hold, "role_name": role_name})

    lines = [
        "Generated Route",
        "─" * 22,
        f"Total holds:  {len(placed)}",
        f"(EOS/missing: {len(holds) - len(placed)})",
        "",
        "Role counts",
    ]
    for role, cnt in Counter(h["role_name"] for h in placed).most_common():
        lines.append(f"  {role:<10} {cnt}")
    lines += ["", "Hold types"]
    for htype, cnt in Counter(h["type_name"] for h in placed).most_common():
        lines.append(f"  {htype:<12} {cnt}")
    lines += [
        "",
        "Legend",
        "  ● green   = start",
        "  ● cyan    = middle",
        "  ● magenta = finish",
        "  ● orange  = foot",
    ]

    ax_info.text(
        0.05, 0.97, "\n".join(lines),
        transform=ax_info.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=8.5,
        linespacing=1.55,
    )
    ax_info.axis("off")
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a route by sampling from the VAE latent space.")
    parser.add_argument("--checkpoint-path", default=str(_DEFAULT_CKPT))
    parser.add_argument("--db-path",         default=str(_DEFAULT_DB))
    parser.add_argument("--n-holds",  type=int, default=15,
                        help="Number of holds to generate (default 15).")
    parser.add_argument("--seed",     type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--output-path", default=None,
                        help="Save figure to this path instead of showing it.")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Loading checkpoint: {args.checkpoint_path}")
    model, _ = build_model_from_checkpoint(args.checkpoint_path, vocabs=None, device=device)
    model.eval()

    ckpt   = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    vocabs = ckpt["vocabs"]

    print(f"Sampling route with n_holds={args.n_holds}, seed={args.seed}")
    holds = sample_route(model, vocabs, n_holds=args.n_holds, device=device, seed=args.seed)
    print(f"Valid holds after EOS filtering: {len(holds)}")
    for h in holds:
        print(f"  hole_id={h['hole_id']:5d}  role_id={h['role_id']:<4}  type={h['type_name']}")

    visualize_generated_route(
        holds,
        db_path=Path(args.db_path),
        images_root=_IMAGES_ROOT,
        title=f"Generated Route  n_holds={args.n_holds}  seed={args.seed}",
        output_path=Path(args.output_path) if args.output_path else None,
    )


if __name__ == "__main__":
    main()
