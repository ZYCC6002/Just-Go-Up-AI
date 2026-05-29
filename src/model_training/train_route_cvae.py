from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_preprocessing.route_preprocessing import build_training_samples_from_db, split_route_samples
from model_training.model_utils import (
    filter_samples_by_decoder_max_len,
    iter_minibatches,
    select_device,
)
from model_training.route_vae_bottleneck import (
    GradeAngleAdversaryHead,
    RouteConditionalVAE,
    RouteVAEBottleneck,
    RouteVAEBottleneckConfig,
    kl_divergence_loss,
)
from model_training.route_vae_decoder import RouteTransformerDecoder, RouteVAEDecoderConfig
from model_training.route_vae_encoder import RouteTransformerEncoder
from model_training.training_utils import DecoderEOSIds, masked_mse, prepare_cvae_training_batch


@dataclass
class EpochMetrics:
    total_loss: float
    categorical_loss: float
    numeric_loss: float
    kl_raw: float          # unweighted KL — for kl=X(+Y) diagnostics, not the loss contribution
    adversary_loss: float
    num_batches: int


def _compute_kl_beta(*, epoch: int, target_kl_beta: float, kl_warmup_epochs: int) -> float:
    """Linear KL annealing from 0 -> target_kl_beta over warmup epochs."""
    if kl_warmup_epochs <= 0:
        return float(target_kl_beta)
    ratio = min(max(epoch, 0), kl_warmup_epochs) / float(kl_warmup_epochs)
    return float(target_kl_beta) * ratio


def _save_loss_curve(
    *,
    epochs: list[int],
    train_total: list[float],
    val_total: list[float],
    train_recon: list[float],
    val_recon: list[float],
    train_kl: list[float],
    val_kl: list[float],
    output_path: Path,
) -> None:
    if not epochs:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    for ax, train, val, title in [
        (ax1, train_total, val_total, "Total Loss"),
        (ax2, train_recon, val_recon, "Reconstruction Loss"),
        (ax3, train_kl, val_kl, "KL Divergence Loss"),
    ]:
        ax.plot(epochs, train, label="Train", linewidth=2)
        ax.plot(epochs, val, label="Val", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved loss curve: {output_path}")


def _build_model(
    vocabs: Any,
    device: torch.device,
    *,
    latent_dim: int,
    encoder_use_cond_adaln: bool,
    encoder_d_model: int,
    encoder_nhead: int,
    encoder_num_layers: int,
    encoder_dim_feedforward: int,
    decoder_use_cond_adaln: bool,
    decoder_d_model: int,
    decoder_num_layers: int,
    decoder_dim_feedforward: int,
    use_absolute_pos: bool = True,
    use_type_feature: bool = True,
    shape_desc_dim: int = 9,
    route_pool_mode: str = "cls",
) -> tuple[RouteConditionalVAE, DecoderEOSIds, dict[str, float]]:
    if encoder_d_model % encoder_nhead != 0:
        raise ValueError(
            f"--encoder-d-model ({encoder_d_model}) must be divisible by --encoder-nhead ({encoder_nhead})"
        )
    if decoder_d_model % 8 != 0:
        raise ValueError(f"--decoder-d-model ({decoder_d_model}) must be divisible by nhead=8")

    enc_cfg = vocabs.to_transformer_config(
        d_model=encoder_d_model,
        nhead=encoder_nhead,
        num_layers=encoder_num_layers,
        dim_feedforward=encoder_dim_feedforward,
    )
    enc_cfg.use_cond_adaln = encoder_use_cond_adaln
    enc_cfg.use_absolute_pos = use_absolute_pos
    enc_cfg.use_type_feature = use_type_feature
    enc_cfg.shape_desc_dim = shape_desc_dim
    enc_cfg.route_pool_mode = route_pool_mode

    dec_cfg = RouteVAEDecoderConfig(
        type_vocab_size=enc_cfg.type_vocab_size,
        role_vocab_size=enc_cfg.role_vocab_size,
        hole_id_vocab_size=enc_cfg.hole_id_vocab_size,
        latent_dim=latent_dim,
        use_cond_adaln=decoder_use_cond_adaln,
        d_model=decoder_d_model,
        num_layers=decoder_num_layers,
        dim_feedforward=decoder_dim_feedforward,
        # Sync delta embedding dim with encoder
        delta_embed_dim=enc_cfg.delta_embed_dim,
        use_knn_features=False,  # decoder never uses full-sequence knn features
        use_absolute_pos=use_absolute_pos,
        use_type_feature=use_type_feature,
        x_min=enc_cfg.x_min,
        x_max=enc_cfg.x_max,
        y_min=enc_cfg.y_min,
        y_max=enc_cfg.y_max,
        angle_min=enc_cfg.angle_min,
        angle_max=enc_cfg.angle_max,
        grade_min=enc_cfg.grade_min,
        grade_max=enc_cfg.grade_max,
    )
    bottleneck_cfg = RouteVAEBottleneckConfig(
        encoder_embedding_dim=enc_cfg.d_model,
        latent_dim=latent_dim,
    )

    model = RouteConditionalVAE(
        encoder=RouteTransformerEncoder(enc_cfg),
        bottleneck=RouteVAEBottleneck(bottleneck_cfg),
        decoder=RouteTransformerDecoder(dec_cfg),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Encoder: d_model={enc_cfg.d_model} nhead={enc_cfg.nhead} "
        f"layers={enc_cfg.num_layers} ffn={enc_cfg.dim_feedforward}"
    )
    print(
        f"Decoder: d_model={dec_cfg.d_model} nhead={dec_cfg.nhead} "
        f"layers={dec_cfg.num_layers} ffn={dec_cfg.dim_feedforward}"
    )
    print(f"Total parameters: {total_params:,}")

    eos_ids = DecoderEOSIds(
        type_eos_id=model.decoder.type_eos_id,
        role_eos_id=model.decoder.role_eos_id,
        hole_eos_id=model.decoder.hole_eos_id,
    )
    norm_ranges = {
        "angle_min": float(enc_cfg.angle_min),
        "angle_max": float(enc_cfg.angle_max),
        "grade_min": float(enc_cfg.grade_min),
        "grade_max": float(enc_cfg.grade_max),
    }
    return model, eos_ids, norm_ranges


def _load_or_build_samples_and_vocabs(args: argparse.Namespace) -> tuple[list[Any], Any]:
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_key = {
        "db_path": str(Path(args.db_path).resolve()),
        "metadata_source": args.metadata_source,
        "metadata_product_id": args.metadata_product_id,
        "required_product_id": 1,
        "require_full_metadata": args.require_full_metadata,
        "max_routes": args.max_routes,
        "min_holds": args.min_holds,
    }

    if cache_path.exists() and not args.rebuild_cache:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("cache_key") == cache_key:
            print(f"Loaded preprocessed dataset cache: {cache_path}")
            return payload["samples"], payload["vocabs"]

    print("Building dataset from DB (cache miss)...")
    samples, vocabs = build_training_samples_from_db(
        args.db_path,
        require_full_metadata=args.require_full_metadata,
        metadata_source=args.metadata_source,
        metadata_product_id=args.metadata_product_id,
        max_routes=args.max_routes,
        min_holds=args.min_holds,
    )
    torch.save({"cache_key": cache_key, "samples": samples, "vocabs": vocabs}, cache_path)
    print(f"Saved preprocessed dataset cache: {cache_path}")
    return samples, vocabs



def _compute_batch_losses(
    model: RouteConditionalVAE,
    batch_samples: list[Any],
    *,
    device: torch.device,
    eos_ids: DecoderEOSIds,
    kl_beta: float,
    numeric_weight: float,
    ignore_index: int = -100,
    sample_latent: bool = True,
    adversary: GradeAngleAdversaryHead | None = None,
    grade_adversary_weight: float = 0.0,
    grade_adversary_alpha: float = 1.0,
    angle_min: float = 0.0,
    angle_max: float = 70.0,
    grade_min: float = 10.0,
    grade_max: float = 33.0,
    free_bits: float = 0.0,
    adversary_in_total: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    prepared = prepare_cvae_training_batch(
        batch_samples, eos_ids=eos_ids, device=device, ignore_index=ignore_index
    )
    out = model(
        encoder_batch=prepared["encoder_batch"],
        decoder_batch=prepared["decoder_input_batch"],
        angle=prepared["angle"],
        grade=prepared["grade"],
        sample_latent=sample_latent,
    )

    cat_targets = prepared["categorical_targets"]
    categorical_loss = sum(
        F.cross_entropy(
            out[f"{feat}_logits"].reshape(-1, out[f"{feat}_logits"].shape[-1]),
            cat_targets[f"{feat}_target"].reshape(-1),
            ignore_index=ignore_index,
        )
        for feat in ("type", "role", "hole")
    )

    num_targets = prepared["numeric_targets"]
    mask = prepared["valid_numeric_mask"]
    numeric_loss = sum(
        masked_mse(out[f"{feat}_pred"], num_targets[f"{feat}_target"], mask)
        for feat in ("x", "y", "depth", "orientation_sin", "orientation_cos", "size")
    )

    # kl_unified is a single term: add to total with weight 1.0 — kl_beta and free-bits are baked in.
    # kl_raw is the unweighted KL, used only for logging (kl=X(+Y) diagnostic).
    kl_unified, kl_raw = kl_divergence_loss(
        out["mu"], out["logvar"], kl_beta=kl_beta, reduction="mean", free_bits=free_bits
    )
    total = categorical_loss + numeric_weight * numeric_loss + kl_unified

    adversary_loss_val = 0.0
    if adversary is not None and grade_adversary_weight > 0.0:
        grade_pred, angle_pred = adversary(out["z"], alpha=grade_adversary_alpha)

        # Grade target: route's own grade at its stored angle — exactly what z encodes.
        # z = f(route, angle_A), so grade_A is the directly inferable signal.
        grade_range = max(grade_max - grade_min, 1e-6)
        grade_norm = ((prepared["grade"].to(torch.float32) - grade_min) / grade_range).clamp(0.0, 1.0)
        grade_adv_loss = F.mse_loss(grade_pred, grade_norm)

        angle_range = max(angle_max - angle_min, 1e-6)
        angle_norm = ((prepared["angle"].to(torch.float32) - angle_min) / angle_range).clamp(0.0, 1.0)
        angle_adv_loss = F.mse_loss(angle_pred, angle_norm)

        adversary_loss = grade_adv_loss + angle_adv_loss

        adversary_loss_val = float(adversary_loss.detach().cpu())
        if adversary_in_total:
            # Training only: GRL on z reverses this term's gradient for encoder/bottleneck —
            # adversary head learns to predict grade; encoder learns to prevent prediction.
            # Excluded from validation total: a better-disentangled model has HIGHER adversary
            # loss (encoder won), which would wrongly make val_total look worse.
            total = total + grade_adversary_weight * adversary_loss

    stats = {
        "total": float(total.detach().cpu()),
        "categorical": float(categorical_loss.detach().cpu()),
        "numeric": float(numeric_loss.detach().cpu()),
        "kl_raw": float(kl_raw.detach().cpu()),
        "adversary": adversary_loss_val,
    }
    return total, stats


def _run_epoch(
    model: RouteConditionalVAE,
    samples: list[Any],
    *,
    optimizer: AdamW | None,
    device: torch.device,
    eos_ids: DecoderEOSIds,
    batch_size: int,
    kl_beta: float,
    numeric_weight: float,
    grad_clip_norm: float,
    sample_latent: bool,
    adversary: GradeAngleAdversaryHead | None = None,
    adversary_optimizer: AdamW | None = None,
    grade_adversary_weight: float = 0.0,
    grade_adversary_alpha: float = 1.0,
    angle_min: float = 0.0,
    angle_max: float = 70.0,
    grade_min: float = 10.0,
    grade_max: float = 33.0,
    free_bits: float = 0.0,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)
    if adversary is not None:
        adversary.train(is_train)

    loss_sum = cat_sum = num_sum = kl_raw_sum = adv_sum = 0.0
    batches = 0

    for batch_samples in iter_minibatches(samples, batch_size, shuffle=is_train):
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if adversary_optimizer is not None:
                adversary_optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            loss, stats = _compute_batch_losses(
                model, batch_samples,
                device=device, eos_ids=eos_ids,
                kl_beta=kl_beta, numeric_weight=numeric_weight,
                sample_latent=sample_latent,
                adversary=adversary,
                grade_adversary_weight=grade_adversary_weight,
                grade_adversary_alpha=grade_adversary_alpha,
                angle_min=angle_min, angle_max=angle_max,
                grade_min=grade_min, grade_max=grade_max,
                free_bits=free_bits,
                adversary_in_total=is_train,  # adversary excluded from val total (see _compute_batch_losses)
            )
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if adversary_optimizer is not None:
                    nn.utils.clip_grad_norm_(adversary.parameters(), grad_clip_norm)
                    adversary_optimizer.step()

        loss_sum += stats["total"]
        cat_sum += stats["categorical"]
        num_sum += stats["numeric"]
        kl_raw_sum += stats["kl_raw"]
        adv_sum += stats["adversary"]
        batches += 1

    if batches == 0:
        return EpochMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    return EpochMetrics(
        total_loss=loss_sum / batches,
        categorical_loss=cat_sum / batches,
        numeric_loss=num_sum / batches,
        kl_raw=kl_raw_sum / batches,
        adversary_loss=adv_sum / batches,
        num_batches=batches,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train route conditional VAE.")
    parser.add_argument("--db-path", type=str, default=str(PROJECT_ROOT / "data/raw/kilter_database.sqlite"))
    parser.add_argument("--metadata-source", type=str, default="kilter_board_csv")
    parser.add_argument("--metadata-product-id", type=int, default=1)
    parser.add_argument("--require-full-metadata", action="store_true")
    parser.add_argument("--max-routes", type=int, default=1000000)
    parser.add_argument("--min-holds", type=int, default=1)
    parser.add_argument("--cache-path", type=str, default=str(PROJECT_ROOT / "data/preprocessed_routes_cache.pt"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--numeric-weight", type=float, default=0.25)
    parser.add_argument("--kl-beta", type=float, default=0.1)
    parser.add_argument("--encoder-use-cond-adaln", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-absolute-pos", action=argparse.BooleanOptionalAction, default=True,
        help="Include absolute x/y sinusoidal embeddings in hold tokens. "
             "Set --no-use-absolute-pos to ablate (keep only deltas + knn for spatial info).",
    )
    parser.add_argument(
        "--use-type-feature", action=argparse.BooleanOptionalAction, default=True,
        help="Include hold-type categorical embedding and type fractions in shape_desc. "
             "Set --no-use-type-feature to ablate (forces z to encode geometry, not hold type).",
    )
    parser.add_argument(
        "--route-pool-mode", type=str, default="attention", choices=["attention", "cls"],
        help="Pooling strategy for the route embedding fed to the bottleneck. "
             "'attention' = learned query attends over hold tokens (recommended; avoids shape_desc shortcut). "
             "'cls' = shape/CLS token (fast; risks shortcutting via pre-computed shape_desc stats).",
    )
    # Encoder architecture (saved in checkpoint — must be restored exactly for resume/inference)
    parser.add_argument("--encoder-d-model", type=int, default=256,
                        help="Encoder transformer model dimension. Must be divisible by --encoder-nhead.")
    parser.add_argument("--encoder-nhead", type=int, default=8,
                        help="Number of encoder attention heads. encoder-d-model / encoder-nhead = dims per head.")
    parser.add_argument("--encoder-num-layers", type=int, default=6,
                        help="Number of encoder transformer layers.")
    parser.add_argument("--encoder-dim-feedforward", type=int, default=1024,
                        help="Encoder FFN hidden dim. Recommended: 4 × encoder-d-model.")
    parser.add_argument("--decoder-use-cond-adaln", action=argparse.BooleanOptionalAction, default=True)
    # Decoder architecture
    parser.add_argument("--decoder-d-model", type=int, default=128,
                        help="Decoder transformer model dimension (kept smaller than encoder).")
    parser.add_argument("--decoder-num-layers", type=int, default=4,
                        help="Number of decoder transformer layers.")
    parser.add_argument("--decoder-dim-feedforward", type=int, default=512,
                        help="Decoder FFN hidden dim. Recommended: 4 × decoder-d-model.")
    parser.add_argument("--kl-warmup-epochs", type=int, default=None,
                        help="Linear KL warmup epochs. Defaults to 40%% of total epochs.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default=str(PROJECT_ROOT / "data/route_cvae.pt"))
    parser.add_argument("--loss-plot-path", type=str, default=str(PROJECT_ROOT / "data/route_cvae_loss_curve.png"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--early-stop-delta", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--grade-adversary-weight", type=float, default=0.0,
                        help="Weight for grade/angle adversarial disentanglement loss (0 = disabled).")
    parser.add_argument("--grade-adversary-alpha", type=float, default=1.0,
                        help="Gradient reversal layer scale factor for the adversary head.")
    parser.add_argument("--free-bits", type=float, default=0.0,
                        help="Minimum KL per latent dimension (nats). Prevents posterior collapse by "
                             "clamping each dimension's KL from below before summing. "
                             "Recommended: 0.5 nats/dim.")
    args = parser.parse_args()

    if args.kl_warmup_epochs is None:
        args.kl_warmup_epochs = max(1, int(math.ceil(0.40 * args.epochs)))
    if args.kl_warmup_epochs < 0:
        raise ValueError("--kl-warmup-epochs must be >= 0")
    if args.early_stop_delta is not None and args.early_stop_delta < 0:
        raise ValueError("--early-stop-delta must be >= 0")
    if args.early_stop_patience < 1:
        raise ValueError("--early-stop-patience must be >= 1")

    torch.manual_seed(args.seed)
    device = select_device()
    print(f"Using device: {device}")

    samples, vocabs = _load_or_build_samples_and_vocabs(args)
    train_samples, val_samples, test_samples = split_route_samples(
        samples, train_ratio=0.8, val_ratio=0.1, seed=args.seed
    )
    print(f"Routes: total={len(samples)} train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    # Auto-detect shape_desc_dim from data so the model always matches the cache.
    shape_desc_dim = int(samples[0].tokens["shape_desc"].shape[0])
    print(f"shape_desc_dim={shape_desc_dim} (auto-detected from cache)")
    # Save back so it is persisted in the checkpoint args.
    args.shape_desc_dim = shape_desc_dim

    model, eos_ids, norm_ranges = _build_model(
        vocabs, device,
        latent_dim=args.latent_dim,
        encoder_use_cond_adaln=args.encoder_use_cond_adaln,
        encoder_d_model=args.encoder_d_model,
        encoder_nhead=args.encoder_nhead,
        encoder_num_layers=args.encoder_num_layers,
        encoder_dim_feedforward=args.encoder_dim_feedforward,
        decoder_use_cond_adaln=args.decoder_use_cond_adaln,
        decoder_d_model=args.decoder_d_model,
        decoder_num_layers=args.decoder_num_layers,
        decoder_dim_feedforward=args.decoder_dim_feedforward,
        use_absolute_pos=args.use_absolute_pos,
        use_type_feature=args.use_type_feature,
        shape_desc_dim=shape_desc_dim,
        route_pool_mode=args.route_pool_mode,
    )

    max_seq_len = model.decoder.cfg.max_seq_len
    skipped = []
    for split_name, split_samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        filtered, n_skipped = filter_samples_by_decoder_max_len(split_samples, max_seq_len=max_seq_len)
        if split_name == "train":
            train_samples = filtered
        elif split_name == "val":
            val_samples = filtered
        else:
            test_samples = filtered
        skipped.append((split_name, n_skipped))

    total_skipped = sum(n for _, n in skipped)
    if total_skipped:
        detail = " ".join(f"{name}={n}" for name, n in skipped)
        print(f"Skipped overlength routes (max_seq_len={max_seq_len}): {detail}")
        print(f"After filtering: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    if not train_samples:
        raise ValueError("No training samples remain after max_seq_len filtering.")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    adversary: GradeAngleAdversaryHead | None = None
    adversary_optimizer: AdamW | None = None
    if args.grade_adversary_weight > 0.0:
        adversary = GradeAngleAdversaryHead(args.latent_dim).to(device)
        adversary_optimizer = AdamW(adversary.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Grade+angle adversary enabled (weight={args.grade_adversary_weight}, alpha={args.grade_adversary_alpha})")

    best_val = math.inf
    start_epoch = 1
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_path = Path(args.resume_path) if args.resume_path else checkpoint_path
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if adversary is not None and "adversary_state_dict" in resume_ckpt:
            adversary.load_state_dict(resume_ckpt["adversary_state_dict"])
        if adversary_optimizer is not None and "adversary_optimizer_state_dict" in resume_ckpt:
            adversary_optimizer.load_state_dict(resume_ckpt["adversary_optimizer_state_dict"])
        if adversary is not None and "adversary_state_dict" in resume_ckpt:
            adversary.load_state_dict(resume_ckpt["adversary_state_dict"])
        if adversary_optimizer is not None and "adversary_optimizer_state_dict" in resume_ckpt:
            adversary_optimizer.load_state_dict(resume_ckpt["adversary_optimizer_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_val = float(resume_ckpt.get("best_val", math.inf))
        print(f"Resumed from {resume_path} (next_epoch={start_epoch}, best_val={best_val:.4f})")

    epoch_history: list[int] = []
    train_total_history: list[float] = []
    val_total_history: list[float] = []       # recon + kl_w + floor_pen (adversary excluded — see _compute_batch_losses)
    train_recon_history: list[float] = []
    val_recon_history: list[float] = []
    train_kl_history: list[float] = []        # kl_beta * kl_loss (full weighted KL, not excess-only)
    val_kl_history: list[float] = []
    early_stop_stable_epochs = 0

    for epoch in range(start_epoch, args.epochs + 1):
        kl_beta = _compute_kl_beta(
            epoch=epoch, target_kl_beta=args.kl_beta, kl_warmup_epochs=args.kl_warmup_epochs
        )
        train_m = _run_epoch(
            model, train_samples,
            optimizer=optimizer, device=device, eos_ids=eos_ids,
            batch_size=args.batch_size, kl_beta=kl_beta,
            numeric_weight=args.numeric_weight, grad_clip_norm=args.grad_clip_norm,
            sample_latent=True,
            adversary=adversary, adversary_optimizer=adversary_optimizer,
            grade_adversary_weight=args.grade_adversary_weight,
            grade_adversary_alpha=args.grade_adversary_alpha,
            free_bits=args.free_bits,
            **norm_ranges,
        )
        with torch.no_grad():
            val_m = _run_epoch(
                model, val_samples,
                optimizer=None, device=device, eos_ids=eos_ids,
                batch_size=args.batch_size, kl_beta=kl_beta,
                numeric_weight=args.numeric_weight, grad_clip_norm=args.grad_clip_norm,
                sample_latent=False,
                adversary=adversary,
                grade_adversary_weight=args.grade_adversary_weight,
                grade_adversary_alpha=args.grade_adversary_alpha,
                free_bits=args.free_bits,
                **norm_ranges,
            )

        train_recon = train_m.categorical_loss + args.numeric_weight * train_m.numeric_loss
        val_recon = val_m.categorical_loss + args.numeric_weight * val_m.numeric_loss
        kl_floor = args.free_bits * args.latent_dim

        # kl_w is the actual KL contribution to total — derived rather than stored separately.
        # Identity (train): total = recon + kl_w + adv_weight*adv
        # Identity (val):   total = recon + kl_w                   (adversary excluded from val total)
        # kl=X(+Y): X = raw unweighted KL, Y = excess above kl_floor (diagnostic).
        train_kl_w = train_m.total_loss - train_recon - args.grade_adversary_weight * train_m.adversary_loss
        val_kl_w = val_m.total_loss - val_recon
        train_kl_excess = max(train_m.kl_raw - kl_floor, 0.0)
        val_kl_excess = max(val_m.kl_raw - kl_floor, 0.0)

        adv_str = (
            f" adv={train_m.adversary_loss:.4f}"
            if args.grade_adversary_weight > 0.0
            else ""
        )
        val_adv_str = (
            f" adv={val_m.adversary_loss:.4f}"
            if args.grade_adversary_weight > 0.0
            else ""
        )
        print(
            f"Epoch {epoch:03d} | kl_beta={kl_beta:.6f} "
            f"train total={train_m.total_loss:.4f} recon={train_recon:.4f} kl_w={train_kl_w:.4f} "
            f"cat={train_m.categorical_loss:.4f} num={train_m.numeric_loss:.4f} "
            f"kl={train_m.kl_raw:.4f}(+{train_kl_excess:.4f}){adv_str} | "
            f"val total={val_m.total_loss:.4f} recon={val_recon:.4f} kl_w={val_kl_w:.4f} "
            f"cat={val_m.categorical_loss:.4f} num={val_m.numeric_loss:.4f} "
            f"kl={val_m.kl_raw:.4f}(+{val_kl_excess:.4f}){val_adv_str}"
        )

        epoch_history.append(epoch)
        train_total_history.append(train_m.total_loss)
        val_total_history.append(val_m.total_loss)
        train_recon_history.append(train_recon)
        val_recon_history.append(val_recon)
        train_kl_history.append(train_kl_w)
        val_kl_history.append(val_kl_w)

        # val_m.total_loss = recon + kl_w + floor_pen (adversary excluded at compute time).
        # This is the right checkpoint criterion: lower = better reconstruction and more active dims.
        if val_m.total_loss < best_val:
            best_val = val_m.total_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "args": vars(args),
            }
            if adversary is not None:
                checkpoint["adversary_state_dict"] = adversary.state_dict()
            if adversary_optimizer is not None:
                checkpoint["adversary_optimizer_state_dict"] = adversary_optimizer.state_dict()
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint (val={best_val:.4f}): {checkpoint_path}")

        if args.early_stop_delta is not None and len(val_total_history) >= 2:
            val_delta = abs(val_total_history[-1] - val_total_history[-2])
            if val_delta < args.early_stop_delta:
                early_stop_stable_epochs += 1
                print(
                    f"Early-stopping monitor | val_delta={val_delta:.6f} < {args.early_stop_delta:.6f} "
                    f"(stable_epochs={early_stop_stable_epochs}/{args.early_stop_patience})"
                )
                if early_stop_stable_epochs >= args.early_stop_patience:
                    print(
                        f"Early stopping triggered: |Δval_total| < {args.early_stop_delta} "
                        f"for {args.early_stop_patience} consecutive epoch(s)."
                    )
                    break
            else:
                early_stop_stable_epochs = 0

    _save_loss_curve(
        epochs=epoch_history,
        train_total=train_total_history,
        val_total=val_total_history,
        train_recon=train_recon_history,
        val_recon=val_recon_history,
        train_kl=train_kl_history,
        val_kl=val_kl_history,
        output_path=Path(args.loss_plot_path),
    )

    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    with torch.no_grad():
        test_m = _run_epoch(
            model, test_samples,
            optimizer=None, device=device, eos_ids=eos_ids,
            batch_size=args.batch_size, kl_beta=args.kl_beta,
            numeric_weight=args.numeric_weight, grad_clip_norm=args.grad_clip_norm,
            sample_latent=False,
            adversary=adversary,
            grade_adversary_weight=args.grade_adversary_weight,
            grade_adversary_alpha=args.grade_adversary_alpha,
            free_bits=args.free_bits,
            **norm_ranges,
        )
    test_recon = test_m.categorical_loss + args.numeric_weight * test_m.numeric_loss
    # Test uses is_train=False → adversary excluded from total. Identity: total = recon + kl_w.
    test_kl_w = test_m.total_loss - test_recon
    test_kl_floor = args.free_bits * args.latent_dim
    test_kl_excess = max(test_m.kl_raw - test_kl_floor, 0.0)
    test_adv_str = (
        f" adv={test_m.adversary_loss:.4f}"
        if args.grade_adversary_weight > 0.0
        else ""
    )
    print(
        f"Test | total={test_m.total_loss:.4f} recon={test_recon:.4f} kl_w={test_kl_w:.4f} "
        f"cat={test_m.categorical_loss:.4f} num={test_m.numeric_loss:.4f} "
        f"kl={test_m.kl_raw:.4f}(+{test_kl_excess:.4f}){test_adv_str}"
    )


if __name__ == "__main__":
    main()
