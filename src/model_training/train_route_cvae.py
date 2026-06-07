from __future__ import annotations

import argparse
import math
import random
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
    RouteConditionalVAE,
    RouteVAEBottleneck,
    RouteVAEBottleneckConfig,
    kl_divergence_loss,
)
from model_training.route_vae_decoder import RouteTransformerDecoder, RouteVAEDecoderConfig
from model_training.route_vae_encoder import RouteTransformerEncoder, RouteMlpEncoderConfig, RouteMlpEncoder
from model_training.route_vae_parallel_decoder import RouteParallelDecoder, RouteVAEParallelDecoderConfig
from model_training.training_utils import DecoderEOSIds, masked_mse, prepare_cvae_training_batch


@dataclass
class EpochMetrics:
    total_loss: float
    categorical_loss: float
    numeric_loss: float
    kl_raw: float       # unweighted KL — for kl=X(+Y) diagnostics, not the loss contribution
    num_batches: int
    pairwise_loss: float = 0.0      # unweighted pairwise distance MSE (0 when pairwise_weight=0)
    move_vector_loss: float = 0.0   # unweighted move-vector MSE (0 when move_vector_weight=0)


def _compute_kl_beta(
    *,
    epoch: int,
    target_kl_beta: float,
    kl_warmup_epochs: int,
    kl_warmup_offset: int = 0,
    kl_cycle_epochs: int = 0,
    kl_cycle_ratio: float = 0.5,
) -> float:
    """Compute the KL beta for this epoch.

    Cyclic annealing (kl_cycle_epochs > 0): each cycle linearly ramps beta from
    0 → target over the first (cycle_epochs × ratio) epochs, then holds at target
    for the remainder.  Repeats every kl_cycle_epochs epochs throughout training.
    This forces repeated z-expansion phases where the decoder must learn to use z.

    Linear warmup fallback (kl_cycle_epochs == 0): kl_beta stays at 0 for the
    first kl_warmup_offset epochs, then ramps linearly from 0 → target over the
    next kl_warmup_epochs epochs, then holds at target.
    """
    if kl_cycle_epochs > 0:
        cycle_pos = (epoch - 1) % kl_cycle_epochs   # 0-indexed within current cycle
        ramp_len = max(1, int(kl_cycle_epochs * kl_cycle_ratio))
        if cycle_pos < ramp_len:
            return float(target_kl_beta) * (cycle_pos / ramp_len)
        return float(target_kl_beta)
    if epoch <= kl_warmup_offset:
        return 0.0
    if kl_warmup_epochs <= 0:
        return float(target_kl_beta)
    ramp_epoch = epoch - kl_warmup_offset
    ratio = min(max(ramp_epoch, 0), kl_warmup_epochs) / float(kl_warmup_epochs)
    return float(target_kl_beta) * ratio


def _compute_ss_ratio(*, epoch: int, target_ss_ratio: float, ss_warmup_epochs: int) -> float:
    """Linear warmup for scheduled sampling ratio: 0 → target over ss_warmup_epochs."""
    if target_ss_ratio <= 0.0:
        return 0.0
    if ss_warmup_epochs <= 0:
        return float(target_ss_ratio)
    ratio = min(max(epoch, 0), ss_warmup_epochs) / float(ss_warmup_epochs)
    return float(target_ss_ratio) * ratio


def _compute_move_vector_loss(
    x_pred: torch.Tensor,
    y_pred: torch.Tensor,
    x_tgt: torch.Tensor,
    y_tgt: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE on consecutive directed move vectors (Δx, Δy) in sequence order.

    Unlike pairwise distance loss (which is rotation-invariant and cannot distinguish
    left-leaning from right-leaning routes), move vectors directly supervise the
    directional shape features that ``step_x_var`` and ``move_size`` measure.

    Args:
        x_pred, y_pred: predicted positions [B, L], normalised to [0, 1].
        x_tgt,  y_tgt:  target  positions  [B, L], normalised to [0, 1].
        valid_mask:     [B, L] bool — True for non-padded positions.

    Returns:
        Scalar mean loss (0.0 when fewer than 2 valid positions per sample).
    """
    if x_pred.shape[1] < 2:
        return x_pred.new_tensor(0.0)

    dx_pred = x_pred[:, 1:] - x_pred[:, :-1]   # [B, L-1]
    dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
    dx_tgt  = x_tgt[:, 1:]  - x_tgt[:, :-1]
    dy_tgt  = y_tgt[:, 1:]  - y_tgt[:, :-1]

    # Include a pair only when both adjacent positions are non-padded.
    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]   # [B, L-1]
    if not pair_mask.any():
        return x_pred.new_tensor(0.0)

    return masked_mse(dx_pred, dx_tgt, pair_mask) + masked_mse(dy_pred, dy_tgt, pair_mask)


def _compute_mask_max(*, epoch: int, warmup_epochs: int) -> float:
    """Return the upper bound for mask-rate sampling at this epoch.

    During warmup (epoch < warmup_epochs) the maximum mask rate ramps linearly
    from 0 → 1.  After warmup it stays at 1.0 so each training batch samples
    mask_rate ~ Uniform[0, 1).
    """
    if warmup_epochs <= 0:
        return 1.0
    return min(epoch / warmup_epochs, 1.0)


def _mix_predictions_into_batch(
    gt_batch: dict[str, torch.Tensor],
    model_out: dict[str, torch.Tensor],
    ss_ratio: float,
    dec_cfg: RouteVAEDecoderConfig,
) -> dict[str, torch.Tensor]:
    """Replace decoder input positions 1..L-1 with own predictions from position i-1.

    For each position i ∈ [1, L-1], flips a coin: with probability ss_ratio use the
    model's prediction from output position i-1 (denormalised to raw batch values);
    otherwise keep the ground-truth input.

    Categorical argmax excludes the EOS slot so EOS never feeds back mid-sequence.
    Continuous values are denormalised identically to _autoregressive_forward.
    """
    B, L = gt_batch["type_encoded_id"].shape
    device = gt_batch["type_encoded_id"].device
    if L <= 1:
        return gt_batch

    # Output positions 0..L-2 become candidate inputs at positions 1..L-1.
    # Exclude EOS from argmax (last logit index) to prevent EOS propagating mid-sequence.
    pred_type = model_out["type_logits"][:, :L-1, :-1].argmax(-1)   # [B, L-1]
    pred_role = model_out["role_logits"][:, :L-1, :-1].argmax(-1)
    pred_hole = model_out["hole_logits"][:, :L-1, :-1].argmax(-1)

    pred_x       = model_out["x_pred"][:, :L-1] * (dec_cfg.x_max - dec_cfg.x_min) + dec_cfg.x_min
    pred_y       = model_out["y_pred"][:, :L-1] * (dec_cfg.y_max - dec_cfg.y_min) + dec_cfg.y_min
    pred_ori_sin = model_out["orientation_sin_pred"][:, :L-1] * 2.0 - 1.0
    pred_ori_cos = model_out["orientation_cos_pred"][:, :L-1] * 2.0 - 1.0
    pred_size    = model_out["size_pred"][:, :L-1] * 3.0 + 2.0

    use_pred = torch.rand(B, L - 1, device=device) < ss_ratio  # [B, L-1]

    def _mix_cat(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        out = gt.clone()
        out[:, 1:] = torch.where(use_pred, pred, gt[:, 1:])
        return out

    def _mix_float(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        out = gt.clone()
        out[:, 1:] = torch.where(use_pred, pred.to(gt.dtype), gt[:, 1:])
        return out

    return {
        **gt_batch,
        "type_encoded_id":  _mix_cat(gt_batch["type_encoded_id"],   pred_type),
        "role_encoded_id":  _mix_cat(gt_batch["role_encoded_id"],   pred_role),
        "hole_encoded_id":  _mix_cat(gt_batch["hole_encoded_id"],   pred_hole),
        "x":               _mix_float(gt_batch["x"],                pred_x),
        "y":               _mix_float(gt_batch["y"],                pred_y),
        "orientation_sin": _mix_float(gt_batch["orientation_sin"],  pred_ori_sin),
        "orientation_cos": _mix_float(gt_batch["orientation_cos"],  pred_ori_cos),
        "size":            _mix_float(gt_batch["size"],             pred_size),
    }


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


def _set_decoder_self_attn_frozen(model: RouteConditionalVAE, frozen: bool) -> int:
    """Freeze or unfreeze self-attention in every decoder layer.

    Supports both the old nn.TransformerDecoder backbone (model.decoder.decoder)
    and the new AdaLN decoder (model.decoder.layers with AdaLNTransformerLayer).
    Returns the number of affected parameters (0 for parallel decoder or any
    decoder without identifiable self-attention layers).
    """
    # New AdaLN decoder: model.decoder.layers is a ModuleList of AdaLNTransformerLayer
    layers_direct = getattr(model.decoder, "layers", None)
    if layers_direct is not None and hasattr(layers_direct, "__iter__"):
        count = 0
        for layer in layers_direct:
            sa = getattr(layer, "self_attn", None)
            if sa is not None:
                for p in sa.parameters():
                    p.requires_grad_(not frozen)
                    count += p.numel()
        if count > 0:
            return count

    # Legacy fallback: old nn.TransformerDecoder backbone (model.decoder.decoder)
    transformer = getattr(model.decoder, "decoder", None)
    if transformer is None or not hasattr(transformer, "layers"):
        return 0
    count = 0
    for layer in transformer.layers:
        sa = getattr(layer, "self_attn", None)
        if sa is not None:
            for p in sa.parameters():
                p.requires_grad_(not frozen)
                count += p.numel()
    return count


def _build_model(
    vocabs: Any,
    device: torch.device,
    *,
    latent_dim: int,
    encoder_d_model: int,
    encoder_nhead: int,
    encoder_num_layers: int,
    encoder_dim_feedforward: int,
    decoder_d_model: int,
    decoder_num_layers: int,
    decoder_dim_feedforward: int,
    use_absolute_pos: bool = True,
    use_type_feature: bool = True,
    use_cls_token: bool = False,
    shape_desc_dim: int = 9,
    use_mlp_encoder: bool = False,
    decoder_token_dropout: float = 0.0,
    decoder_mask_rate: float = 0.0,
    use_parallel_decoder: bool = False,
) -> tuple[RouteConditionalVAE, DecoderEOSIds]:
    # --- Build encoder ---
    if use_mlp_encoder:
        mlp_enc_cfg = RouteMlpEncoderConfig(
            shape_desc_dim=shape_desc_dim,
            d_model=encoder_d_model,
            num_layers=encoder_num_layers,
        )
        encoder: RouteTransformerEncoder | RouteMlpEncoder = RouteMlpEncoder(mlp_enc_cfg)
        enc_d_model = encoder_d_model
        # Vocab sizes still needed for decoder heads; borrow from vocabs.
        _vcfg = vocabs.to_transformer_config(d_model=encoder_d_model, nhead=8,
                                              num_layers=1, dim_feedforward=256)
        type_vocab_size   = _vcfg.type_vocab_size
        role_vocab_size   = _vcfg.role_vocab_size
        hole_id_vocab_size = _vcfg.hole_id_vocab_size
        x_min, x_max = _vcfg.x_min, _vcfg.x_max
        y_min, y_max = _vcfg.y_min, _vcfg.y_max
    else:
        if encoder_d_model % encoder_nhead != 0:
            raise ValueError(
                f"--encoder-d-model ({encoder_d_model}) must be divisible by --encoder-nhead ({encoder_nhead})"
            )
        enc_cfg = vocabs.to_transformer_config(
            d_model=encoder_d_model,
            nhead=encoder_nhead,
            num_layers=encoder_num_layers,
            dim_feedforward=encoder_dim_feedforward,
        )
        enc_cfg.use_absolute_pos = use_absolute_pos
        enc_cfg.use_type_feature = use_type_feature
        enc_cfg.use_cls_token = use_cls_token
        if use_cls_token:
            enc_cfg.shape_desc_dim = shape_desc_dim
        encoder = RouteTransformerEncoder(enc_cfg)
        enc_d_model        = enc_cfg.d_model
        type_vocab_size    = enc_cfg.type_vocab_size
        role_vocab_size    = enc_cfg.role_vocab_size
        hole_id_vocab_size = enc_cfg.hole_id_vocab_size
        x_min, x_max = enc_cfg.x_min, enc_cfg.x_max
        y_min, y_max = enc_cfg.y_min, enc_cfg.y_max

    bottleneck_cfg = RouteVAEBottleneckConfig(
        encoder_embedding_dim=enc_d_model,
        latent_dim=latent_dim,
        hidden_dim=enc_d_model,
    )

    if use_parallel_decoder:
        dec_cfg: RouteVAEParallelDecoderConfig | RouteVAEDecoderConfig = RouteVAEParallelDecoderConfig(
            type_vocab_size=type_vocab_size,
            role_vocab_size=role_vocab_size,
            hole_id_vocab_size=hole_id_vocab_size,
            latent_dim=latent_dim,
            d_model=decoder_d_model,
            mlp_depth=decoder_num_layers,
        )
        decoder = RouteParallelDecoder(dec_cfg)
    else:
        if decoder_d_model % 8 != 0:
            raise ValueError(f"--decoder-d-model ({decoder_d_model}) must be divisible by nhead=8")
        dec_cfg = RouteVAEDecoderConfig(
            type_vocab_size=type_vocab_size,
            role_vocab_size=role_vocab_size,
            hole_id_vocab_size=hole_id_vocab_size,
            latent_dim=latent_dim,
            d_model=decoder_d_model,
            num_layers=decoder_num_layers,
            dim_feedforward=decoder_dim_feedforward,
            use_absolute_pos=use_absolute_pos,
            use_type_feature=use_type_feature,
            token_dropout=decoder_token_dropout,
            mask_rate=decoder_mask_rate,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
        )
        decoder = RouteTransformerDecoder(dec_cfg)

    model = RouteConditionalVAE(
        encoder=encoder,
        bottleneck=RouteVAEBottleneck(bottleneck_cfg),
        decoder=decoder,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    if use_mlp_encoder:
        print(
            f"Encoder (MLP baseline): shape_desc_dim={mlp_enc_cfg.shape_desc_dim} "
            f"d_model={mlp_enc_cfg.d_model} layers={mlp_enc_cfg.num_layers}"
        )
    else:
        print(
            f"Encoder: d_model={enc_cfg.d_model} nhead={enc_cfg.nhead} "
            f"layers={enc_cfg.num_layers} ffn={enc_cfg.dim_feedforward}"
        )
    if use_parallel_decoder:
        print(
            f"Decoder (parallel MLP): d_model={dec_cfg.d_model} "
            f"mlp_depth={dec_cfg.mlp_depth}"
        )
    elif decoder_mask_rate > 0.0:
        print(
            f"Decoder (masked bidirectional + AdaLN): d_model={dec_cfg.d_model} nhead={dec_cfg.nhead} "
            f"layers={dec_cfg.num_layers} ffn={dec_cfg.dim_feedforward} mask_rate={dec_cfg.mask_rate}"
        )
    else:
        print(
            f"Decoder (autoregressive + AdaLN): d_model={dec_cfg.d_model} nhead={dec_cfg.nhead} "
            f"layers={dec_cfg.num_layers} ffn={dec_cfg.dim_feedforward}"
        )
    print(f"Total parameters: {total_params:,}")

    eos_ids = DecoderEOSIds(
        type_eos_id=model.decoder.type_eos_id,
        role_eos_id=model.decoder.role_eos_id,
        hole_eos_id=model.decoder.hole_eos_id,
    )
    return model, eos_ids


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
        "min_quality_average": args.min_quality_average,
        "min_ascensionist_count": args.min_ascensionist_count,
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
        min_quality_average=args.min_quality_average,
        min_ascensionist_count=args.min_ascensionist_count,
    )
    torch.save({"cache_key": cache_key, "samples": samples, "vocabs": vocabs}, cache_path)
    print(f"Saved preprocessed dataset cache: {cache_path}")
    return samples, vocabs



def _autoregressive_forward(
    model: RouteConditionalVAE,
    prepared: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Autoregressive forward pass with no teacher forcing.

    Encodes normally (mean z), then decodes one token at a time, feeding each
    step's predicted token back as the next step's input.  All B sequences in
    the batch are processed in parallel; only the L time-steps are sequential.

    Denormalisation constants must match training_utils.prepare_cvae_training_batch:
      x   : raw ∈ [x_min, x_max]  →  target = (x − x_min) / (x_max − x_min)
      y   : raw ∈ [y_min, y_max]  →  target = (y − y_min) / (y_max − y_min)
      ori : raw ∈ [−1, 1]         →  target = (ori + 1) / 2
      size: raw ∈ [2, 5]          →  target = (size − 2) / 3
    """
    enc_out = model.encoder(prepared["encoder_batch"])
    bn_out  = model.bottleneck(enc_out["route_embedding"], sample_latent=False)
    z       = bn_out["z"]

    dec_cfg    = model.decoder.cfg
    target_len = prepared["categorical_targets"]["type_target"].shape[1]  # L + 1
    B          = z.shape[0]

    # Running decoder input — raw feature values updated at each step.
    # Position 0 is the BOS slot (zeros, overwritten by bos_embedding inside decoder).
    seq = {
        "type_encoded_id": torch.zeros(B, target_len, dtype=torch.long, device=device),
        "role_encoded_id": torch.zeros(B, target_len, dtype=torch.long, device=device),
        "hole_encoded_id": torch.zeros(B, target_len, dtype=torch.long, device=device),
        "x":               torch.zeros(B, target_len,                   device=device),
        "y":               torch.zeros(B, target_len,                   device=device),
        "orientation_sin": torch.zeros(B, target_len,                   device=device),
        "orientation_cos": torch.ones( B, target_len,                   device=device),
        "size":            torch.full( (B, target_len), 3.5,            device=device),
        "padding_mask":    torch.zeros(B, target_len, dtype=torch.bool, device=device),
    }

    type_logits_list:  list[torch.Tensor] = []
    role_logits_list:  list[torch.Tensor] = []
    hole_logits_list:  list[torch.Tensor] = []
    x_pred_list:       list[torch.Tensor] = []
    y_pred_list:       list[torch.Tensor] = []
    ori_sin_pred_list: list[torch.Tensor] = []
    ori_cos_pred_list: list[torch.Tensor] = []
    size_pred_list:    list[torch.Tensor] = []

    for t in range(target_len):
        batch_t = {k: v[:, :t + 1] for k, v in seq.items()}
        dec_out = model.decoder(batch_t, z=z)

        type_logits_list .append(dec_out["type_logits"]         [:, t, :])
        role_logits_list .append(dec_out["role_logits"]         [:, t, :])
        hole_logits_list .append(dec_out["hole_logits"]         [:, t, :])
        x_pred_list      .append(dec_out["x_pred"]              [:, t])
        y_pred_list      .append(dec_out["y_pred"]              [:, t])
        ori_sin_pred_list.append(dec_out["orientation_sin_pred"][:, t])
        ori_cos_pred_list.append(dec_out["orientation_cos_pred"][:, t])
        size_pred_list   .append(dec_out["size_pred"]           [:, t])

        if t + 1 < target_len:
            seq["type_encoded_id"][:, t + 1] = dec_out["type_logits"][:, t, :-1].argmax(-1)
            seq["role_encoded_id"][:, t + 1] = dec_out["role_logits"][:, t, :-1].argmax(-1)
            seq["hole_encoded_id"][:, t + 1] = dec_out["hole_logits"][:, t, :-1].argmax(-1)
            seq["x"]              [:, t + 1] = dec_out["x_pred"]              [:, t] * (dec_cfg.x_max - dec_cfg.x_min) + dec_cfg.x_min
            seq["y"]              [:, t + 1] = dec_out["y_pred"]              [:, t] * (dec_cfg.y_max - dec_cfg.y_min) + dec_cfg.y_min
            seq["orientation_sin"][:, t + 1] = dec_out["orientation_sin_pred"][:, t] * 2.0 - 1.0
            seq["orientation_cos"][:, t + 1] = dec_out["orientation_cos_pred"][:, t] * 2.0 - 1.0
            seq["size"]           [:, t + 1] = dec_out["size_pred"]           [:, t] * 3.0 + 2.0

    return {
        "mu":                       bn_out["mu"],
        "logvar":                   bn_out["logvar"],
        "z":                        z,
        "encoder_route_embedding":  enc_out["route_embedding"],
        "encoder_token_embeddings": enc_out["token_embeddings"],
        "type_logits":              torch.stack(type_logits_list,  dim=1),
        "role_logits":              torch.stack(role_logits_list,  dim=1),
        "hole_logits":              torch.stack(hole_logits_list,  dim=1),
        "x_pred":                   torch.stack(x_pred_list,       dim=1),
        "y_pred":                   torch.stack(y_pred_list,       dim=1),
        "orientation_sin_pred":     torch.stack(ori_sin_pred_list, dim=1),
        "orientation_cos_pred":     torch.stack(ori_cos_pred_list, dim=1),
        "size_pred":                torch.stack(size_pred_list,    dim=1),
    }


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
    free_bits: float = 0.0,
    pairwise_weight: float = 0.0,
    move_vector_weight: float = 0.0,
    hole_loss_weight: float = 1.0,
    autoregressive: bool = False,
    ss_ratio: float = 0.0,
    train_mask_rate: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    prepared = prepare_cvae_training_batch(
        batch_samples, eos_ids=eos_ids, device=device, ignore_index=ignore_index
    )
    is_parallel = not getattr(model.decoder, "is_autoregressive", True)
    is_masked_decoder = (not is_parallel) and getattr(model.decoder.cfg, "mask_rate", 0.0) > 0.0

    if autoregressive and not is_parallel and not is_masked_decoder:
        out = _autoregressive_forward(model, prepared, device=device)
    elif ss_ratio > 0.0 and model.training and not is_parallel and not is_masked_decoder:
        # Two-pass scheduled sampling: first pass (eval, no_grad) gets clean predictions;
        # second pass (train) trains on the mixed ground-truth/own-prediction inputs.
        # Not applicable to the parallel or masked decoder.
        dec_cfg = model.decoder.cfg
        model.eval()
        with torch.no_grad():
            first_out = model(
                encoder_batch=prepared["encoder_batch"],
                decoder_batch=prepared["decoder_input_batch"],
                sample_latent=False,  # use posterior mean so predictions and gradient pass share the same z
            )
        model.train()
        mixed_batch = _mix_predictions_into_batch(
            prepared["decoder_input_batch"], first_out, ss_ratio, dec_cfg
        )
        out = model(
            encoder_batch=prepared["encoder_batch"],
            decoder_batch=mixed_batch,
            sample_latent=sample_latent,
        )
    elif is_masked_decoder:
        # Masked bidirectional: pass unshifted target_batch; masking happens inside the decoder.
        # train_mask_rate overrides cfg.mask_rate when the training loop samples dynamically.
        out = model(
            encoder_batch=prepared["encoder_batch"],
            decoder_batch=prepared["target_batch"],
            sample_latent=sample_latent,
            mask_rate=train_mask_rate,
        )
    else:
        out = model(
            encoder_batch=prepared["encoder_batch"],
            decoder_batch=prepared["decoder_input_batch"],
            sample_latent=sample_latent,
        )

    if is_masked_decoder:
        # Losses computed only at randomly masked positions; visible holds are inputs, not targets.
        # When mask_rate=0 (early warmup) no positions are masked → fall back to all non-padded
        # positions so the loss is never degenerate and warmup epochs still train the model.
        is_masked_out = out["is_masked"]          # [B, L]
        target_batch  = prepared["target_batch"]
        padding_mask  = target_batch["padding_mask"].bool()
        if is_masked_out.any():
            loss_pos = is_masked_out & ~padding_mask  # [B, L]: masked & non-padded
        else:
            loss_pos = ~padding_mask                  # fallback: all non-padded positions

        type_tgt = target_batch["type_encoded_id"].long().masked_fill(~loss_pos, ignore_index)
        role_tgt = target_batch["role_encoded_id"].long().masked_fill(~loss_pos, ignore_index)
        hole_tgt = target_batch["hole_encoded_id"].long().masked_fill(~loss_pos, ignore_index)

        categorical_loss = (
            F.cross_entropy(out["type_logits"].reshape(-1, out["type_logits"].shape[-1]),
                            type_tgt.reshape(-1), ignore_index=ignore_index)
            + F.cross_entropy(out["role_logits"].reshape(-1, out["role_logits"].shape[-1]),
                              role_tgt.reshape(-1), ignore_index=ignore_index)
            + hole_loss_weight * F.cross_entropy(
                out["hole_logits"].reshape(-1, out["hole_logits"].shape[-1]),
                hole_tgt.reshape(-1), ignore_index=ignore_index)
        )

        dec_cfg = model.decoder.cfg
        def _norm(v: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
            return ((v.float() - lo) / (hi - lo)).clamp(0.0, 1.0)

        x_tgt    = _norm(target_batch["x"],               dec_cfg.x_min, dec_cfg.x_max)
        y_tgt    = _norm(target_batch["y"],               dec_cfg.y_min, dec_cfg.y_max)
        ori_sin  = _norm(target_batch["orientation_sin"],  -1.0, 1.0)
        ori_cos  = _norm(target_batch["orientation_cos"],  -1.0, 1.0)
        size_tgt = _norm(target_batch["size"],             2.0, 5.0)

        numeric_loss = (
            masked_mse(out["x_pred"],               x_tgt,    loss_pos)
            + masked_mse(out["y_pred"],             y_tgt,    loss_pos)
            + masked_mse(out["orientation_sin_pred"], ori_sin, loss_pos)
            + masked_mse(out["orientation_cos_pred"], ori_cos, loss_pos)
            + masked_mse(out["size_pred"],          size_tgt, loss_pos)
        )
        # Variables for move-vector loss (all non-padded positions, not just masked).
        _mv_x_tgt  = x_tgt
        _mv_y_tgt  = y_tgt
        _mv_valid  = ~padding_mask
    else:
        cat_targets = prepared["categorical_targets"]
        categorical_loss = sum(
            F.cross_entropy(
                out[f"{feat}_logits"].reshape(-1, out[f"{feat}_logits"].shape[-1]),
                cat_targets[f"{feat}_target"].reshape(-1),
                ignore_index=ignore_index,
            )
            for feat in ("type", "role")
        ) + hole_loss_weight * F.cross_entropy(
            out["hole_logits"].reshape(-1, out["hole_logits"].shape[-1]),
            cat_targets["hole_target"].reshape(-1),
            ignore_index=ignore_index,
        )

        num_targets = prepared["numeric_targets"]
        mask = prepared["valid_numeric_mask"]
        numeric_loss = sum(
            masked_mse(out[f"{feat}_pred"], num_targets[f"{feat}_target"], mask)
            for feat in ("x", "y", "orientation_sin", "orientation_cos", "size")
        )
        # Variables for move-vector loss.
        _mv_x_tgt = num_targets["x_target"]
        _mv_y_tgt = num_targets["y_target"]
        _mv_valid  = mask

    # kl_unified is a single term: add to total with weight 1.0 — kl_beta and free-bits are baked in.
    # kl_raw is the unweighted KL, used only for logging (kl=X(+Y) diagnostic).
    kl_unified, kl_raw = kl_divergence_loss(
        out["mu"], out["logvar"], kl_beta=kl_beta, reduction="mean", free_bits=free_bits
    )
    total = categorical_loss + numeric_weight * numeric_loss + kl_unified

    # Pairwise distance loss: MSE on L2 distances between all valid hold pairs.
    # x/y targets are normalised to [0, 1] (board span), so pairwise distances
    # lie in [0, √2].  Dividing by √2 maps to [0, 1], keeping scale comparable
    # to per-token numeric_loss.  Only upper-triangle pairs are used to avoid
    # double-counting; padded / EOS tokens are excluded via valid_numeric_mask.
    pairwise_val = 0.0
    if pairwise_weight > 0.0:
        _SQRT2 = 2.0 ** 0.5
        if is_masked_decoder:
            # Pairwise over masked positions only (visible holds have trivially accurate x/y).
            x_pred_pw = out["x_pred"]   # [B, L]
            y_pred_pw = out["y_pred"]
            L = x_pred_pw.shape[1]
            triu = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x_pred_pw.device), diagonal=1)
            pair_mask = loss_pos.unsqueeze(2) & loss_pos.unsqueeze(1) & triu.unsqueeze(0)
            dx_pred = x_pred_pw.unsqueeze(2) - x_pred_pw.unsqueeze(1)
            dy_pred = y_pred_pw.unsqueeze(2) - y_pred_pw.unsqueeze(1)
            d_pred  = torch.sqrt(dx_pred ** 2 + dy_pred ** 2 + 1e-8)
            dx_true = x_tgt.unsqueeze(2) - x_tgt.unsqueeze(1)
            dy_true = y_tgt.unsqueeze(2) - y_tgt.unsqueeze(1)
            d_true  = torch.sqrt(dx_true ** 2 + dy_true ** 2 + 1e-8)
        else:
            x_pred_pw = out["x_pred"]                # [B, L+1]
            y_pred_pw = out["y_pred"]                # [B, L+1]
            x_true_pw = num_targets["x_target"]      # [B, L+1]
            y_true_pw = num_targets["y_target"]      # [B, L+1]
            dx_pred = x_pred_pw.unsqueeze(2) - x_pred_pw.unsqueeze(1)
            dy_pred = y_pred_pw.unsqueeze(2) - y_pred_pw.unsqueeze(1)
            d_pred  = torch.sqrt(dx_pred ** 2 + dy_pred ** 2 + 1e-8)
            dx_true = x_true_pw.unsqueeze(2) - x_true_pw.unsqueeze(1)
            dy_true = y_true_pw.unsqueeze(2) - y_true_pw.unsqueeze(1)
            d_true  = torch.sqrt(dx_true ** 2 + dy_true ** 2 + 1e-8)
            Lp1 = x_pred_pw.shape[1]
            triu = torch.triu(
                torch.ones(Lp1, Lp1, dtype=torch.bool, device=x_pred_pw.device), diagonal=1,
            )
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & triu.unsqueeze(0)
        pairwise_loss = masked_mse(d_pred / _SQRT2, d_true / _SQRT2, pair_mask)
        total = total + numeric_weight * pairwise_weight * pairwise_loss
        pairwise_val = float(pairwise_loss.detach().cpu())

    # Move-vector loss: MSE on directed Δx/Δy between consecutive holds.
    # Captures directional shape (step_x_var, move_size) that pairwise distance loss misses.
    # Scaled identically to pairwise: numeric_weight * move_vector_weight * loss.
    move_vector_val = 0.0
    if move_vector_weight > 0.0:
        mv_loss = _compute_move_vector_loss(
            out["x_pred"], out["y_pred"],
            _mv_x_tgt, _mv_y_tgt,
            _mv_valid,
        )
        total = total + numeric_weight * move_vector_weight * mv_loss
        move_vector_val = float(mv_loss.detach().cpu())

    stats = {
        "total": float(total.detach().cpu()),
        "categorical": float(categorical_loss.detach().cpu()),
        "numeric": float(numeric_loss.detach().cpu()),
        "kl_raw": float(kl_raw.detach().cpu()),
        "pairwise": pairwise_val,
        "move_vector": move_vector_val,
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
    free_bits: float = 0.0,
    pairwise_weight: float = 0.0,
    move_vector_weight: float = 0.0,
    hole_loss_weight: float = 1.0,
    autoregressive: bool = False,
    ss_ratio: float = 0.0,
    mask_rate_max: float | None = None,
    mask_rate_min: float = 0.0,
) -> EpochMetrics:
    """Run one epoch.

    mask_rate_max: when not None, sample mask_rate ~ Uniform[mask_rate_min, mask_rate_max)
    for each training batch.  None means use the decoder's cfg.mask_rate unchanged
    (val/test, or fixed-rate training).

    mask_rate_min: lower bound for the sampled training mask rate.  Default 0.0 (original
    behaviour).  Raising this (e.g. to 0.5) removes easy low-masking batches where the
    decoder can ignore z entirely and reconstruct from dense visible-hold context.
    Val/test always use cfg.mask_rate regardless of this value.
    """
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = cat_sum = num_sum = kl_raw_sum = pairwise_sum = move_vector_sum = 0.0
    batches = 0

    for batch_samples in iter_minibatches(samples, batch_size, shuffle=is_train):
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # Sample a fresh mask rate for this batch when dynamic masking is active.
        if mask_rate_max is not None:
            lo = min(mask_rate_min, mask_rate_max)  # guard against lo > hi edge case
            batch_mask_rate: float | None = random.uniform(lo, mask_rate_max)
        else:
            batch_mask_rate = None  # use decoder's cfg.mask_rate

        with torch.set_grad_enabled(is_train):
            loss, stats = _compute_batch_losses(
                model, batch_samples,
                device=device, eos_ids=eos_ids,
                kl_beta=kl_beta, numeric_weight=numeric_weight,
                sample_latent=sample_latent,
                free_bits=free_bits,
                pairwise_weight=pairwise_weight,
                move_vector_weight=move_vector_weight,
                hole_loss_weight=hole_loss_weight,
                autoregressive=autoregressive,
                ss_ratio=ss_ratio,
                train_mask_rate=batch_mask_rate,
            )
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        loss_sum += stats["total"]
        cat_sum += stats["categorical"]
        num_sum += stats["numeric"]
        kl_raw_sum += stats["kl_raw"]
        pairwise_sum += stats["pairwise"]
        move_vector_sum += stats["move_vector"]
        batches += 1

    if batches == 0:
        return EpochMetrics(0.0, 0.0, 0.0, 0.0, 0)
    return EpochMetrics(
        total_loss=loss_sum / batches,
        categorical_loss=cat_sum / batches,
        numeric_loss=num_sum / batches,
        kl_raw=kl_raw_sum / batches,
        num_batches=batches,
        pairwise_loss=pairwise_sum / batches,
        move_vector_loss=move_vector_sum / batches,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train route conditional VAE.")
    parser.add_argument("--db-path", type=str, default=str(PROJECT_ROOT / "data/raw/kilter_database.sqlite"))
    parser.add_argument("--metadata-source", type=str, default="kilter_board_csv")
    parser.add_argument("--metadata-product-id", type=int, default=1)
    parser.add_argument("--require-full-metadata", action="store_true")
    parser.add_argument("--max-routes", type=int, default=1000000)
    parser.add_argument("--min-holds", type=int, default=1)
    parser.add_argument("--min-quality-average", type=float, default=2.5,
                        help="Minimum quality_average (0–4 scale) to include a route. Default 2.5.")
    parser.add_argument("--min-ascensionist-count", type=int, default=15,
                        help="Minimum number of ascents to include a route. Default 15.")
    parser.add_argument("--cache-path", type=str, default=str(PROJECT_ROOT / "data/preprocessed_routes_cache.pt"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=256,
                        help="Batch size for validation and test. Larger than --batch-size is fine "
                             "because autoregressive decoding has no backprop memory pressure.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--numeric-weight", type=float, default=0.25)
    parser.add_argument("--kl-beta", type=float, default=0.1)
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
        "--mlp-encoder", action=argparse.BooleanOptionalAction, default=False,
        help="Replace the transformer encoder with a shallow MLP that maps shape_desc directly "
             "to route_embedding.  Baseline to measure reconstruction loss achievable from "
             "hand-crafted route statistics alone.  Reuses --encoder-d-model and "
             "--encoder-num-layers for MLP width/depth; all other encoder args are ignored.",
    )
    parser.add_argument(
        "--encoder-use-cls-token", action=argparse.BooleanOptionalAction, default=False,
        help="Prepend a CLS token (shape_desc → MLP → d_model) at position 0 of the encoder "
             "sequence.  All hold tokens attend to this global route summary at every layer. "
             "CLS is excluded from pooling.  shape_desc_dim is auto-detected from the cache.",
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
    # Decoder architecture
    parser.add_argument("--decoder-d-model", type=int, default=128,
                        help="Decoder transformer model dimension (kept smaller than encoder).")
    parser.add_argument("--decoder-num-layers", type=int, default=4,
                        help="Number of decoder transformer layers.")
    parser.add_argument("--decoder-dim-feedforward", type=int, default=512,
                        help="Decoder FFN hidden dim. Recommended: 4 × decoder-d-model.")
    parser.add_argument(
        "--parallel-decoder", action=argparse.BooleanOptionalAction, default=False,
        help="Use a non-autoregressive parallel MLP decoder instead of the transformer decoder. "
             "Each output position is predicted independently from z + a position embedding — "
             "no cross-token communication.  Forces z to encode all route structure (Phase 1). "
             "Ignores --decoder-token-dropout, --scheduled-sampling-ratio, and "
             "--decoder-dim-feedforward (not applicable to an MLP).  "
             "--decoder-d-model and --decoder-num-layers control the MLP width and depth.",
    )
    parser.add_argument("--decoder-token-dropout", type=float, default=0.0,
                        help="Probability of replacing each decoder input token (positions 1+) "
                             "with a learned mask embedding during training. Forces the decoder "
                             "to rely on z rather than exploiting categorical/spatial patterns "
                             "in the teacher-forced context. Disabled at inference. "
                             "Suggested range: 0.15–0.25.")
    parser.add_argument("--freeze-self-attn-epochs", type=int, default=0,
                        help="Freeze decoder self-attention for this many epochs at the start of "
                             "training. While frozen, cross-token communication is disabled so the "
                             "decoder must rely on z (cross-attention) rather than visible-hold "
                             "context, forcing the encoder to encode richer structure into z early. "
                             "Self-attention is unfrozen for the remaining epochs. Only meaningful "
                             "for the masked transformer decoder. Default: 0 (disabled).")
    parser.add_argument("--decoder-mask-rate", type=float, default=0.0,
                        help="Eval mask rate for the masked bidirectional decoder. When > 0, enables "
                             "masked decoder mode: bidirectional attention, loss at masked positions only. "
                             "At eval/test this exact rate is used (fixed). During training the rate is "
                             "sampled from Uniform[0, mask_max) each batch when --mask-warmup-epochs > 0, "
                             "otherwise this fixed rate is used for training too. Suggested: 0.5.")
    parser.add_argument("--mask-warmup-epochs", type=int, default=0,
                        help="Epochs over which the training mask-rate maximum ramps from 0 → 1. "
                             "Each training batch samples mask_rate ~ Uniform[mask_rate_min, epoch/warmup_epochs). "
                             "After warmup, samples from Uniform[mask_rate_min, 1). "
                             "Val/test always use the fixed --decoder-mask-rate. Default: 0 (fixed rate).")
    parser.add_argument("--mask-rate-min", type=float, default=0.0,
                        help="Minimum training mask rate when dynamic masking is active. "
                             "Training batches sample mask_rate ~ Uniform[mask_rate_min, mask_max). "
                             "Default 0.0 samples the full range [0, 1). "
                             "Raising this (e.g. 0.5) removes easy low-masking batches where the "
                             "decoder can ignore z and reconstruct from dense visible-hold context, "
                             "forcing z to contribute on every training step. "
                             "Val/test always use the fixed --decoder-mask-rate regardless.")
    parser.add_argument("--scheduled-sampling-ratio", type=float, default=0.0,
                        help="Target probability of replacing each decoder input token with the "
                             "model's own prediction from the previous position (scheduled sampling). "
                             "Unlike --decoder-token-dropout, the replacement is the model's actual "
                             "predicted token rather than a mask embedding, directly training the "
                             "model to recover from its own errors. Requires two forward passes per "
                             "training batch (first pass clean teacher-forced, second pass mixed). "
                             "Ramped from 0 to this target over --ss-warmup-epochs. "
                             "Can be combined with --decoder-token-dropout. Suggested: 0.25–0.5.")
    parser.add_argument("--ss-warmup-epochs", type=int, default=None,
                        help="Epochs to linearly ramp scheduled-sampling-ratio from 0 to its target. "
                             "Defaults to --kl-warmup-epochs if not set.")
    parser.add_argument("--kl-warmup-epochs", type=int, default=None,
                        help="Linear KL warmup epochs. Defaults to 40%% of total epochs. "
                             "Ignored when --kl-cycle-epochs > 0.")
    parser.add_argument("--kl-warmup-offset", type=int, default=0,
                        help="Epochs to hold kl_beta=0 before the linear warmup ramp begins. "
                             "Useful for letting the model converge as an autoencoder first. "
                             "Checkpoints are only saved after offset + warmup epochs. Default 0.")
    parser.add_argument("--kl-cycle-epochs", type=int, default=0,
                        help="Cyclic KL annealing: cycle length in epochs.  Each cycle linearly "
                             "ramps beta 0 → kl-beta over (kl-cycle-epochs × kl-cycle-ratio) "
                             "epochs, then holds at kl-beta for the rest.  Repeats throughout "
                             "training, giving repeated z-expansion phases where the decoder is "
                             "forced to learn z-dependent pathways before KL pressure compresses z. "
                             "When > 0, overrides --kl-warmup-epochs.  Default 0 (disabled).")
    parser.add_argument("--kl-cycle-ratio", type=float, default=0.5,
                        help="Fraction of each cycle spent ramping beta 0 → max.  "
                             "E.g. 0.5 with --kl-cycle-epochs 60 → 30 ramp + 30 plateau per cycle.  "
                             "Default 0.5.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default=str(PROJECT_ROOT / "data/route_cvae.pt"))
    parser.add_argument("--loss-plot-path", type=str, default=str(PROJECT_ROOT / "data/route_cvae_loss_curve.png"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder + bottleneck weights so only the decoder is trained. "
                             "When combined with --resume, loads encoder/bottleneck weights from the "
                             "checkpoint with strict=False (decoder keys missing from the checkpoint "
                             "are initialised randomly). Use to train a new decoder on a pretrained "
                             "frozen encoder. Implies --reset-best-val.")
    parser.add_argument("--reset-best-val", action="store_true",
                        help="When resuming, reset best_val to inf so the new training regime "
                             "can save checkpoints. Use when changing kl_beta/free_bits mid-run.")
    parser.add_argument("--early-stop-delta", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--free-bits", type=float, default=0.0,
                        help="Minimum KL per latent dimension (nats). Prevents posterior collapse by "
                             "clamping each dimension's KL from below before summing. "
                             "Recommended: 0.5 nats/dim.")
    parser.add_argument("--pairwise-weight", type=float, default=0.0,
                        help="Weight for pairwise distance loss (compounded with --numeric-weight). "
                             "Effective contribution = numeric_weight × pairwise_weight × MSE, so "
                             "pairwise_weight=1.0 gives one additional numeric-scale term. MSE is "
                             "between predicted and true L2 distances for all valid hold pairs, "
                             "normalised by √2. Creates gradient signal for inter-hold spatial "
                             "structure which per-token x/y MSE alone cannot provide. "
                             "Suggested starting value: 1.0–4.0.")
    parser.add_argument("--move-vector-weight", type=float, default=0.0,
                        help="Weight for move-vector loss (compounded with --numeric-weight). "
                             "Effective contribution = numeric_weight × move_vector_weight × MSE. "
                             "Computes MSE on directed Δx/Δy between consecutive holds in sequence "
                             "order, supervising step direction and size. Unlike pairwise distance "
                             "loss (rotation-invariant), this directly penalises wrong move "
                             "directionality and captures step_x_var and move_size. "
                             "Use instead of or alongside --pairwise-weight. "
                             "Suggested starting value: 2.0–4.0.")
    parser.add_argument("--hole-loss-weight", type=float, default=1.0,
                        help="Multiplier on the hole cross-entropy term in the reconstruction loss. "
                             "Hole vocab size is 593 (log₂(593)≈9.2 bits) vs type=7 and role=~5, so "
                             "hole CE dominates gradient signal by ~3× type+role combined and ~150× "
                             "x/y MSE. Reducing this (e.g. 0.1) shifts the gradient balance toward "
                             "spatial structure and away from set-membership encoding. "
                             "Default 1.0 preserves backward compatibility.")
    args = parser.parse_args()

    if args.kl_warmup_epochs is None:
        args.kl_warmup_epochs = max(1, int(math.ceil(0.40 * args.epochs)))
    if args.kl_warmup_epochs < 0:
        raise ValueError("--kl-warmup-epochs must be >= 0")
    if args.ss_warmup_epochs is None:
        args.ss_warmup_epochs = args.kl_warmup_epochs
    if args.ss_warmup_epochs < 0:
        raise ValueError("--ss-warmup-epochs must be >= 0")
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

    shape_desc_dim = 9
    if args.encoder_use_cls_token or args.mlp_encoder:
        shape_desc_dim = int(samples[0].tokens["shape_desc"].shape[0])
        print(f"shape_desc_dim={shape_desc_dim} (auto-detected from cache)")

    model, eos_ids = _build_model(
        vocabs, device,
        latent_dim=args.latent_dim,
        encoder_d_model=args.encoder_d_model,
        encoder_nhead=args.encoder_nhead,
        encoder_num_layers=args.encoder_num_layers,
        encoder_dim_feedforward=args.encoder_dim_feedforward,
        decoder_d_model=args.decoder_d_model,
        decoder_num_layers=args.decoder_num_layers,
        decoder_dim_feedforward=args.decoder_dim_feedforward,
        use_absolute_pos=args.use_absolute_pos,
        use_type_feature=args.use_type_feature,
        use_cls_token=args.encoder_use_cls_token,
        shape_desc_dim=shape_desc_dim,
        use_mlp_encoder=args.mlp_encoder,
        decoder_token_dropout=args.decoder_token_dropout,
        decoder_mask_rate=args.decoder_mask_rate,
        use_parallel_decoder=args.parallel_decoder,
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

    best_val = math.inf
    start_epoch = 1
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_path = Path(args.resume_path) if args.resume_path else checkpoint_path
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        # strict=False when freeze_encoder: encoder/bottleneck keys load normally;
        # decoder keys absent from the checkpoint (new architecture) are skipped and
        # left at their random init.  Unexpected keys (old decoder) are also ignored.
        missing, unexpected = model.load_state_dict(resume_ckpt["model_state_dict"], strict=not args.freeze_encoder)
        if args.freeze_encoder and (missing or unexpected):
            print(f"  Partial load: {len(missing)} missing decoder keys (random init), {len(unexpected)} unexpected (ignored)")
        if not args.freeze_encoder and "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        # freeze_encoder: fresh decoder training — epoch counter restarts from 1.
        # Regular resume: continue from where the checkpoint left off.
        start_epoch = 1 if args.freeze_encoder else int(resume_ckpt.get("epoch", 0)) + 1
        best_val = float(resume_ckpt.get("best_val", math.inf))
        if args.reset_best_val or args.freeze_encoder:
            best_val = math.inf
            print(f"Resumed from {resume_path} (next_epoch={start_epoch}, best_val reset to inf)")
        else:
            print(f"Resumed from {resume_path} (next_epoch={start_epoch}, best_val={best_val:.4f})")

    # Freeze encoder + bottleneck: only the decoder is trained.
    # Must come after resume so encoder weights are loaded before being frozen.
    # Rebuilds optimizer so frozen params never receive gradient updates or
    # accumulate Adam momentum — cleaner than keeping them in the optimizer.
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        for p in model.bottleneck.parameters():
            p.requires_grad_(False)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        n_frozen = sum(p.numel() for p in model.encoder.parameters()) + \
                   sum(p.numel() for p in model.bottleneck.parameters())
        n_trainable = sum(p.numel() for p in trainable_params)
        print(f"Encoder + bottleneck frozen ({n_frozen:,} params). Decoder only: {n_trainable:,} trainable params.")

    # When kl_warmup_epochs > 0 the best-of-warmup model is essentially a kl_beta≈0
    # autoencoder whose total loss will always beat any post-warmup model.  Track
    # warmup completion so we reset best_val at the boundary and only save checkpoints
    # from the post-warmup regime.  If resuming past the boundary, treat as complete
    # immediately so best_val from the checkpoint carries over normally.
    warmup_completed: bool = (
        (args.kl_warmup_epochs <= 0 and args.kl_warmup_offset <= 0)
        or start_epoch > args.kl_warmup_offset + args.kl_warmup_epochs
    )

    # Self-attention freeze: disabled by default (freeze_self_attn_epochs=0).
    # When enabled, self-attention weights start frozen so the decoder cannot
    # aggregate visible-hold context, forcing early z utilisation.  After the
    # freeze period the weights unfreeze and the decoder learns to complement z.
    # The optimizer already holds all parameters; frozen ones are simply skipped
    # (grad=None) until unfrozen, after which Adam accumulates moments normally.
    # NOTE: must come after resume so start_epoch reflects any loaded checkpoint.
    freeze_sa_epochs = args.freeze_self_attn_epochs
    if freeze_sa_epochs > 0:
        if start_epoch <= freeze_sa_epochs:
            n = _set_decoder_self_attn_frozen(model, frozen=True)
            if n > 0:
                print(f"Self-attention frozen ({n:,} params) for epochs 1–{freeze_sa_epochs}.")
            else:
                print("Warning: --freeze-self-attn-epochs set but decoder has no self-attention to freeze.")
        else:
            # Explicitly unfreeze in case the checkpoint was saved while SA was still frozen
            # (e.g. freeze_sa_epochs was lowered between runs).
            n = _set_decoder_self_attn_frozen(model, frozen=False)
            print(f"Resuming past freeze period (epoch {start_epoch} > {freeze_sa_epochs}); self-attention unfrozen ({n:,} params).")

    epoch_history: list[int] = []
    train_total_history: list[float] = []
    val_total_history: list[float] = []       # recon + kl_w + floor_pen (adversary excluded — see _compute_batch_losses)
    train_recon_history: list[float] = []
    val_recon_history: list[float] = []
    train_kl_history: list[float] = []
    val_kl_history: list[float] = []
    early_stop_stable_epochs = 0

    for epoch in range(start_epoch, args.epochs + 1):
        if freeze_sa_epochs > 0 and epoch == freeze_sa_epochs + 1:
            n = _set_decoder_self_attn_frozen(model, frozen=False)
            if n > 0:
                print(f"Epoch {epoch:03d}: self-attention unfrozen ({n:,} params).")

        kl_beta = _compute_kl_beta(
            epoch=epoch, target_kl_beta=args.kl_beta, kl_warmup_epochs=args.kl_warmup_epochs,
            kl_warmup_offset=args.kl_warmup_offset,
            kl_cycle_epochs=args.kl_cycle_epochs, kl_cycle_ratio=args.kl_cycle_ratio,
        )
        ss_ratio = _compute_ss_ratio(
            epoch=epoch,
            target_ss_ratio=args.scheduled_sampling_ratio,
            ss_warmup_epochs=args.ss_warmup_epochs,
        )
        # Dynamic masking: sample mask_rate ~ Uniform[0, mask_max) per training batch.
        # mask_max ramps from 0 → 1 over mask_warmup_epochs.  None = use cfg.mask_rate fixed.
        is_masked_decoder = args.decoder_mask_rate > 0.0
        train_mask_max = (
            _compute_mask_max(epoch=epoch, warmup_epochs=args.mask_warmup_epochs)
            if is_masked_decoder and args.mask_warmup_epochs > 0
            else None
        )
        train_m = _run_epoch(
            model, train_samples,
            optimizer=optimizer, device=device, eos_ids=eos_ids,
            batch_size=args.batch_size, kl_beta=kl_beta,
            numeric_weight=args.numeric_weight, grad_clip_norm=args.grad_clip_norm,
            sample_latent=True,
            free_bits=args.free_bits,
            pairwise_weight=args.pairwise_weight,
            move_vector_weight=args.move_vector_weight,
            hole_loss_weight=args.hole_loss_weight,
            ss_ratio=ss_ratio,
            mask_rate_max=train_mask_max,
            mask_rate_min=args.mask_rate_min,
        )
        with torch.no_grad():
            val_m = _run_epoch(
                model, val_samples,
                optimizer=None, device=device, eos_ids=eos_ids,
                batch_size=args.val_batch_size, kl_beta=kl_beta,
                numeric_weight=args.numeric_weight, grad_clip_norm=args.grad_clip_norm,
                sample_latent=False,
                free_bits=args.free_bits,
                pairwise_weight=args.pairwise_weight,
                move_vector_weight=args.move_vector_weight,
                hole_loss_weight=args.hole_loss_weight,
                autoregressive=False,  # teacher-forced: stable criterion for checkpoint selection
                # mask_rate_max=None: val always uses the fixed cfg.mask_rate for a consistent metric
            )

        train_recon = train_m.categorical_loss + args.numeric_weight * train_m.numeric_loss
        val_recon = val_m.categorical_loss + args.numeric_weight * val_m.numeric_loss
        kl_floor = args.free_bits * args.latent_dim

        # kl_w: KL contribution to total. Identity: total = recon + kl_w + pairwise_w
        # kl=X(+Y): X = raw unweighted KL, Y = excess above kl_floor (diagnostic).
        train_pairwise_w = args.numeric_weight * args.pairwise_weight * train_m.pairwise_loss
        val_pairwise_w   = args.numeric_weight * args.pairwise_weight * val_m.pairwise_loss
        train_kl_w = train_m.total_loss - train_recon - train_pairwise_w
        val_kl_w   = val_m.total_loss - val_recon - val_pairwise_w
        train_kl_excess = max(train_m.kl_raw - kl_floor, 0.0)
        val_kl_excess = max(val_m.kl_raw - kl_floor, 0.0)

        pw_str = (
            f" pw={train_m.pairwise_loss:.4f}/{val_m.pairwise_loss:.4f}"
            if args.pairwise_weight > 0.0 else ""
        )
        mv_str = (
            f" mv={train_m.move_vector_loss:.4f}/{val_m.move_vector_loss:.4f}"
            if args.move_vector_weight > 0.0 else ""
        )
        ss_str = f" ss={ss_ratio:.3f}" if args.scheduled_sampling_ratio > 0.0 else ""
        mask_str = (
            f" mask=[{args.mask_rate_min:.2f},{train_mask_max:.2f})"
            if train_mask_max is not None else ""
        )
        print(
            f"Epoch {epoch:03d} | kl_beta={kl_beta:.6f}{ss_str}{mask_str} "
            f"train total={train_m.total_loss:.4f} recon={train_recon:.4f} kl_w={train_kl_w:.4f} "
            f"cat={train_m.categorical_loss:.4f} num={train_m.numeric_loss:.4f} "
            f"kl={train_m.kl_raw:.4f}(+{train_kl_excess:.4f}){pw_str}{mv_str} | "
            f"val total={val_m.total_loss:.4f} recon={val_recon:.4f} kl_w={val_kl_w:.4f} "
            f"cat={val_m.categorical_loss:.4f} num={val_m.numeric_loss:.4f} "
            f"kl={val_m.kl_raw:.4f}(+{val_kl_excess:.4f})"
        )

        epoch_history.append(epoch)
        train_total_history.append(train_m.total_loss)
        val_total_history.append(val_m.total_loss)
        train_recon_history.append(train_recon)
        val_recon_history.append(val_recon)
        train_kl_history.append(train_kl_w)
        val_kl_history.append(val_kl_w)

        # Only save checkpoints after KL warmup completes.  Reset best_val at the boundary
        # so post-warmup epochs compete on their own terms (not against the near-zero-KL
        # autoencoder loss from early warmup epochs).
        is_post_warmup = (
            (args.kl_warmup_epochs <= 0 and args.kl_warmup_offset <= 0)
            or epoch > args.kl_warmup_offset + args.kl_warmup_epochs
        )
        if is_post_warmup and not warmup_completed:
            warmup_completed = True
            best_val = math.inf
            print(f"Epoch {epoch:03d}: KL warmup complete — tracking best post-warmup checkpoint.")

        # val_m.total_loss = recon + kl_w + floor_pen (adversary excluded at compute time).
        # This is the right checkpoint criterion: lower = better reconstruction and more active dims.
        if is_post_warmup and val_m.total_loss < best_val:
            best_val = val_m.total_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "args": vars(args),
                # Vocabs embedded so downstream scripts (routes_cluster.py) can
                # rebuild the exact training-time model architecture without needing
                # the preprocessed cache or local DB.  build_model_from_checkpoint
                # prefers these over runtime-supplied vocabs when present.
                "vocabs": vocabs,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint (val={best_val:.4f}): {checkpoint_path}")
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

    # Teacher-forced test: feeds ground-truth previous tokens to the AR decoder.
    # NOT directly comparable to the parallel decoder's 3.8045 (the parallel decoder is a
    # plain MLP(z) with no token context — it cannot exploit previous-token ground truth at all).
    # Teacher-forced AR will trivially beat 3.8045 because it has strictly more information.
    # Use this for debugging and sanity-checking the forward pass, not for baseline comparison.
    with torch.no_grad():
        test_m = _run_epoch(
            model, test_samples,
            optimizer=None, device=device, eos_ids=eos_ids,
            batch_size=args.val_batch_size, kl_beta=args.kl_beta,
            numeric_weight=args.numeric_weight, grad_clip_norm=args.grad_clip_norm,
            sample_latent=False,
            free_bits=args.free_bits,
            pairwise_weight=args.pairwise_weight,
            move_vector_weight=args.move_vector_weight,
            hole_loss_weight=args.hole_loss_weight,
            autoregressive=False,  # teacher-forced
        )
    test_recon = test_m.categorical_loss + args.numeric_weight * test_m.numeric_loss
    test_pairwise_w = args.numeric_weight * args.pairwise_weight * test_m.pairwise_loss
    test_kl_w = test_m.total_loss - test_recon - test_pairwise_w
    test_kl_excess = max(test_m.kl_raw - args.free_bits * args.latent_dim, 0.0)
    print(
        f"Test (teacher-forced) | total={test_m.total_loss:.4f} recon={test_recon:.4f} kl_w={test_kl_w:.4f} "
        f"cat={test_m.categorical_loss:.4f} num={test_m.numeric_loss:.4f} "
        f"kl={test_m.kl_raw:.4f}(+{test_kl_excess:.4f})"
    )

    # AR generation test: the decoder generates each token from its own previous predictions.
    # This is the fair comparison to the parallel decoder's 3.8045 — both start from z alone
    # with no ground-truth token context.  If the AR decoder has learned to generate coherently,
    # its AR generation recon should be below 3.8045 (z + causal self-context > z alone).
    # Skipped for the parallel decoder (is_parallel=True) and masked decoder.
    _is_parallel_dec = not getattr(model.decoder, "is_autoregressive", True)
    _is_masked_dec   = (not _is_parallel_dec) and getattr(model.decoder.cfg, "mask_rate", 0.0) > 0.0
    if not _is_parallel_dec and not _is_masked_dec:
        with torch.no_grad():
            ar_test_m = _run_epoch(
                model, test_samples,
                optimizer=None, device=device, eos_ids=eos_ids,
                batch_size=args.val_batch_size, kl_beta=args.kl_beta,
                numeric_weight=args.numeric_weight, grad_clip_norm=args.grad_clip_norm,
                sample_latent=False,
                free_bits=args.free_bits,
                pairwise_weight=args.pairwise_weight,
                hole_loss_weight=args.hole_loss_weight,
                autoregressive=True,  # true AR generation
            )
        ar_test_recon = ar_test_m.categorical_loss + args.numeric_weight * ar_test_m.numeric_loss
        ar_test_kl_excess = max(ar_test_m.kl_raw - args.free_bits * args.latent_dim, 0.0)
        print(
            f"Test (AR generation)  | recon={ar_test_recon:.4f} "
            f"cat={ar_test_m.categorical_loss:.4f} num={ar_test_m.numeric_loss:.4f} "
            f"kl={ar_test_m.kl_raw:.4f}(+{ar_test_kl_excess:.4f})"
        )


if __name__ == "__main__":
    main()
