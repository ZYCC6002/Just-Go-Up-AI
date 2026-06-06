from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .model_utils import (
    HoldTokenEmbedder,
    ScalarSinusoidalEmbedding,
)


@dataclass
class RouteVAEEncoderConfig:
    """Configuration for hold-token transformer encoder."""

    # Feature vocab sizes (set based on preprocessing vocabularies)
    type_vocab_size: int
    role_vocab_size: int
    hole_id_vocab_size: int

    # Per-feature embedding sizes
    type_embed_dim: int = 32          # most style-discriminating feature
    role_embed_dim: int = 8
    hole_id_embed_dim: int = 16
    x_embed_dim: int = 16
    y_embed_dim: int = 16
    depth_embed_dim: int = 8
    orientation_sin_embed_dim: int = 8
    orientation_cos_embed_dim: int = 8
    size_embed_dim: int = 8

    # Ablation flags — set False to remove the feature from the token embedding.
    # These must match whatever the preprocessed cache was built with.
    use_absolute_pos: bool = True   # if False, x/y sinusoidal embeddings are dropped
    use_type_feature: bool = True   # if False, hold-type embedding is dropped

    # CLS token: prepend a route-level summary token (shape_desc → MLP → d_model) at
    # position 0.  All hold tokens attend to/from this token so every layer has access
    # to the global route descriptor.  CLS is excluded from pooling; only hold-token
    # outputs contribute to the route_embedding / z.
    # Requires "shape_desc" in the collated batch (always present when built from cache).
    use_cls_token: bool = False
    shape_desc_dim: int = 9        # shape_desc width; ignored when use_cls_token=False

    # Transformer sizes
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Input normalization ranges — derived from actual hole positions for
    # product_id=1 (Kilter Board Original):
    #   SELECT MIN(x), MAX(x), MIN(y), MAX(y) FROM holes WHERE product_id=1
    #   → x ∈ [-20, 164], y ∈ [4, 176]
    x_min: float = -20.0
    x_max: float = 164.0
    y_min: float = 4.0
    y_max: float = 176.0


class RouteTransformerEncoder(nn.Module):
    """Transformer encoder over hold tokens with per-feature embeddings.

    Expected token-level inputs (each [B, L], except padding_mask):
    - type_encoded_id, role_encoded_id, hole_encoded_id
    - x, y, depth, orientation_sin, orientation_cos, size
    - padding_mask: bool [B, L], True for padded tokens
    """

    def __init__(self, cfg: RouteVAEEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.hold_embedder = HoldTokenEmbedder(cfg, use_depth=True)

        # Disable nested-tensor path for MPS compatibility.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            layer_norm_eps=cfg.layer_norm_eps,
            batch_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=cfg.num_layers,
                enable_nested_tensor=False,
                mask_check=False,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=cfg.num_layers,
                enable_nested_tensor=False,
            )
        self.final_norm = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

        # Mean+max pooling: concat mean and max over hold tokens → Linear(2*d_model, d_model).
        self.pool_proj = nn.Linear(2 * cfg.d_model, cfg.d_model)

        if cfg.use_cls_token:
            self.shape_token_mlp: nn.Sequential | None = nn.Sequential(
                nn.Linear(cfg.shape_desc_dim, cfg.d_model),
                nn.GELU(),
                nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps),
            )
        else:
            self.shape_token_mlp = None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode hold-token sequences.

        Returns:
            {
              "token_embeddings": [B, L, d_model],
              "route_embedding":  [B, d_model],
            }
        """
        padding_mask = batch["padding_mask"].bool()   # [B, L]
        tokens = self.hold_embedder.embed(batch)       # [B, L, d_model]
        # Zero out padding positions before they enter the transformer.
        tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if self.shape_token_mlp is not None:
            shape_token = self.shape_token_mlp(
                batch["shape_desc"].to(torch.float32)
            ).unsqueeze(1)                                                  # [B, 1, d_model]
            tokens = torch.cat([shape_token, tokens], dim=1)               # [B, L+1, d_model]
            cls_pad = torch.zeros(
                tokens.shape[0], 1, dtype=torch.bool, device=padding_mask.device
            )
            attn_mask = torch.cat([cls_pad, padding_mask], dim=1)          # [B, L+1]
        else:
            attn_mask = padding_mask

        encoded = self.encoder(tokens, src_key_padding_mask=attn_mask)     # [B, L(+1), d_model]
        encoded = self.final_norm(encoded)

        # Route embedding: mean+max pool over hold tokens only (exclude CLS at position 0).
        hold_enc = encoded[:, 1:, :] if self.shape_token_mlp is not None else encoded
        valid = (~padding_mask).unsqueeze(-1).to(hold_enc.dtype)           # [B, L, 1]
        mean_pooled = (hold_enc * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        max_pooled = hold_enc.masked_fill(
            padding_mask.unsqueeze(-1), float("-inf")
        ).max(dim=1).values
        route_embedding = self.pool_proj(torch.cat([mean_pooled, max_pooled], dim=-1))

        return {"token_embeddings": hold_enc, "route_embedding": route_embedding}


def collate_hold_token_batch(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad variable-length hold-token samples into a batch."""
    if not samples:
        raise ValueError("samples must be non-empty")

    CATEGORICAL_KEYS = {
        "type_encoded_id", "role_encoded_id", "hole_encoded_id",
    }
    FEATURE_KEYS = [
        "type_encoded_id", "role_encoded_id", "hole_encoded_id",
        "x", "y", "depth", "orientation_sin", "orientation_cos", "size",
    ]

    lengths = [int(s["type_encoded_id"].shape[0]) for s in samples]
    max_len = max(lengths)
    batch_size = len(samples)

    def _pad_1d(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        if tensor.shape[0] == target_len:
            return tensor
        pad = torch.zeros(target_len - tensor.shape[0], dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=0)

    batch: dict[str, torch.Tensor] = {}
    for key in FEATURE_KEYS:
        batch[key] = torch.stack([_pad_1d(s[key], max_len) for s in samples], dim=0)

    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_mask[i, :length] = False
    batch["padding_mask"] = padding_mask

    # shape_desc is per-sample (not per-hold) — stack without padding.
    if "shape_desc" in samples[0]:
        batch["shape_desc"] = torch.stack([s["shape_desc"] for s in samples], dim=0)

    # Cast types
    for key in CATEGORICAL_KEYS:
        if key in batch:
            batch[key] = batch[key].long()
    for key in ["x", "y", "depth", "orientation_sin", "orientation_cos", "size"]:
        if key in batch:
            batch[key] = batch[key].to(torch.float32)

    return batch


__all__ = [
    "RouteVAEEncoderConfig",
    "RouteTransformerEncoder",
    "ScalarSinusoidalEmbedding",
    "collate_hold_token_batch",
]
