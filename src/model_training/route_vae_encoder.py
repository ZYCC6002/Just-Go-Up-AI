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
        # Without this, pad tokens carry real-looking embeddings (type=0 → jug,
        # x=0 → near board origin) and flow through all layers as fake holds.
        tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)  # [B, L, d_model]
        encoded = self.final_norm(encoded)

        # Route embedding: mean+max pool over non-padded hold tokens.
        valid = (~padding_mask).unsqueeze(-1).to(encoded.dtype)            # [B, L, 1]
        mean_pooled = (encoded * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        max_pooled = encoded.masked_fill(
            padding_mask.unsqueeze(-1), float("-inf")
        ).max(dim=1).values
        route_embedding = self.pool_proj(torch.cat([mean_pooled, max_pooled], dim=-1))

        return {"token_embeddings": encoded, "route_embedding": route_embedding}


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
