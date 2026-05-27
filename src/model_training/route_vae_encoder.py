from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .model_utils import (
    ConditionAdaLayerNorm,
    HoldTokenEmbedder,
    ScalarSinusoidalEmbedding,
    normalize_minmax,
)


@dataclass
class RouteVAEEncoderConfig:
    """Configuration for hold-token transformer encoder."""

    # Feature vocab sizes (set based on preprocessing vocabularies)
    type_vocab_size: int
    function_vocab_size: int
    role_vocab_size: int
    hole_id_vocab_size: int

    # Per-feature embedding sizes
    type_embed_dim: int = 32          # increased from 16: most style-discriminating feature
    function_embed_dim: int = 8
    role_embed_dim: int = 8
    hole_id_embed_dim: int = 16
    x_embed_dim: int = 16
    y_embed_dim: int = 16
    depth_embed_dim: int = 8
    orientation_sin_embed_dim: int = 8
    orientation_cos_embed_dim: int = 8
    size_embed_dim: int = 8

    # Grip category embedding (per-hold style abstraction)
    grip_category_vocab_size: int = 0    # 0 = disabled (backward compat with old vocabs)
    grip_category_embed_dim: int = 8

    # Move delta embeddings (per-hold, encoder + decoder)
    delta_embed_dim: int = 8             # used for delta_x_prev, delta_y_prev, dist_to_nearest

    # Nearest-neighbour distance feature (encoder-only; requires full sequence)
    use_dist_to_nearest: bool = True

    # Transformer sizes
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    condition_hidden_dim: int = 64
    use_condition: bool = True
    use_cond_adaln: bool = True

    # Input normalization ranges
    x_min: float = 0.0
    x_max: float = 140.0
    y_min: float = 0.0
    y_max: float = 160.0
    angle_min: float = 0.0
    angle_max: float = 70.0
    grade_min: float = 10.0
    grade_max: float = 33.0


class RouteTransformerEncoder(nn.Module):
    """Transformer encoder over hold tokens with per-feature embeddings.

    Expected token-level inputs (each [B, L], except padding_mask):
    - type_encoded_id, function_encoded_id, role_encoded_id, hole_encoded_id
    - x, y, depth, orientation_sin, orientation_cos, size
    - delta_x_prev, delta_y_prev, dist_to_nearest  (new move-size features)
    - grip_category_id                              (new grip-category feature)
    - padding_mask: bool [B, L], True for padded tokens

    Expected route-level inputs:
    - shape_desc: float [B, 9]  (route shape descriptor, prepended as CLS token)
    """

    def __init__(self, cfg: RouteVAEEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.hold_embedder = HoldTokenEmbedder(cfg)

        # Shape descriptor MLP: projects [9] route-level stats → d_model CLS token
        self.shape_token_mlp = nn.Sequential(
            nn.Linear(9, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps),
        )

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

        # Condition embedding: [angle, grade, grade_missing] -> d_model
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cfg.condition_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.condition_hidden_dim, cfg.d_model),
        )
        if cfg.use_cond_adaln:
            self.pre_encoder_adaln = ConditionAdaLayerNorm(cfg.d_model, cfg.d_model, cfg.layer_norm_eps)
            self.post_encoder_adaln = ConditionAdaLayerNorm(cfg.d_model, cfg.d_model, cfg.layer_norm_eps)
        else:
            self.pre_encoder_adaln = None
            self.post_encoder_adaln = None

    def _build_condition_embedding(
        self,
        *,
        angle: torch.Tensor,
        grade: torch.Tensor,
        grade_missing: torch.Tensor | None,
    ) -> torch.Tensor:
        if grade_missing is None:
            grade_missing = torch.zeros_like(grade, dtype=torch.float32)
        else:
            grade_missing = grade_missing.to(torch.float32)

        cond_input = torch.stack(
            [
                normalize_minmax(angle, self.cfg.angle_min, self.cfg.angle_max),
                normalize_minmax(grade, self.cfg.grade_min, self.cfg.grade_max),
                grade_missing,
            ],
            dim=-1,
        )
        return self.cond_mlp(cond_input)

    @staticmethod
    def masked_mean_pool(token_embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Mean pool over non-padded tokens.

        Args:
            token_embeddings: [B, L, D]
            padding_mask: [B, L] where True means padded token
        """
        valid_mask = (~padding_mask).unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * valid_mask).sum(dim=1)
        counts = valid_mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        angle: torch.Tensor,
        grade: torch.Tensor,
        grade_missing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode hold-token sequences.

        Returns:
            {
              "token_embeddings": [B, L, d_model],   # per-hold (excludes shape token)
              "route_embedding":  [B, d_model],       # shape-token output (position 0)
            }
        """
        padding_mask = batch["padding_mask"].bool()   # [B, L]
        tokens = self.hold_embedder.embed(batch)       # [B, L, d_model]

        # Build and prepend shape token (route-level CLS)
        shape_desc = batch["shape_desc"].to(tokens.dtype)           # [B, 9]
        shape_token = self.shape_token_mlp(shape_desc).unsqueeze(1) # [B, 1, d_model]
        tokens = torch.cat([shape_token, tokens], dim=1)            # [B, L+1, d_model]

        # Extend padding mask: shape token is always non-padded
        batch_size = tokens.shape[0]
        cls_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device)
        padding_mask = torch.cat([cls_pad, padding_mask], dim=1)    # [B, L+1]

        cond_emb: torch.Tensor | None = None
        if self.cfg.use_condition:
            cond_emb = self._build_condition_embedding(
                angle=angle, grade=grade, grade_missing=grade_missing
            )
            if self.pre_encoder_adaln is not None:
                tokens = self.pre_encoder_adaln(tokens, cond_emb)
            else:
                tokens = tokens + cond_emb.unsqueeze(1)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)  # [B, L+1, d_model]

        if self.post_encoder_adaln is not None and cond_emb is not None:
            encoded = self.post_encoder_adaln(encoded, cond_emb)
        encoded = self.final_norm(encoded)

        # Route embedding = shape/CLS token at position 0
        route_embedding = encoded[:, 0, :]      # [B, d_model]
        # Token embeddings = hold tokens only (exclude shape token)
        token_embeddings = encoded[:, 1:, :]    # [B, L, d_model]

        return {"token_embeddings": token_embeddings, "route_embedding": route_embedding}


def collate_hold_token_batch(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad variable-length hold-token samples into a batch.

    Each sample is a dict with 1-D tensors of length L_i for most keys, plus
    a special sample-level key `shape_desc` with shape [9] that is stacked
    (not padded).
    """
    if not samples:
        raise ValueError("samples must be non-empty")

    CATEGORICAL_KEYS = {
        "type_encoded_id", "function_encoded_id", "role_encoded_id",
        "hole_encoded_id", "grip_category_id",
    }
    # Core per-hold keys always present
    FEATURE_KEYS = [
        "type_encoded_id", "function_encoded_id", "role_encoded_id", "hole_encoded_id",
        "x", "y", "depth", "orientation_sin", "orientation_cos", "size",
    ]
    # Optional per-hold features — include only if present in this batch's samples
    for optional_key in ("delta_x_prev", "delta_y_prev", "dist_to_nearest", "grip_category_id"):
        if optional_key in samples[0]:
            FEATURE_KEYS = FEATURE_KEYS + [optional_key]

    # shape_desc is a per-sample tensor [9], not per-hold — stacked separately
    SAMPLE_LEVEL_KEYS = {"shape_desc"}

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
    for key in ["x", "y", "depth", "orientation_sin", "orientation_cos", "size",
                "delta_x_prev", "delta_y_prev", "dist_to_nearest"]:
        if key in batch:
            batch[key] = batch[key].to(torch.float32)

    # Stack sample-level tensors (not per-hold, so no padding needed)
    for key in SAMPLE_LEVEL_KEYS:
        if key in samples[0]:
            batch[key] = torch.stack([s[key] for s in samples], dim=0)

    return batch


__all__ = [
    "RouteVAEEncoderConfig",
    "RouteTransformerEncoder",
    "ScalarSinusoidalEmbedding",
    "collate_hold_token_batch",
]
