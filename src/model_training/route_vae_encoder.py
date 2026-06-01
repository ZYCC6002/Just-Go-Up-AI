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
    role_vocab_size: int
    hole_id_vocab_size: int

    # Per-feature embedding sizes
    type_embed_dim: int = 32          # increased from 16: most style-discriminating feature
    role_embed_dim: int = 8
    hole_id_embed_dim: int = 16
    x_embed_dim: int = 16
    y_embed_dim: int = 16
    depth_embed_dim: int = 8
    orientation_sin_embed_dim: int = 8
    orientation_cos_embed_dim: int = 8
    size_embed_dim: int = 8

    # Move delta embeddings (per-hold, encoder + decoder)
    delta_embed_dim: int = 8             # used for delta_x_prev, delta_y_prev

    # k-NN neighbourhood features (encoder-only; requires full sequence)
    use_knn_features: bool = False
    knn_embed_dim: int = 16

    # Ablation flags — set False to remove the feature from the token embedding.
    # These must match whatever the preprocessed cache was built with.
    use_absolute_pos: bool = True   # if False, x/y sinusoidal embeddings are dropped
    use_type_feature: bool = True   # if False, hold-type embedding is dropped
    use_delta_features: bool = False  # if False, Δx/Δy move-delta embeddings are dropped

    # Route-level shape descriptor dimension — auto-detected from data at training time
    # so it stays consistent with whatever route_preprocessing.py produced.
    shape_desc_dim: int = 9

    # Pooling strategy for the route embedding fed to the bottleneck.
    # "mean_max"  — concatenate mean-pooled and max-pooled hold token outputs [B, 2*d_model]
    #               then project to d_model.  Mean captures aggregate route statistics;
    #               max surfaces the most extreme hold per style dimension (largest move,
    #               hardest hold type, etc.).  Theoretically motivated by DeepSets:
    #               {mean, max} can approximate any continuous set function.  Default.
    # "attention" — K learned queries attend over hold token outputs via multi-head attention.
    #               Kept for backward compat with existing checkpoints; in practice converges
    #               to near-uniform weights (effectively mean pooling) due to the VAE
    #               reconstruction loss not rewarding query specialisation.
    # "cls"       — shape/CLS token at position 0; kept for backward compat only.
    route_pool_mode: str = "mean_max"
    # Number of learned query vectors (attention mode only).
    pool_num_queries: int = 4
    # Heads in the pooling MHA (attention mode only).
    pool_nhead: int = 4

    # Transformer sizes
    d_model: int = 256            # increased from 128: 32 dims/head, richer per-hold representations
    nhead: int = 8                # 8 heads × 32 dims = d_model; heads can specialise (spatial, type, sequential)
    num_layers: int = 6           # increased from 4: allows hold→pair→regional→route hierarchy
    dim_feedforward: int = 1024   # 4× d_model (standard ratio); was 256 = 2×, which bottlenecked feature synthesis
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    condition_hidden_dim: int = 128  # scaled with d_model (was 64)
    use_cond_adaln: bool = True

    # Input normalization ranges — derived from actual hole positions for
    # product_id=1 (Kilter Board Original):
    #   SELECT MIN(x), MAX(x), MIN(y), MAX(y) FROM holes WHERE product_id=1
    #   → x ∈ [-20, 164], y ∈ [4, 176]
    x_min: float = -20.0
    x_max: float = 164.0
    y_min: float = 4.0
    y_max: float = 176.0
    angle_min: float = 0.0
    angle_max: float = 70.0
    grade_min: float = 10.0
    grade_max: float = 33.0


class RouteTransformerEncoder(nn.Module):
    """Transformer encoder over hold tokens with per-feature embeddings.

    Expected token-level inputs (each [B, L], except padding_mask):
    - type_encoded_id, role_encoded_id, hole_encoded_id
    - x, y, depth, orientation_sin, orientation_cos, size
    - delta_x_prev, delta_y_prev, dist_to_nearest  (new move-size features)
    - padding_mask: bool [B, L], True for padded tokens

    Expected route-level inputs:
    - shape_desc: float [B, 9]  (route shape descriptor, prepended as CLS token)
    """

    def __init__(self, cfg: RouteVAEEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.hold_embedder = HoldTokenEmbedder(cfg, use_delta=getattr(cfg, "use_delta_features", True))

        # Shape descriptor MLP: projects shape_desc → d_model CLS token.
        # Dimension is cfg.shape_desc_dim (9 full / 5 shape-only experiment).
        self.shape_token_mlp = nn.Sequential(
            nn.Linear(cfg.shape_desc_dim, cfg.d_model),
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

        # Condition embedding: [angle, grade] -> d_model
        self.cond_mlp = nn.Sequential(
            nn.Linear(2, cfg.condition_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.condition_hidden_dim, cfg.d_model),
        )
        if cfg.use_cond_adaln:
            self.pre_encoder_adaln = ConditionAdaLayerNorm(cfg.d_model, cfg.d_model, cfg.layer_norm_eps)
        else:
            self.pre_encoder_adaln = None

        # Route embedding pooling — see route_pool_mode docstring above.
        # "mean_max": simple concat of mean + max over hold tokens, projected to d_model.
        #             No additional learned parameters beyond a single Linear.
        # "attention": K learned queries (backward compat with existing checkpoints).
        # "cls": shape token fallback (backward compat only).
        pool_mode = getattr(cfg, "route_pool_mode", "attention")
        self._pool_mode = pool_mode
        if pool_mode == "mean_max":
            self.pool_queries: nn.Parameter | None = None  # type: ignore[assignment]
            self.pool_attn: nn.MultiheadAttention | None = None
            self.pool_proj: nn.Linear | None = nn.Linear(2 * cfg.d_model, cfg.d_model)
        elif pool_mode == "attention":
            K = cfg.pool_num_queries
            self.pool_queries = nn.Parameter(torch.randn(K, cfg.d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=cfg.d_model,
                num_heads=cfg.pool_nhead,
                dropout=0.0,
                batch_first=True,
            )
            # Project K concatenated summaries back to d_model
            self.pool_proj = nn.Linear(K * cfg.d_model, cfg.d_model)
        else:
            # "cls" fallback
            self.pool_queries = None
            self.pool_attn = None
            self.pool_proj = None

    def _build_condition_embedding(
        self,
        *,
        angle: torch.Tensor,
        grade: torch.Tensor,
    ) -> torch.Tensor:
        cond_input = torch.stack(
            [
                normalize_minmax(angle, self.cfg.angle_min, self.cfg.angle_max),
                normalize_minmax(grade, self.cfg.grade_min, self.cfg.grade_max),
            ],
            dim=-1,
        )
        return self.cond_mlp(cond_input)

    @staticmethod
    def masked_mean_pool(token_embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Mean pool over non-padded tokens (utility; not used by default pooling path).

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

        # Build and prepend shape token (route-level CLS).
        # Truncate to cfg.shape_desc_dim so old checkpoints (shape_desc_dim=5) work
        # correctly with caches that store the full 9D shape_desc.  The first 5
        # features are identical between the 5D and 9D layouts (backward-compat).
        shape_desc = batch["shape_desc"].to(tokens.dtype)[:, :self.cfg.shape_desc_dim]  # [B, D_sd]
        shape_token = self.shape_token_mlp(shape_desc).unsqueeze(1)                      # [B, 1, d_model]
        tokens = torch.cat([shape_token, tokens], dim=1)            # [B, L+1, d_model]

        # Extend padding mask: shape token is always non-padded
        batch_size = tokens.shape[0]
        cls_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device)
        padding_mask = torch.cat([cls_pad, padding_mask], dim=1)    # [B, L+1]

        cond_emb: torch.Tensor | None = None
        if self.pre_encoder_adaln is not None:
            cond_emb = self._build_condition_embedding(angle=angle, grade=grade)
            tokens = self.pre_encoder_adaln(tokens, cond_emb)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)  # [B, L+1, d_model]
        encoded = self.final_norm(encoded)

        # Token embeddings = hold tokens only (exclude shape token at position 0)
        token_embeddings = encoded[:, 1:, :]    # [B, L, d_model]

        # Route embedding: pool over hold tokens (shape/CLS token excluded to prevent
        # shortcutting via pre-computed shape_desc statistics).
        hold_padding = padding_mask[:, 1:]  # [B, L] — padding mask for hold tokens only
        if self._pool_mode == "mean_max":
            # Mean pool over non-padded tokens
            valid = (~hold_padding).unsqueeze(-1).to(token_embeddings.dtype)  # [B, L, 1]
            mean_pooled = (token_embeddings * valid).sum(1) / valid.sum(1).clamp(min=1.0)  # [B, D]
            # Max pool — mask padded positions to -inf before reduction
            max_pooled = token_embeddings.masked_fill(
                hold_padding.unsqueeze(-1), float("-inf")
            ).max(dim=1).values                                               # [B, D]
            route_embedding = self.pool_proj(
                torch.cat([mean_pooled, max_pooled], dim=-1)
            )                                                                 # [B, d_model]
        elif self._pool_mode == "attention" and self.pool_attn is not None:
            q = self.pool_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, d_model]
            pooled, _ = self.pool_attn(
                q, token_embeddings, token_embeddings,
                key_padding_mask=hold_padding,
            )                                                               # [B, K, d_model]
            route_embedding = self.pool_proj(pooled.flatten(1))            # [B, d_model]
        else:
            # "cls" fallback for backward-compat with pre-pool checkpoints
            route_embedding = encoded[:, 0, :]  # [B, d_model]

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
        "type_encoded_id", "role_encoded_id", "hole_encoded_id",
    }
    # Core per-hold keys always present
    FEATURE_KEYS = [
        "type_encoded_id", "role_encoded_id", "hole_encoded_id",
        "x", "y", "depth", "orientation_sin", "orientation_cos", "size",
    ]
    # Optional 1-D per-hold features — include only if present in this batch's samples
    for optional_key in ("delta_x_prev", "delta_y_prev"):
        if optional_key in samples[0]:
            FEATURE_KEYS = FEATURE_KEYS + [optional_key]

    # knn_features is [L, 9] per sample — 2D, handled separately below
    has_knn = "knn_features" in samples[0]

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

    def _pad_2d(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad a [L, D] tensor to [target_len, D] by appending zero rows."""
        if tensor.shape[0] == target_len:
            return tensor
        pad = torch.zeros(target_len - tensor.shape[0], tensor.shape[1], dtype=tensor.dtype, device=tensor.device)
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
                "delta_x_prev", "delta_y_prev"]:
        if key in batch:
            batch[key] = batch[key].to(torch.float32)

    # knn_features: [L, 9] per sample → [B, max_len, 9]
    if has_knn:
        batch["knn_features"] = torch.stack(
            [_pad_2d(s["knn_features"], max_len) for s in samples], dim=0
        ).to(torch.float32)

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
