from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal embedding for scalar coordinates
# ---------------------------------------------------------------------------

class ScalarSinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for scalar coordinates (e.g. x/y board coordinates)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("Sinusoidal embedding dimension must be even.")
        self.embed_dim = embed_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            values: Tensor of shape [B, L] or [N] with scalar coordinates in [0, 1].

        Returns:
            Tensor of shape [B, L, embed_dim] (or [N, embed_dim] for 1D input).
        """
        device = values.device
        dtype = values.dtype
        if values.dim() not in (1, 2):
            raise ValueError("values must have shape [B, L] or [N].")

        values_f = values.to(torch.float32)
        half_dim = self.embed_dim // 2
        exponents = torch.arange(half_dim, device=device, dtype=torch.float32)
        denom = torch.pow(10000.0, 2 * exponents / self.embed_dim)

        angles = values_f.unsqueeze(-1) / denom
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        emb = torch.stack((sin, cos), dim=-1).reshape(*sin.shape[:-1], self.embed_dim)
        return emb.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Adaptive LayerNorm (shared by encoder and decoder)
# ---------------------------------------------------------------------------

class ConditionAdaLayerNorm(nn.Module):
    """Adaptive LayerNorm modulated by a route-level condition embedding."""

    def __init__(self, d_model: int, cond_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.Linear(cond_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    hidden states [B, L, d_model]
            cond: condition embedding [B, cond_dim]
        """
        gamma_beta = self.modulation(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        x_norm = self.norm(x)
        return x_norm * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


# ---------------------------------------------------------------------------
# Normalization helpers (module-level, used by both encoder and decoder)
# ---------------------------------------------------------------------------

def normalize_minmax(values: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    denom = max(vmax - vmin, 1e-6)
    return ((values.to(torch.float32) - vmin) / denom).clamp(0.0, 1.0)


def normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    # depth range: [0, 3]
    return (depth.to(torch.float32) / 3.0).clamp(0.0, 1.0)


def normalize_size(size: torch.Tensor) -> torch.Tensor:
    # size range: [2, 5]
    return ((size.to(torch.float32) - 2.0) / 3.0).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Hold token embedder (shared between encoder and decoder)
# ---------------------------------------------------------------------------

class HoldTokenEmbedder(nn.Module):
    """Embeds all per-hold features into a d_model token.

    Both encoder and decoder use per-hold feature embeddings.
    Accepts any config object that provides the expected attribute names.

    New features vs. original:
    - type_embed_dim raised to 32 (most style-discriminating feature)
    - grip_category_id: per-hold grip-feel category (jug/sloper/crimp/pinch/foot_type/unknown)
    - delta_x_prev, delta_y_prev: move vector from the previous hold (y-sorted sequence)
    - dist_to_nearest: distance to the nearest other hold (encoder-only; set
      use_dist_to_nearest=False on the decoder config to skip this feature)
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.x_min: float = cfg.x_min
        self.x_max: float = cfg.x_max
        self.y_min: float = cfg.y_min
        self.y_max: float = cfg.y_max

        # Categorical embeddings
        self.type_embedding = nn.Embedding(cfg.type_vocab_size, cfg.type_embed_dim)
        self.function_embedding = nn.Embedding(cfg.function_vocab_size, cfg.function_embed_dim)
        self.role_embedding = nn.Embedding(cfg.role_vocab_size, cfg.role_embed_dim)
        self.hole_embedding = nn.Embedding(cfg.hole_id_vocab_size, cfg.hole_id_embed_dim)

        # Optional grip category (vocab_size=0 disables it for backward compatibility)
        grip_cat_size = getattr(cfg, "grip_category_vocab_size", 0)
        grip_cat_dim = getattr(cfg, "grip_category_embed_dim", 8)
        if grip_cat_size > 0:
            self.grip_category_embedding: nn.Embedding | None = nn.Embedding(grip_cat_size, grip_cat_dim)
            self._grip_cat_dim = grip_cat_dim
        else:
            self.grip_category_embedding = None
            self._grip_cat_dim = 0

        # Sinusoidal positional embeddings for x/y
        self.x_embedding = ScalarSinusoidalEmbedding(cfg.x_embed_dim)
        self.y_embedding = ScalarSinusoidalEmbedding(cfg.y_embed_dim)

        # Numeric scalar projections
        self.depth_projection = nn.Linear(1, cfg.depth_embed_dim)
        self.orientation_sin_projection = nn.Linear(1, cfg.orientation_sin_embed_dim)
        self.orientation_cos_projection = nn.Linear(1, cfg.orientation_cos_embed_dim)
        self.size_projection = nn.Linear(1, cfg.size_embed_dim)

        # Move-delta projections (delta from previous hold in y-sorted sequence)
        delta_dim = getattr(cfg, "delta_embed_dim", 8)
        self.delta_x_projection = nn.Linear(1, delta_dim)
        self.delta_y_projection = nn.Linear(1, delta_dim)
        self._delta_dim = delta_dim

        # Nearest-neighbour distance (encoder-only; requires full hold set)
        use_dist = getattr(cfg, "use_dist_to_nearest", True)
        if use_dist:
            self.dist_projection: nn.Linear | None = nn.Linear(1, delta_dim)
            self._dist_dim = delta_dim
        else:
            self.dist_projection = None
            self._dist_dim = 0

        token_input_dim = (
            cfg.type_embed_dim
            + cfg.function_embed_dim
            + cfg.role_embed_dim
            + cfg.hole_id_embed_dim
            + cfg.x_embed_dim
            + cfg.y_embed_dim
            + cfg.depth_embed_dim
            + cfg.orientation_sin_embed_dim
            + cfg.orientation_cos_embed_dim
            + cfg.size_embed_dim
            + self._grip_cat_dim    # 0 if disabled
            + self._delta_dim * 2   # delta_x + delta_y
            + self._dist_dim        # 0 if disabled
        )
        self.token_projection = nn.Sequential(
            nn.Linear(token_input_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps),
            nn.Dropout(cfg.dropout),
        )

    def embed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed a batch of hold tokens to [B, L, d_model]."""
        type_emb = self.type_embedding(batch["type_encoded_id"])
        function_emb = self.function_embedding(batch["function_encoded_id"])
        role_emb = self.role_embedding(batch["role_encoded_id"])
        hole_emb = self.hole_embedding(batch["hole_encoded_id"])

        x_emb = self.x_embedding(normalize_minmax(batch["x"], self.x_min, self.x_max))
        y_emb = self.y_embedding(normalize_minmax(batch["y"], self.y_min, self.y_max))

        depth_emb = self.depth_projection(normalize_depth(batch["depth"]).unsqueeze(-1))
        orientation_sin_emb = self.orientation_sin_projection(
            batch["orientation_sin"].to(torch.float32).unsqueeze(-1)
        )
        orientation_cos_emb = self.orientation_cos_projection(
            batch["orientation_cos"].to(torch.float32).unsqueeze(-1)
        )
        size_emb = self.size_projection(normalize_size(batch["size"]).unsqueeze(-1))

        # Move-delta features
        delta_x_emb = self.delta_x_projection(
            batch["delta_x_prev"].to(torch.float32).unsqueeze(-1)
        )
        delta_y_emb = self.delta_y_projection(
            batch["delta_y_prev"].to(torch.float32).unsqueeze(-1)
        )

        parts = [
            type_emb, function_emb, role_emb, hole_emb,
            x_emb, y_emb,
            depth_emb, orientation_sin_emb, orientation_cos_emb, size_emb,
            delta_x_emb, delta_y_emb,
        ]

        # Optional: grip category
        if self.grip_category_embedding is not None and "grip_category_id" in batch:
            parts.append(self.grip_category_embedding(batch["grip_category_id"]))

        # Optional: nearest-neighbour distance (encoder-only)
        if self.dist_projection is not None and "dist_to_nearest" in batch:
            parts.append(
                self.dist_projection(batch["dist_to_nearest"].to(torch.float32).unsqueeze(-1))
            )

        token_concat = torch.cat(parts, dim=-1)
        return self.token_projection(token_concat)


# ---------------------------------------------------------------------------
# Runtime utilities
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_minibatches(items: list[Any], batch_size: int, *, shuffle: bool = False):
    import random
    indices = list(range(len(items)))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        yield [items[j] for j in indices[i:i + batch_size]]


def filter_samples_by_decoder_max_len(
    samples: list[Any], *, max_seq_len: int
) -> tuple[list[Any], int]:
    """Drop samples whose token length exceeds the decoder's max target length.

    Decoder input length = target length + 1 (BOS), so valid target len <= max_seq_len - 1.
    """
    max_target_len = max_seq_len - 1
    kept: list[Any] = []
    skipped = 0
    for sample in samples:
        token_len = int(sample.tokens["type_encoded_id"].shape[0])
        if token_len <= max_target_len:
            kept.append(sample)
        else:
            skipped += 1
    return kept, skipped


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(
    checkpoint_path: Any,
    vocabs: Any,
    device: torch.device,
    *,
    latent_dim_override: int | None = None,
) -> tuple[Any, Any]:  # (RouteConditionalVAE, DecoderEOSIds)
    """Build and load a RouteConditionalVAE from a saved checkpoint.

    Reads encoder/decoder hyperparameters from the checkpoint's saved args,
    reconstructs the model architecture, loads weights, and returns eval mode model.
    """
    from .route_vae_encoder import RouteTransformerEncoder
    from .route_vae_decoder import RouteTransformerDecoder, RouteVAEDecoderConfig
    from .route_vae_bottleneck import RouteConditionalVAE, RouteVAEBottleneck, RouteVAEBottleneckConfig
    from .training_utils import DecoderEOSIds

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args: dict[str, Any] = dict(ckpt.get("args", {}))

    enc_cfg = vocabs.to_transformer_config()
    enc_cfg.use_condition = bool(ckpt_args.get("encoder_use_condition", True))
    enc_cfg.use_cond_adaln = bool(ckpt_args.get("encoder_use_cond_adaln", True))

    latent_dim = int(
        latent_dim_override if latent_dim_override is not None
        else ckpt_args.get("latent_dim", 32)
    )
    dec_cfg = RouteVAEDecoderConfig(
        type_vocab_size=enc_cfg.type_vocab_size,
        function_vocab_size=enc_cfg.function_vocab_size,
        role_vocab_size=enc_cfg.role_vocab_size,
        hole_id_vocab_size=enc_cfg.hole_id_vocab_size,
        latent_dim=latent_dim,
        use_cond_adaln=bool(ckpt_args.get("decoder_use_cond_adaln", True)),
        z_memory_tokens=int(ckpt_args.get("decoder_z_memory_tokens", 4)),
        # Match encoder embedding dims (stored in checkpoint args for backward compat)
        type_embed_dim=int(ckpt_args.get("type_embed_dim", 32)),
        grip_category_vocab_size=enc_cfg.grip_category_vocab_size,
        grip_category_embed_dim=enc_cfg.grip_category_embed_dim,
        delta_embed_dim=enc_cfg.delta_embed_dim,
        use_dist_to_nearest=False,  # decoder never uses full-sequence dist feature
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
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    eos_ids = DecoderEOSIds(
        type_eos_id=model.decoder.type_eos_id,
        function_eos_id=model.decoder.function_eos_id,
        role_eos_id=model.decoder.role_eos_id,
        hole_eos_id=model.decoder.hole_eos_id,
    )
    return model, eos_ids


__all__ = [
    "ScalarSinusoidalEmbedding",
    "ConditionAdaLayerNorm",
    "HoldTokenEmbedder",
    "normalize_minmax",
    "normalize_depth",
    "normalize_size",
    "select_device",
    "iter_minibatches",
    "filter_samples_by_decoder_max_len",
    "build_model_from_checkpoint",
]
