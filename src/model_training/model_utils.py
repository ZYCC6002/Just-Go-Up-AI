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
    - delta_x_prev, delta_y_prev: move vector from the previous hold (y-sorted sequence)
    - dist_to_nearest: distance to the nearest other hold (encoder-only; set
      use_dist_to_nearest=False on the decoder config to skip this feature)
    """

    def __init__(self, cfg: Any, *, use_delta: bool = True, use_depth: bool = True) -> None:
        super().__init__()
        self.x_min: float = cfg.x_min
        self.x_max: float = cfg.x_max
        self.y_min: float = cfg.y_min
        self.y_max: float = cfg.y_max

        use_abs = getattr(cfg, "use_absolute_pos", True)
        use_type = getattr(cfg, "use_type_feature", True)

        # Categorical embeddings
        if use_type:
            self.type_embedding: nn.Embedding | None = nn.Embedding(cfg.type_vocab_size, cfg.type_embed_dim)
            self._type_dim = cfg.type_embed_dim
        else:
            self.type_embedding = None
            self._type_dim = 0

        self.role_embedding = nn.Embedding(cfg.role_vocab_size, cfg.role_embed_dim)
        self.hole_embedding = nn.Embedding(cfg.hole_id_vocab_size, cfg.hole_id_embed_dim)

        # Sinusoidal positional embeddings for x/y (absolute board coordinates)
        if use_abs:
            self.x_embedding: ScalarSinusoidalEmbedding | None = ScalarSinusoidalEmbedding(cfg.x_embed_dim)
            self.y_embedding: ScalarSinusoidalEmbedding | None = ScalarSinusoidalEmbedding(cfg.y_embed_dim)
            self._xy_dim = cfg.x_embed_dim + cfg.y_embed_dim
        else:
            self.x_embedding = None
            self.y_embedding = None
            self._xy_dim = 0

        # Depth (encoder only — not reconstructed by decoder)
        if use_depth:
            self.depth_projection: nn.Linear | None = nn.Linear(1, cfg.depth_embed_dim)
            self._depth_dim = cfg.depth_embed_dim
        else:
            self.depth_projection = None
            self._depth_dim = 0

        self.orientation_sin_projection = nn.Linear(1, cfg.orientation_sin_embed_dim)
        self.orientation_cos_projection = nn.Linear(1, cfg.orientation_cos_embed_dim)
        self.size_projection = nn.Linear(1, cfg.size_embed_dim)

        # Move-delta projections (encoder only — derivable from predicted x/y in decoder)
        delta_dim = getattr(cfg, "delta_embed_dim", 8)
        if use_delta:
            self.delta_x_projection: nn.Linear | None = nn.Linear(1, delta_dim)
            self.delta_y_projection: nn.Linear | None = nn.Linear(1, delta_dim)
            self._delta_dim = delta_dim
        else:
            self.delta_x_projection = None
            self.delta_y_projection = None
            self._delta_dim = 0

        # k-NN neighbourhood features (encoder-only; requires full hold set)
        use_knn = getattr(cfg, "use_knn_features", True)
        knn_embed_dim = getattr(cfg, "knn_embed_dim", 16)
        if use_knn:
            self.knn_projection: nn.Linear | None = nn.Linear(9, knn_embed_dim)
            self._knn_dim = knn_embed_dim
        else:
            self.knn_projection = None
            self._knn_dim = 0

        token_input_dim = (
            self._type_dim      # 0 if use_type_feature=False
            + cfg.role_embed_dim
            + cfg.hole_id_embed_dim
            + self._xy_dim      # 0 if use_absolute_pos=False
            + self._depth_dim   # 0 if use_depth=False
            + cfg.orientation_sin_embed_dim
            + cfg.orientation_cos_embed_dim
            + cfg.size_embed_dim
            + self._delta_dim * 2   # 0 if use_delta=False
            + self._knn_dim         # 0 if use_knn_features=False
        )
        self.token_projection = nn.Sequential(
            nn.Linear(token_input_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps),
            nn.Dropout(cfg.dropout),
        )

    def embed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed a batch of hold tokens to [B, L, d_model]."""
        role_emb = self.role_embedding(batch["role_encoded_id"])
        hole_emb = self.hole_embedding(batch["hole_encoded_id"])

        orientation_sin_emb = self.orientation_sin_projection(
            batch["orientation_sin"].to(torch.float32).unsqueeze(-1)
        )
        orientation_cos_emb = self.orientation_cos_projection(
            batch["orientation_cos"].to(torch.float32).unsqueeze(-1)
        )
        size_emb = self.size_projection(normalize_size(batch["size"]).unsqueeze(-1))

        parts: list[torch.Tensor] = []

        # Optional: hold type categorical embedding
        if self.type_embedding is not None:
            parts.append(self.type_embedding(batch["type_encoded_id"]))

        parts.extend([role_emb, hole_emb])

        # Optional: absolute board coordinates (x/y sinusoidal)
        if self.x_embedding is not None:
            parts.append(self.x_embedding(normalize_minmax(batch["x"], self.x_min, self.x_max)))
            parts.append(self.y_embedding(normalize_minmax(batch["y"], self.y_min, self.y_max)))

        # Optional: depth (encoder only — physical hold property, not reconstructed by decoder)
        if self.depth_projection is not None:
            parts.append(self.depth_projection(normalize_depth(batch["depth"]).unsqueeze(-1)))

        parts.extend([orientation_sin_emb, orientation_cos_emb, size_emb])

        # Optional: move-delta features (encoder only)
        if self.delta_x_projection is not None:
            parts.append(self.delta_x_projection(batch["delta_x_prev"].to(torch.float32).unsqueeze(-1)))
            parts.append(self.delta_y_projection(batch["delta_y_prev"].to(torch.float32).unsqueeze(-1)))

        # Optional: k-NN neighbourhood features (encoder-only; [B, L, 9] → [B, L, knn_embed_dim])
        if self.knn_projection is not None and "knn_features" in batch:
            parts.append(
                self.knn_projection(batch["knn_features"].to(torch.float32))
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
# State dict helpers
# ---------------------------------------------------------------------------

def _adapt_pool_state_dict(state_dict: dict, d_model: int) -> dict:
    """Adapt old single-query pool weights to the new multi-query format.

    Old format (single-query attention pooling):
        encoder.pool_query          [d_model]      → rename + unsqueeze
        encoder.pool_attn.*                         → unchanged (shape is nhead-agnostic)
        (no pool_proj)

    New format (multi-query attention pooling):
        encoder.pool_queries        [K, d_model]
        encoder.pool_attn.*
        encoder.pool_proj.weight    [d_model, K*d_model]
        encoder.pool_proj.bias      [d_model]

    When K=1 (old checkpoint): pool_proj becomes an identity Linear(d_model, d_model).
    Pre-pooling checkpoints ("cls" mode) have neither key — no change needed.
    """
    import torch
    if "encoder.pool_queries" in state_dict:
        return state_dict  # already new multi-query format
    if "encoder.pool_query" not in state_dict:
        return state_dict  # pre-pooling "cls" checkpoint — no adaptation needed

    # Old single-query: rename and reshape
    old_q = state_dict.pop("encoder.pool_query")           # [d_model]
    state_dict["encoder.pool_queries"] = old_q.unsqueeze(0)  # [1, d_model]

    # Add identity pool_proj (Linear(d_model, d_model)) so K=1 is a no-op projection
    state_dict["encoder.pool_proj.weight"] = torch.eye(d_model, dtype=old_q.dtype)
    state_dict["encoder.pool_proj.bias"]   = torch.zeros(d_model, dtype=old_q.dtype)

    return state_dict


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

    Vocab priority (highest to lowest):
      1. Checkpoint-embedded vocabs (``ckpt["vocabs"]``) — present in checkpoints
         saved after the vocabs-in-checkpoint change.  These are the exact vocabs
         used during training, so the model architecture always matches the weights.
      2. Runtime ``vocabs`` argument — used for old checkpoints that predate the
         embedded-vocabs change, and during training itself (before the first save).
    """
    from .route_vae_encoder import RouteTransformerEncoder
    from .route_vae_decoder import RouteTransformerDecoder, RouteVAEDecoderConfig
    from .route_vae_bottleneck import RouteConditionalVAE, RouteVAEBottleneck, RouteVAEBottleneckConfig
    from .training_utils import DecoderEOSIds

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args: dict[str, Any] = dict(ckpt.get("args", {}))

    # Prefer vocabs embedded in the checkpoint so the model architecture always
    # matches the saved weights, regardless of what the local DB produces.
    ckpt_vocabs = ckpt.get("vocabs")
    effective_vocabs = ckpt_vocabs if ckpt_vocabs is not None else vocabs
    if ckpt_vocabs is not None and ckpt_vocabs is not vocabs:
        print("build_model_from_checkpoint: using checkpoint-embedded vocabs.")

    enc_cfg = effective_vocabs.to_transformer_config(
        d_model=int(ckpt_args.get("encoder_d_model", 256)),
        nhead=int(ckpt_args.get("encoder_nhead", 8)),
        num_layers=int(ckpt_args.get("encoder_num_layers", 6)),
        dim_feedforward=int(ckpt_args.get("encoder_dim_feedforward", 1024)),
    )
    enc_cfg.use_cond_adaln = bool(ckpt_args.get("encoder_use_cond_adaln", True))
    enc_cfg.use_absolute_pos = bool(ckpt_args.get("use_absolute_pos", True))
    enc_cfg.use_type_feature = bool(ckpt_args.get("use_type_feature", True))
    # Spatial feature flags — default True so old checkpoints (which always had these on) load correctly.
    enc_cfg.use_knn_features = bool(ckpt_args.get("use_knn_features", True))
    enc_cfg.use_delta_features = bool(ckpt_args.get("use_delta_features", True))
    enc_cfg.shape_desc_dim = int(ckpt_args.get("shape_desc_dim", 9))
    # Default to "cls" for old checkpoints that pre-date attention pooling —
    # those checkpoints have no pool_queries / pool_attn weights, so "attention"
    # would cause a strict load_state_dict failure.
    enc_cfg.route_pool_mode = str(ckpt_args.get("route_pool_mode", "cls"))
    # Pool architecture — old single-query checkpoints default to K=1 / nhead=encoder_nhead
    # so their pool_query [d_model] and pool_attn weights load cleanly after key adaptation.
    enc_cfg.pool_num_queries = int(ckpt_args.get("pool_num_queries", 1))
    enc_cfg.pool_nhead = int(ckpt_args.get("pool_nhead", int(ckpt_args.get("encoder_nhead", 8))))

    latent_dim = int(
        latent_dim_override if latent_dim_override is not None
        else ckpt_args.get("latent_dim", 32)
    )
    dec_cfg = RouteVAEDecoderConfig(
        type_vocab_size=enc_cfg.type_vocab_size,
        role_vocab_size=enc_cfg.role_vocab_size,
        hole_id_vocab_size=enc_cfg.hole_id_vocab_size,
        latent_dim=latent_dim,
        use_cond_adaln=bool(ckpt_args.get("decoder_use_cond_adaln", True)),
        d_model=int(ckpt_args.get("decoder_d_model", 128)),
        num_layers=int(ckpt_args.get("decoder_num_layers", 4)),
        dim_feedforward=int(ckpt_args.get("decoder_dim_feedforward", 512)),
        # Match encoder embedding dims (stored in checkpoint args for backward compat)
        type_embed_dim=int(ckpt_args.get("type_embed_dim", 32)),
        use_knn_features=False,    # decoder never uses full-sequence knn features
        use_absolute_pos=enc_cfg.use_absolute_pos,
        use_type_feature=enc_cfg.use_type_feature,
        # token_dropout is a training-only flag; default 0.0 so old checkpoints
        # (which have no mask_token weight) load cleanly without it.
        token_dropout=float(ckpt_args.get("decoder_token_dropout", 0.0)),
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
    state_dict = _adapt_pool_state_dict(dict(ckpt["model_state_dict"]), enc_cfg.d_model)
    # Back-compat: old checkpoints predate mask_token; initialise to zeros so
    # the architecture matches without requiring a strict-load failure.
    if "decoder.mask_token" not in state_dict:
        state_dict["decoder.mask_token"] = torch.zeros(dec_cfg.d_model, device=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    eos_ids = DecoderEOSIds(
        type_eos_id=model.decoder.type_eos_id,
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
