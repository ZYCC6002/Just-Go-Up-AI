from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model_utils import (
    ConditionAdaLayerNorm,
    HoldTokenEmbedder,
    normalize_minmax,
)


@dataclass
class RouteVAEDecoderConfig:
    """Configuration for autoregressive route decoder."""

    # Feature vocab sizes
    type_vocab_size: int
    role_vocab_size: int
    hole_id_vocab_size: int

    # Latent / conditioning sizes
    latent_dim: int
    condition_hidden_dim: int = 64
    use_cond_adaln: bool = True
    z_memory_tokens: int = 4

    # Per-feature embedding sizes (must match encoder)
    type_embed_dim: int = 32          # increased from 16 to match encoder
    role_embed_dim: int = 8
    hole_id_embed_dim: int = 16
    x_embed_dim: int = 16
    y_embed_dim: int = 16
    depth_embed_dim: int = 8
    orientation_sin_embed_dim: int = 8
    orientation_cos_embed_dim: int = 8
    size_embed_dim: int = 8

    # Move delta embeddings (decoder uses delta_x_prev and delta_y_prev)
    delta_embed_dim: int = 8

    # k-NN neighbourhood features: disabled for decoder (requires full sequence,
    # unavailable during autoregressive generation; only the encoder gets this feature)
    use_knn_features: bool = False

    # Decoder transformer sizes
    # d_model is deliberately kept at 128 (encoder uses 256).
    # The decoder's job is conditional generation given z + angle/grade — it doesn't
    # need the same representational richness as the style encoder. Keeping it lighter
    # also speeds up the autoregressive training inner loop.
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512   # 4× d_model (standard ratio); was 256 = 2×
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    max_seq_len: int = 128

    # Input normalization ranges (keep consistent with encoder)
    x_min: float = -20.0
    x_max: float = 164.0
    y_min: float = 4.0
    y_max: float = 176.0
    angle_min: float = 0.0
    angle_max: float = 70.0
    grade_min: float = 10.0
    grade_max: float = 33.0


class RouteTransformerDecoder(nn.Module):
    """Autoregressive transformer decoder.

    Cross-attention memory comes from latent z.
    Angle/grade conditioning is injected via AdaLN modulation.

    Expected teacher-forcing input keys (shifted-right with BOS at position 0):
    - type_encoded_id, role_encoded_id, hole_encoded_id
    - x, y, depth, orientation_sin, orientation_cos, size
    - padding_mask (bool, True for padded tokens)
    """

    def __init__(self, cfg: RouteVAEDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # EOS ids for each categorical output head (base vocab + 1 EOS slot each).
        self.type_eos_id = cfg.type_vocab_size
        self.role_eos_id = cfg.role_vocab_size
        self.hole_eos_id = cfg.hole_id_vocab_size

        self.hold_embedder = HoldTokenEmbedder(cfg)

        # Learned sequence position embedding
        self.sequence_position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        # Learned BOS anchor token
        self.bos_embedding = nn.Parameter(torch.randn(cfg.d_model) * 0.02)

        # Latent memory tokens (cross-attention keys/values)
        if cfg.z_memory_tokens < 1:
            raise ValueError("z_memory_tokens must be >= 1")
        self.z_memory_tokens = cfg.z_memory_tokens
        self.z_mlp = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.condition_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.condition_hidden_dim, cfg.d_model * cfg.z_memory_tokens),
        )
        self.z_memory_positional = nn.Parameter(torch.randn(cfg.z_memory_tokens, cfg.d_model) * 0.02)

        # Angle/grade condition pathway for AdaLN
        self.angle_mlp = nn.Sequential(
            nn.Linear(1, cfg.condition_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.condition_hidden_dim, cfg.d_model),
        )
        self.grade_mlp = nn.Sequential(
            nn.Linear(2, cfg.condition_hidden_dim),  # [grade, grade_missing]
            nn.GELU(),
            nn.Linear(cfg.condition_hidden_dim, cfg.d_model),
        )
        self.condition_fusion = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.condition_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.condition_hidden_dim, cfg.d_model),
        )

        if cfg.use_cond_adaln:
            # Only post-decoder AdaLN is kept. A pre-decoder AdaLN was previously
            # used here but was removed: injecting a strong angle/grade signal
            # BEFORE the transformer allowed the decoder to satisfy reconstruction
            # without ever attending to z, causing posterior collapse. A single
            # post-decoder modulation retains grade/angle conditioning while
            # significantly weakening this shortcut.
            self.post_decoder_adaln = ConditionAdaLayerNorm(cfg.d_model, cfg.d_model, cfg.layer_norm_eps)
        else:
            self.post_decoder_adaln = None

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            layer_norm_eps=cfg.layer_norm_eps,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_layers)
        self.final_norm = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

        # Output heads: 3 categorical (base vocab + EOS) + 6 numeric
        self.type_head = nn.Linear(cfg.d_model, cfg.type_vocab_size + 1)
        self.role_head = nn.Linear(cfg.d_model, cfg.role_vocab_size + 1)
        self.hole_head = nn.Linear(cfg.d_model, cfg.hole_id_vocab_size + 1)
        self.x_head = nn.Linear(cfg.d_model, 1)
        self.y_head = nn.Linear(cfg.d_model, 1)
        self.depth_head = nn.Linear(cfg.d_model, 1)
        self.orientation_sin_head = nn.Linear(cfg.d_model, 1)
        self.orientation_cos_head = nn.Linear(cfg.d_model, 1)
        self.size_head = nn.Linear(cfg.d_model, 1)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        return torch.triu(mask, diagonal=1)

    def _build_condition_memory(self, *, z: torch.Tensor) -> torch.Tensor:
        """Expand z into memory tokens for cross-attention. Returns [B, z_memory_tokens, d_model]."""
        batch_size = z.shape[0]
        z_tokens = self.z_mlp(z.to(torch.float32)).view(batch_size, self.z_memory_tokens, self.cfg.d_model)
        return z_tokens + self.z_memory_positional.unsqueeze(0)

    def _build_condition_embedding(
        self,
        *,
        angle: torch.Tensor,
        grade: torch.Tensor,
        grade_missing: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build route-level condition embedding for AdaLN from angle/grade."""
        if grade_missing is None:
            grade_missing = torch.zeros_like(grade, dtype=torch.float32)
        else:
            grade_missing = grade_missing.to(torch.float32)

        angle_emb = self.angle_mlp(
            normalize_minmax(angle, self.cfg.angle_min, self.cfg.angle_max).unsqueeze(-1)
        )
        grade_emb = self.grade_mlp(
            torch.stack([
                normalize_minmax(grade, self.cfg.grade_min, self.cfg.grade_max),
                grade_missing,
            ], dim=-1)
        )
        return self.condition_fusion(torch.cat([angle_emb, grade_emb], dim=-1))

    def _build_decoder_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed hold tokens, add position embeddings, and inject BOS at position 0."""
        tokens = self.hold_embedder.embed(batch)

        batch_size, seq_len, _ = tokens.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}.")

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        tokens = tokens + self.sequence_position_embedding(positions)

        # Replace slot 0 with learned BOS token
        bos = (self.bos_embedding + self.sequence_position_embedding.weight[0])
        bos = bos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        tokens = torch.cat([bos, tokens[:, 1:, :]], dim=1)
        return tokens

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        z: torch.Tensor,
        angle: torch.Tensor,
        grade: torch.Tensor,
        grade_missing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forcing forward pass.

        Args:
            batch: tokenized decoder inputs (shifted-right targets with BOS at position 0)
            z:     latent vector [B, latent_dim]
            angle: route angle condition [B]
            grade: route grade condition [B]
            grade_missing: optional mask [B] (1.0 if grade missing)
        """
        padding_mask = batch["padding_mask"].bool()
        tgt = self._build_decoder_tokens(batch)

        memory = self._build_condition_memory(z=z)
        cond_emb = self._build_condition_embedding(angle=angle, grade=grade, grade_missing=grade_missing)

        tgt_mask = self._causal_mask(tgt.shape[1], tgt.device)
        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask,
        )

        if self.post_decoder_adaln is not None:
            decoded = self.post_decoder_adaln(decoded, cond_emb)
        decoded = self.final_norm(decoded)

        return {
            "hidden_states": decoded,
            "type_logits": self.type_head(decoded),
            "role_logits": self.role_head(decoded),
            "hole_logits": self.hole_head(decoded),
            "x_pred": torch.sigmoid(self.x_head(decoded).squeeze(-1)),
            "y_pred": torch.sigmoid(self.y_head(decoded).squeeze(-1)),
            "depth_pred": torch.sigmoid(self.depth_head(decoded).squeeze(-1)),
            "orientation_sin_pred": torch.sigmoid(self.orientation_sin_head(decoded).squeeze(-1)),
            "orientation_cos_pred": torch.sigmoid(self.orientation_cos_head(decoded).squeeze(-1)),
            "size_pred": torch.sigmoid(self.size_head(decoded).squeeze(-1)),
        }


__all__ = ["RouteVAEDecoderConfig", "RouteTransformerDecoder"]
