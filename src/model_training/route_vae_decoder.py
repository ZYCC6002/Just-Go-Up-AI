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

    # k-NN neighbourhood features: disabled for decoder (requires full sequence,
    # unavailable during autoregressive generation; only the encoder gets this feature)
    use_knn_features: bool = False

    # Ablation flags — must match encoder (and the preprocessed cache).
    use_absolute_pos: bool = True
    use_type_feature: bool = True

    # Token dropout: during training, replace each decoder input token (positions 1+,
    # never BOS) with a learned mask embedding with this probability.  Forces the
    # decoder to rely on z rather than exploiting categorical / spatial patterns in
    # the previous-token context.  Disabled at inference (self.training=False).
    token_dropout: float = 0.0

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
    - x, y, orientation_sin, orientation_cos, size
    - padding_mask (bool, True for padded tokens)
    """

    def __init__(self, cfg: RouteVAEDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # EOS ids for each categorical output head (base vocab + 1 EOS slot each).
        self.type_eos_id = cfg.type_vocab_size
        self.role_eos_id = cfg.role_vocab_size
        self.hole_eos_id = cfg.hole_id_vocab_size

        # Depth and delta features excluded: depth is a physical hold property not reconstructed
        # by the decoder; delta is derivable from predicted x/y and teacher-forcing with
        # ground-truth delta creates a train/inference mismatch.
        self.hold_embedder = HoldTokenEmbedder(cfg, use_delta=False, use_depth=False)

        # Learned sequence position embedding
        self.sequence_position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        # Learned BOS anchor token
        self.bos_embedding = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        # Learned mask token: substituted for dropped hold tokens during training.
        # Initialised to zero so early training sees a neutral (near-mean) signal.
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))

        # Project z into a single d_model-dim memory token for cross-attention.
        # The cross-attention's internal K/V projections do the rest of the work;
        # no MLP expansion or multi-token positional tricks needed.
        self.z_proj = nn.Linear(cfg.latent_dim, cfg.d_model)

        # Angle/grade condition pathway for AdaLN
        self.angle_mlp = nn.Sequential(
            nn.Linear(1, cfg.condition_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.condition_hidden_dim, cfg.d_model),
        )
        self.grade_mlp = nn.Sequential(
            nn.Linear(1, cfg.condition_hidden_dim),
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

        # Output heads: 3 categorical (base vocab + EOS) + 5 numeric
        self.type_head = nn.Linear(cfg.d_model, cfg.type_vocab_size + 1)
        self.role_head = nn.Linear(cfg.d_model, cfg.role_vocab_size + 1)
        self.hole_head = nn.Linear(cfg.d_model, cfg.hole_id_vocab_size + 1)
        self.x_head = nn.Linear(cfg.d_model, 1)
        self.y_head = nn.Linear(cfg.d_model, 1)
        self.orientation_sin_head = nn.Linear(cfg.d_model, 1)
        self.orientation_cos_head = nn.Linear(cfg.d_model, 1)
        self.size_head = nn.Linear(cfg.d_model, 1)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        return torch.triu(mask, diagonal=1)

    def _build_condition_memory(self, *, z: torch.Tensor) -> torch.Tensor:
        """Project z to a single d_model memory token for cross-attention. Returns [B, 1, d_model]."""
        return self.z_proj(z.to(torch.float32)).unsqueeze(1)

    def _build_condition_embedding(
        self,
        *,
        angle: torch.Tensor,
        grade: torch.Tensor,
    ) -> torch.Tensor:
        """Build route-level condition embedding for AdaLN from angle/grade."""
        angle_emb = self.angle_mlp(
            normalize_minmax(angle, self.cfg.angle_min, self.cfg.angle_max).unsqueeze(-1)
        )
        grade_emb = self.grade_mlp(
            normalize_minmax(grade, self.cfg.grade_min, self.cfg.grade_max).unsqueeze(-1)
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

        # Token masking: replaces hold tokens (positions 1+) with the learned mask
        # embedding.  BOS is never masked.
        #
        # Two modes:
        #   mask_all=True  — all hold tokens replaced (eval-time, batched non-teacher-forced pass)
        #   training + token_dropout > 0 — random stochastic dropout (training regularisation)
        hold_tokens = tokens[:, 1:, :]
        mask_emb = self.mask_token.view(1, 1, -1).expand_as(hold_tokens)
        if self._mask_all:
            hold_tokens = mask_emb
        elif self.training and self.cfg.token_dropout > 0.0:
            drop = torch.rand(batch_size, hold_tokens.shape[1], device=tokens.device) < self.cfg.token_dropout
            hold_tokens = torch.where(drop.unsqueeze(-1), mask_emb, hold_tokens)
        tokens = torch.cat([tokens[:, :1, :], hold_tokens], dim=1)

        return tokens

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        z: torch.Tensor,
        angle: torch.Tensor,
        grade: torch.Tensor,
        mask_inputs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forcing forward pass.

        Args:
            batch:       tokenized decoder inputs (shifted-right targets with BOS at position 0)
            z:           latent vector [B, latent_dim]
            angle:       route angle condition [B]
            grade:       route grade condition [B]
            mask_inputs: if True, replace ALL hold tokens with the learned mask embedding
                         before the transformer stack.  Used for batched non-teacher-forced
                         validation: the decoder sees only BOS + z (via cross-attention) +
                         position embeddings, with no ground-truth token context.
        """
        self._mask_all = mask_inputs
        padding_mask = batch["padding_mask"].bool()
        tgt = self._build_decoder_tokens(batch)

        memory = self._build_condition_memory(z=z)
        cond_emb = self._build_condition_embedding(angle=angle, grade=grade)

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
            "orientation_sin_pred": torch.sigmoid(self.orientation_sin_head(decoded).squeeze(-1)),
            "orientation_cos_pred": torch.sigmoid(self.orientation_cos_head(decoded).squeeze(-1)),
            "size_pred": torch.sigmoid(self.size_head(decoded).squeeze(-1)),
        }


__all__ = ["RouteVAEDecoderConfig", "RouteTransformerDecoder"]
