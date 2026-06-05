from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model_utils import (
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

    # Latent size
    latent_dim: int

    # Per-feature embedding sizes (must match encoder)
    type_embed_dim: int = 32
    role_embed_dim: int = 8
    hole_id_embed_dim: int = 16
    x_embed_dim: int = 16
    y_embed_dim: int = 16
    depth_embed_dim: int = 8
    orientation_sin_embed_dim: int = 8
    orientation_cos_embed_dim: int = 8
    size_embed_dim: int = 8

    # Ablation flags — must match encoder (and the preprocessed cache).
    use_absolute_pos: bool = True
    use_type_feature: bool = True

    # Token dropout: during training, replace each decoder input token (positions 1+,
    # never BOS) with a learned mask embedding with this probability.  Forces the
    # decoder to rely on z rather than exploiting categorical / spatial patterns in
    # the previous-token context.  Disabled at inference (self.training=False).
    token_dropout: float = 0.0

    # Masked bidirectional mode: when > 0, the decoder operates without a causal
    # mask and randomly replaces this fraction of non-padded input tokens with
    # mask_token.  Loss is computed only at masked positions.  token_dropout and
    # BOS injection are ignored.  Applied at both train and eval so val loss
    # measures masked-hold reconstruction quality (not trivial full-context recall).
    mask_rate: float = 0.0

    # Decoder transformer sizes
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    max_seq_len: int = 128

    # Input normalization ranges (keep consistent with encoder)
    x_min: float = -20.0
    x_max: float = 164.0
    y_min: float = 4.0
    y_max: float = 176.0


class RouteTransformerDecoder(nn.Module):
    """Autoregressive transformer decoder.

    Cross-attention memory comes from latent z (the sole conditioning signal).

    Expected teacher-forcing input keys (shifted-right with BOS at position 0):
    - type_encoded_id, role_encoded_id, hole_encoded_id
    - x, y, orientation_sin, orientation_cos, size
    - padding_mask (bool, True for padded tokens)
    """

    is_autoregressive: bool = True  # distinguishes from RouteParallelDecoder (is_autoregressive=False)

    def __init__(self, cfg: RouteVAEDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # EOS ids for each categorical output head (base vocab + 1 EOS slot each).
        self.type_eos_id = cfg.type_vocab_size
        self.role_eos_id = cfg.role_vocab_size
        self.hole_eos_id = cfg.hole_id_vocab_size

        # Depth and delta features excluded: depth is a physical hold property not reconstructed
        # by the decoder; delta is derivable from predicted x/y.
        self.hold_embedder = HoldTokenEmbedder(cfg, use_depth=False)

        # Learned sequence position embedding
        self.sequence_position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        # Learned BOS anchor token
        self.bos_embedding = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        # Learned mask token: substituted for dropped hold tokens during training.
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))

        # Project z into a single d_model-dim memory token for cross-attention.
        self.z_proj = nn.Linear(cfg.latent_dim, cfg.d_model)

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

    def _build_decoder_tokens(
        self, batch: dict[str, torch.Tensor], *, effective_mask_rate: float
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Embed hold tokens, add position embeddings, then either inject BOS (AR) or mask holds (masked).

        Returns:
            (tokens, is_masked) where is_masked is [B, L] bool in masked mode, None in AR mode.
        """
        tokens = self.hold_embedder.embed(batch)

        batch_size, seq_len, _ = tokens.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}.")

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        tokens = tokens + self.sequence_position_embedding(positions)

        if self.cfg.mask_rate > 0.0:
            # Masked bidirectional mode. effective_mask_rate may differ from cfg.mask_rate
            # when the training loop samples dynamically; cfg.mask_rate is the eval fallback.
            padding_mask = batch["padding_mask"].bool()
            rng = torch.rand(batch_size, seq_len, device=tokens.device)
            is_masked = (rng < effective_mask_rate) & ~padding_mask
            if effective_mask_rate > 0.0:
                mask_emb = self.mask_token.view(1, 1, -1).expand_as(tokens)
                tokens = torch.where(is_masked.unsqueeze(-1), mask_emb, tokens)
            return tokens, is_masked

        # Autoregressive mode: inject BOS at position 0, apply optional token dropout.
        bos = (self.bos_embedding + self.sequence_position_embedding.weight[0])
        bos = bos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        tokens = torch.cat([bos, tokens[:, 1:, :]], dim=1)

        hold_tokens = tokens[:, 1:, :]
        if self.training and self.cfg.token_dropout > 0.0:
            mask_emb = self.mask_token.view(1, 1, -1).expand_as(hold_tokens)
            drop = torch.rand(batch_size, hold_tokens.shape[1], device=tokens.device) < self.cfg.token_dropout
            hold_tokens = torch.where(drop.unsqueeze(-1), mask_emb, hold_tokens)
            tokens = torch.cat([tokens[:, :1, :], hold_tokens], dim=1)

        return tokens, None

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        z: torch.Tensor,
        mask_rate: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forcing forward pass.

        Args:
            batch:     tokenized decoder inputs
            z:         latent vector [B, latent_dim] — sole conditioning signal
            mask_rate: override cfg.mask_rate for this forward pass (used by the training loop
                       to sample a fresh rate each batch). None → use cfg.mask_rate.
        """
        effective_mask_rate = mask_rate if mask_rate is not None else self.cfg.mask_rate
        padding_mask = batch["padding_mask"].bool()
        tgt, is_masked = self._build_decoder_tokens(batch, effective_mask_rate=effective_mask_rate)

        memory = self._build_condition_memory(z=z)

        if self.cfg.mask_rate > 0.0:
            # Bidirectional: no causal mask; every token attends to every other.
            decoded = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_key_padding_mask=padding_mask,
            )
        else:
            tgt_mask = self._causal_mask(tgt.shape[1], tgt.device)
            decoded = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=padding_mask,
            )
        decoded = self.final_norm(decoded)

        out = {
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
        if is_masked is not None:
            out["is_masked"] = is_masked
        return out


__all__ = ["RouteVAEDecoderConfig", "RouteTransformerDecoder"]
