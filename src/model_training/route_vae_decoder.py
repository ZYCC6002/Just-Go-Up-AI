from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model_utils import (
    HoldTokenEmbedder,
)


@dataclass
class RouteVAEDecoderConfig:
    """Configuration for the route decoder (AR or masked bidirectional)."""

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


# ---------------------------------------------------------------------------
# AdaLN building blocks
# ---------------------------------------------------------------------------

class AdaLNModulation(nn.Module):
    """Maps a conditioning vector to (scale, shift, gate) for one AdaLN sublayer.

    Implements the full DiT-style modulation:
      - scale (γ): multiplies the LayerNorm output before the sublayer
      - shift (β): added to the LayerNorm output before the sublayer
      - gate (α): multiplies the sublayer output before the residual addition

    All weights and biases are zero-initialised so that at the start of training:
      scale = 1  (identity normalization)
      shift = 0  (no offset)
      gate  = 0  (sublayer contributes nothing — full residual passthrough)
    This makes each transformer layer a near-identity function at init, giving
    stable early-training dynamics.  The conditioning signal is gradually learned.

    The conditioning input is the z_mlp output from RouteTransformerDecoder.
    """

    def __init__(self, cond_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 3 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (scale, shift, gate).

        scale is 1-centred (starts at 1); shift and gate start at 0.
        """
        out = self.proj(cond)                          # [B, 3 * d_model]
        scale, shift, gate = out.chunk(3, dim=-1)
        return 1.0 + scale, shift, gate                # scale residual: γ=1, β=0, α=0 at init


class AdaLNTransformerLayer(nn.Module):
    """Single transformer layer with AdaLN z-conditioning (no cross-attention).

    z is injected via Adaptive Layer Normalisation (AdaLN) at both the
    self-attention and feed-forward sub-layers.  Unlike cross-attention —
    where the decoder can route around z by assigning near-zero attention
    weight to the z memory token — AdaLN modulates the LayerNorm scale and
    shift at every token at every layer.  The decoder has no architectural
    bypass: even if self-attention relies entirely on visible holds, z still
    reshapes every token's normalised representation before each sublayer.

    AdaLN is applied in the pre-norm position (as in DiT):
      x ← x + Attn(AdaLN(x, z))
      x ← x + FF(AdaLN(x, z))
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        layer_norm_eps: float,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.adaLN_attn = AdaLNModulation(cond_dim, d_model)
        self.adaLN_ff   = AdaLNModulation(cond_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        z_cond: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                [B, L, d_model]
            z_cond:           [B, cond_dim] — pre-processed z from z_mlp
            attn_mask:        [L, L] bool (True = mask) for causal AR mode; None for bidirectional
            key_padding_mask: [B, L] bool (True = padded position)
        """
        # ── Self-attention sub-layer: AdaLN → Attn → gate → residual ───────
        scale1, shift1, gate1 = self.adaLN_attn(z_cond)             # [B, d_model] each
        normed = scale1.unsqueeze(1) * self.norm1(x) + shift1.unsqueeze(1)
        attn_out, _ = self.self_attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate1.unsqueeze(1) * attn_out                       # α1(z) gates attn residual

        # ── Feed-forward sub-layer: AdaLN → FF → gate → residual ────────────
        scale2, shift2, gate2 = self.adaLN_ff(z_cond)               # [B, d_model] each
        normed = scale2.unsqueeze(1) * self.norm2(x) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.ff(normed)                # α2(z) gates FF residual

        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class RouteTransformerDecoder(nn.Module):
    """Transformer decoder with AdaLN z-conditioning.

    z is injected via AdaLN (not cross-attention) so it cannot be bypassed
    regardless of how much visible-hold context self-attention provides.

    Supports two operating modes selected by cfg.mask_rate:
      mask_rate = 0  — autoregressive (causal self-attention, BOS token)
      mask_rate > 0  — masked bidirectional (no causal mask, mask_token injection)

    Expected batch keys (teacher-forcing or masked inputs):
    - type_encoded_id, role_encoded_id, hole_encoded_id
    - x, y, orientation_sin, orientation_cos, size
    - padding_mask (bool, True for padded tokens)
    """

    is_autoregressive: bool = True  # distinguishes from RouteParallelDecoder

    def __init__(self, cfg: RouteVAEDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # EOS ids for each categorical output head (base vocab + 1 EOS slot each).
        self.type_eos_id = cfg.type_vocab_size
        self.role_eos_id = cfg.role_vocab_size
        self.hole_eos_id = cfg.hole_id_vocab_size

        # Depth excluded: physical hold property not reconstructed by the decoder.
        self.hold_embedder = HoldTokenEmbedder(cfg, use_depth=False)

        # Learned sequence position embedding
        self.sequence_position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        # Learned BOS anchor token (AR mode only — position 0 of the shifted sequence)
        self.bos_embedding = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        # Learned mask token (masked bidirectional mode and token dropout)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))

        # z pre-processor: latent_dim → d_model, shared across all AdaLN layers.
        # A single SiLU linear is sufficient; per-layer AdaLN projections add
        # further expressivity on top of this shared representation.
        self.z_mlp = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.d_model),
            nn.SiLU(),
        )

        # AdaLN transformer layers — no cross-attention sublayer.
        self.layers = nn.ModuleList([
            AdaLNTransformerLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                layer_norm_eps=cfg.layer_norm_eps,
                cond_dim=cfg.d_model,   # z_mlp output dim matches d_model
            )
            for _ in range(cfg.num_layers)
        ])
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
        """Upper-triangular bool mask (True = do not attend) for causal AR."""
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        return torch.triu(mask, diagonal=1)

    def _build_decoder_tokens(
        self, batch: dict[str, torch.Tensor], *, effective_mask_rate: float
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Embed hold tokens, add position embeddings, then either inject BOS (AR)
        or mask holds (masked bidirectional).

        Returns:
            (tokens, is_masked) where is_masked is [B, L] bool in masked mode, None in AR mode.
        """
        tokens = self.hold_embedder.embed(batch)

        batch_size, seq_len, _ = tokens.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}.")

        padding_mask = batch["padding_mask"].bool()  # [B, L], True = padding

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        tokens = tokens + self.sequence_position_embedding(positions)
        # Zero out padding positions after embedding + positional encoding.
        # Padding slots would otherwise carry type=0/x=0 embeddings that look like
        # real holds near the board origin; key_padding_mask blocks their attention
        # contribution, but zero-vectors are inert regardless of mask propagation.
        tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if self.cfg.mask_rate > 0.0:
            # Masked bidirectional mode. effective_mask_rate may differ from cfg.mask_rate
            # when the training loop samples dynamically; cfg.mask_rate is the eval fallback.
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
            drop = drop & ~padding_mask[:, 1:]
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
        """Forward pass.

        Args:
            batch:     tokenized decoder inputs
            z:         latent vector [B, latent_dim] — sole conditioning signal, injected via AdaLN
            mask_rate: override cfg.mask_rate for this forward pass (used by the training loop
                       to sample a fresh rate each batch). None → use cfg.mask_rate.
        """
        effective_mask_rate = mask_rate if mask_rate is not None else self.cfg.mask_rate
        padding_mask = batch["padding_mask"].bool()
        tgt, is_masked = self._build_decoder_tokens(batch, effective_mask_rate=effective_mask_rate)

        # Pre-process z once; each AdaLN layer projects from this shared representation.
        z_cond = self.z_mlp(z.to(torch.float32))  # [B, d_model]

        x = tgt
        if self.cfg.mask_rate > 0.0:
            # Masked bidirectional: no causal mask — every token attends to every other.
            for layer in self.layers:
                x = layer(x, z_cond, key_padding_mask=padding_mask)
        else:
            # Autoregressive: causal self-attention mask prevents attending to future tokens.
            causal_mask = self._causal_mask(tgt.shape[1], tgt.device)
            for layer in self.layers:
                x = layer(x, z_cond, attn_mask=causal_mask, key_padding_mask=padding_mask)

        decoded = self.final_norm(x)

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


__all__ = [
    "AdaLNModulation",
    "AdaLNTransformerLayer",
    "RouteVAEDecoderConfig",
    "RouteTransformerDecoder",
]
