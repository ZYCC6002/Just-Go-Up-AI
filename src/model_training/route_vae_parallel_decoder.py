from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RouteVAEParallelDecoderConfig:
    """Configuration for the non-autoregressive parallel MLP decoder.

    Each output position is predicted independently from z + a learned position
    embedding.  No cross-token communication: z must carry all route structure.
    """

    # Feature vocab sizes (must match encoder)
    type_vocab_size: int
    role_vocab_size: int
    hole_id_vocab_size: int

    # Latent size (must match bottleneck)
    latent_dim: int

    # MLP architecture
    d_model: int = 64
    mlp_depth: int = 1          # hidden layers: each is Linear → GELU → LayerNorm → Dropout
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    max_seq_len: int = 128


class RouteParallelDecoder(nn.Module):
    """Non-autoregressive MLP decoder.

    Predicts every output position independently from z and a learned position
    embedding.  No cross-token information flows, so the model cannot chain tokens
    without z encoding the route structure.

    Output interface is identical to RouteTransformerDecoder — loss computation
    and downstream code require no changes.
    """

    is_autoregressive: bool = False

    def __init__(self, cfg: RouteVAEParallelDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # EOS ids (base vocab + 1 EOS slot each, same convention as transformer decoder)
        self.type_eos_id = cfg.type_vocab_size
        self.role_eos_id = cfg.role_vocab_size
        self.hole_eos_id = cfg.hole_id_vocab_size

        # Project z into d_model
        self.z_proj = nn.Linear(cfg.latent_dim, cfg.d_model)

        # Position embeddings — one per output slot
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # Shared MLP applied independently at every position
        layers: list[nn.Module] = []
        for _ in range(cfg.mlp_depth):
            layers.extend([
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps),
                nn.Dropout(cfg.dropout),
            ])
        self.mlp = nn.Sequential(*layers)

        # Output heads — identical to transformer decoder
        self.type_head = nn.Linear(cfg.d_model, cfg.type_vocab_size + 1)
        self.role_head = nn.Linear(cfg.d_model, cfg.role_vocab_size + 1)
        self.hole_head = nn.Linear(cfg.d_model, cfg.hole_id_vocab_size + 1)
        self.x_head = nn.Linear(cfg.d_model, 1)
        self.y_head = nn.Linear(cfg.d_model, 1)
        self.orientation_sin_head = nn.Linear(cfg.d_model, 1)
        self.orientation_cos_head = nn.Linear(cfg.d_model, 1)
        self.size_head = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Parallel prediction — all positions from z in a single pass.

        Args:
            batch: must contain "padding_mask" [B, L_out] to determine output
                   length.  Token feature keys are present but unused — there is
                   no input-token conditioning in a parallel decoder.
            z:     latent vector [B, latent_dim]
        """
        B, L_out = batch["padding_mask"].shape
        if L_out > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {L_out} exceeds max_seq_len={self.cfg.max_seq_len}."
            )

        pos_ids = torch.arange(L_out, device=z.device)                             # [L_out]
        pos_emb = self.pos_embedding(pos_ids)                                       # [L_out, d_model]
        z_exp   = self.z_proj(z.to(torch.float32)).unsqueeze(1).expand(B, L_out, -1)  # [B, L_out, d_model]
        hidden  = self.mlp(z_exp + pos_emb.unsqueeze(0))                           # [B, L_out, d_model]

        return {
            "type_logits":          self.type_head(hidden),
            "role_logits":          self.role_head(hidden),
            "hole_logits":          self.hole_head(hidden),
            "x_pred":               torch.sigmoid(self.x_head(hidden).squeeze(-1)),
            "y_pred":               torch.sigmoid(self.y_head(hidden).squeeze(-1)),
            "orientation_sin_pred": torch.sigmoid(self.orientation_sin_head(hidden).squeeze(-1)),
            "orientation_cos_pred": torch.sigmoid(self.orientation_cos_head(hidden).squeeze(-1)),
            "size_pred":            torch.sigmoid(self.size_head(hidden).squeeze(-1)),
        }


__all__ = ["RouteVAEParallelDecoderConfig", "RouteParallelDecoder"]
