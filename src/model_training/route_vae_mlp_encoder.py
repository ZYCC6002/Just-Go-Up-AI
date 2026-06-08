"""MLP encoder baseline for the route VAE.

Maps the pre-computed 9D shape_desc vector directly to route_embedding via a
shallow MLP — no per-hold token processing.  Serves as a fast baseline to
measure how much reconstruction loss is achievable from hand-crafted aggregate
features alone, without any transformer-based hold-sequence reasoning.

Use with ``--mlp-encoder`` in ``train_route_cvae.py``.  The output dict is
interface-compatible with ``RouteTransformerEncoder``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RouteMlpEncoderConfig:
    """Configuration for the MLP encoder baseline.

    Maps shape_desc (pre-computed route statistics) directly to route_embedding
    via a shallow MLP.  No per-hold processing; serves as a fast baseline to
    measure how much reconstruction loss is achievable from hand-crafted features.
    """
    shape_desc_dim: int = 9
    d_model: int = 256           # output width — must match bottleneck encoder_embedding_dim
    num_layers: int = 3          # total layers (num_layers-1 hidden + 1 output)
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


class RouteMlpEncoder(nn.Module):
    """MLP encoder that maps shape_desc → route_embedding.

    Baseline alternative to RouteTransformerEncoder.  Takes the 9D pre-computed
    route descriptor (x_std, y_std, foot_frac, ...) as input instead of per-hold
    tokens.  Output dict is compatible with RouteConditionalVAE.
    """

    def __init__(self, cfg: RouteMlpEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        in_dim = cfg.shape_desc_dim
        for _ in range(cfg.num_layers - 1):
            layers += [
                nn.Linear(in_dim, cfg.d_model),
                nn.GELU(),
                nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps),
                nn.Dropout(cfg.dropout),
            ]
            in_dim = cfg.d_model
        layers.append(nn.Linear(in_dim, cfg.d_model))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        route_embedding = self.mlp(batch["shape_desc"].to(torch.float32))  # [B, d_model]
        return {
            "route_embedding": route_embedding,
            # No per-token outputs; return empty tensor with correct d_model for compatibility.
            "token_embeddings": torch.zeros(
                route_embedding.shape[0], 0, route_embedding.shape[1],
                device=route_embedding.device, dtype=route_embedding.dtype,
            ),
        }


__all__ = ["RouteMlpEncoderConfig", "RouteMlpEncoder"]
