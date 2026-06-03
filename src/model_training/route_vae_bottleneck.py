from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .route_vae_decoder import RouteTransformerDecoder
from .route_vae_encoder import RouteTransformerEncoder


# Listed angles (degrees) for product_id=1 (Kilter Board Original).
# Sourced from the products_angles table: SELECT angle FROM products_angles WHERE product_id=1.
KILTER_LISTED_ANGLES: list[float] = [0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.]


@dataclass
class RouteVAEBottleneckConfig:
    """Configuration for VAE bottleneck between encoder and decoder."""

    encoder_embedding_dim: int = 128
    latent_dim: int = 32
    hidden_dim: int = 128
    dropout: float = 0.1


class RouteVAEBottleneck(nn.Module):
    """Maps encoder route embeddings to latent distribution and sampled z."""

    def __init__(self, cfg: RouteVAEBottleneckConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.pre = nn.Sequential(
            nn.Linear(cfg.encoder_embedding_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.to_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.to_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, route_embedding: torch.Tensor, *, sample_latent: bool = True) -> dict[str, torch.Tensor]:
        h = self.pre(route_embedding)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)

        if sample_latent:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        return {"z": z, "mu": mu, "logvar": logvar}


def kl_divergence_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    kl_beta: float,
    reduction: Literal["mean", "sum", "none"] = "mean",
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL divergence to N(0, I) for diagonal Gaussian posterior with free-bits.

    Per-sample KL:
      KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Free-bits uses a single piecewise-linear term per dimension:
      loss_i = kl_beta * KL_i + relu(free_bits - KL_i)

    Gradient direction w.r.t. KL_i:
      KL_i < free_bits  →  net slope (kl_beta - 1) < 0  →  gradient INCREASES KL_i (escapes collapse)
      KL_i ≥ free_bits  →  net slope kl_beta > 0         →  gradient DECREASES KL_i (standard β-VAE)

    The equilibrium is at KL_i = free_bits for every dimension.

    Returns:
        (unified_kl, raw_kl): two tensors reduced by `reduction`.
          unified_kl: loss contribution — add to total with weight 1.0, no external scaling needed.
          raw_kl:     unweighted per-sample KL, for logging / kl_excess diagnostics only.

    Args:
        mu:        Posterior mean [B, D].
        logvar:    Posterior log-variance [B, D].
        kl_beta:   β-VAE weighting applied inside the unified term.
        reduction: How to reduce over the batch dimension.
        free_bits: Per-dimension KL floor (nats). Recommended: 0.5–1.0.
    """
    # [B, D]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    raw_per_sample = kl_per_dim.sum(dim=-1)  # [B], diagnostics only

    # Unified term — kl_beta and free-bits handled together.
    if free_bits > 0.0:
        unified_per_sample = (kl_beta * kl_per_dim + torch.relu(free_bits - kl_per_dim)).sum(dim=-1)
    else:
        unified_per_sample = kl_beta * raw_per_sample

    if reduction == "none":
        return unified_per_sample, raw_per_sample
    if reduction == "sum":
        return unified_per_sample.sum(), raw_per_sample.sum()
    return unified_per_sample.mean(), raw_per_sample.mean()


class RouteConditionalVAE(nn.Module):
    """Bridge module: encoder -> bottleneck -> decoder."""

    def __init__(
        self,
        encoder: RouteTransformerEncoder,
        bottleneck: RouteVAEBottleneck,
        decoder: RouteTransformerDecoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(
        self,
        *,
        encoder_batch: dict[str, torch.Tensor],
        decoder_batch: dict[str, torch.Tensor],
        sample_latent: bool = True,
    ) -> dict[str, torch.Tensor]:
        enc_out = self.encoder(encoder_batch)
        bottleneck_out = self.bottleneck(enc_out["route_embedding"], sample_latent=sample_latent)
        dec_out = self.decoder(decoder_batch, z=bottleneck_out["z"])

        return {
            "encoder_token_embeddings": enc_out["token_embeddings"],
            "encoder_route_embedding": enc_out["route_embedding"],
            "z": bottleneck_out["z"],
            "mu": bottleneck_out["mu"],
            "logvar": bottleneck_out["logvar"],
            **dec_out,
        }


__all__ = [
    "KILTER_LISTED_ANGLES",
    "RouteVAEBottleneckConfig",
    "RouteVAEBottleneck",
    "RouteConditionalVAE",
    "kl_divergence_loss",
]
