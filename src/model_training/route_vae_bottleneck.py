from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .route_vae_decoder import RouteTransformerDecoder
from .route_vae_encoder import RouteTransformerEncoder


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
	reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
	"""KL divergence to N(0, I) for diagonal Gaussian posterior.

	Per sample KL:
	  KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
	"""
	kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

	if reduction == "none":
		return kl_per_sample
	if reduction == "sum":
		return kl_per_sample.sum()
	return kl_per_sample.mean()


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
		angle: torch.Tensor,
		grade: torch.Tensor,
		grade_missing: torch.Tensor | None = None,
		sample_latent: bool = True,
	) -> dict[str, torch.Tensor]:
		enc_out = self.encoder(
			encoder_batch,
			angle=angle,
			grade=grade,
			grade_missing=grade_missing,
		)

		bottleneck_out = self.bottleneck(enc_out["route_embedding"], sample_latent=sample_latent)

		dec_out = self.decoder(
			decoder_batch,
			z=bottleneck_out["z"],
			angle=angle,
			grade=grade,
			grade_missing=grade_missing,
		)

		return {
			"encoder_token_embeddings": enc_out["token_embeddings"],
			"encoder_route_embedding": enc_out["route_embedding"],
			"z": bottleneck_out["z"],
			"mu": bottleneck_out["mu"],
			"logvar": bottleneck_out["logvar"],
			**dec_out,
		}


__all__ = [
	"RouteVAEBottleneckConfig",
	"RouteVAEBottleneck",
	"RouteConditionalVAE",
	"kl_divergence_loss",
]
