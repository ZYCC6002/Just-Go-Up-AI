from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .route_vae_decoder import RouteTransformerDecoder
from .route_vae_encoder import RouteTransformerEncoder


class _GradientReversal(torch.autograd.Function):
    """Identity in the forward pass; negates and scales gradient in the backward pass."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (alpha,) = ctx.saved_tensors
        return -alpha * grad_output, None


# Listed angles (degrees) for product_id=1 (Kilter Board Original).
# Sourced from the products_angles table: SELECT angle FROM products_angles WHERE product_id=1.
KILTER_LISTED_ANGLES: list[float] = [0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.]


class PerAngleGradeAdversaryHead(nn.Module):
    """Predicts normalised grade at every listed wall angle from z via a GRL.

    Replaces the old GradeAngleAdversaryHead which predicted a single (grade,
    angle) pair.  Key changes:
    - Output shape [B, num_listed_angles] instead of [B, 2].
    - Predicts grade only (not angle): since the encoder never sees angle as
      input, angle is structurally absent from z — predicting it is redundant.
    - Supplies up to 15× more supervision per route, applying much stronger
      pressure to push grade information out of z across the full angle range.

    Loss is computed only over angles with known grade data (NaN elsewhere),
    so routes with fewer angle variants contribute less signal per step but are
    not penalised.

    Args:
        latent_dim:         Dimension of z.
        num_listed_angles:  Number of output heads (one per listed angle).
                            Default 15 matches Kilter Board Original.
        hidden_dim:         Hidden layer width.
    """

    def __init__(
        self,
        latent_dim: int,
        num_listed_angles: int = len(KILTER_LISTED_ANGLES),
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_listed_angles = num_listed_angles
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_listed_angles),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Args:
            z:     latent vector [B, latent_dim]
            alpha: GRL scale factor (ramp during training to stabilise early epochs)
        Returns:
            [B, num_listed_angles] predicted grades in [0, 1]
        """
        alpha_t = torch.tensor(alpha, dtype=z.dtype, device=z.device)
        z_r = _GradientReversal.apply(z, alpha_t)
        return self.mlp(z_r)


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
	free_bits: float = 0.0,
) -> torch.Tensor:
	"""KL divergence to N(0, I) for diagonal Gaussian posterior.

	Per sample KL:
	  KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

	Args:
		mu:        Mean of the posterior [B, D].
		logvar:    Log-variance of the posterior [B, D].
		reduction: How to reduce over the batch dimension.
		free_bits: Minimum KL per latent dimension (nats). When > 0, each
		           dimension's KL is clamped from below at this threshold before
		           summing. This prevents any dimension from fully collapsing to
		           the prior and forces the encoder to keep every z dimension
		           active. Recommended value: 0.5 nats/dim.
	"""
	# Per-sample, per-dimension KL: [B, D]
	kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

	if free_bits > 0.0:
		# Clamp each dimension's KL at the free-bits floor before summing.
		# Dimensions that try to collapse below `free_bits` nats are forced
		# to stay active; dimensions above the floor are unaffected.
		kl_per_dim = kl_per_dim.clamp(min=free_bits)

	kl_per_sample = kl_per_dim.sum(dim=-1)

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
	"KILTER_LISTED_ANGLES",
	"PerAngleGradeAdversaryHead",
	"RouteVAEBottleneckConfig",
	"RouteVAEBottleneck",
	"RouteConditionalVAE",
	"kl_divergence_loss",
]
