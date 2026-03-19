from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class RouteTransformerConfig:
	"""Configuration for hold-token transformer encoder."""

	# Feature vocab sizes (set based on your preprocessing dictionaries)
	type_vocab_size: int
	function_vocab_size: int
	role_vocab_size: int
	hole_id_vocab_size: int

	# Per-feature embedding sizes
	type_embed_dim: int = 16
	function_embed_dim: int = 8
	role_embed_dim: int = 8
	hole_id_embed_dim: int = 16
	x_embed_dim: int = 16
	y_embed_dim: int = 16
	depth_embed_dim: int = 8
	orientation_sin_embed_dim: int = 8
	orientation_cos_embed_dim: int = 8
	size_embed_dim: int = 8

	# Transformer sizes
	d_model: int = 128
	nhead: int = 8
	num_layers: int = 4
	dim_feedforward: int = 256
	dropout: float = 0.1
	layer_norm_eps: float = 1e-5


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
			values: Tensor of shape [B, L] or [N] with scalar coordinates.

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


class RouteTransformerEncoder(nn.Module):
	"""Transformer encoder over hold tokens with per-feature embeddings.

	Expected token-level inputs (each [B, L], except padding_mask):
	- type_id: categorical hold type index
	- function_id: categorical hold function index
	- role_id: placement role index
	- hole_id: categorical hole index
	- x: board x coordinate (sinusoidal embedding)
	- y: board y coordinate (sinusoidal embedding)
	- depth: numeric
	- orientation_sin: numeric (sin(theta))
	- orientation_cos: numeric (cos(theta))
	- size: numeric
	- padding_mask: bool [B, L], True for padded tokens
	"""

	def __init__(self, cfg: RouteTransformerConfig) -> None:
		super().__init__()
		self.cfg = cfg

		# Categorical feature embeddings
		self.type_embedding = nn.Embedding(cfg.type_vocab_size, cfg.type_embed_dim)
		self.function_embedding = nn.Embedding(cfg.function_vocab_size, cfg.function_embed_dim)
		self.role_embedding = nn.Embedding(cfg.role_vocab_size, cfg.role_embed_dim)
		self.hole_embedding = nn.Embedding(cfg.hole_id_vocab_size, cfg.hole_id_embed_dim)

		# Positional scalar sinusoidal embeddings
		self.x_embedding = ScalarSinusoidalEmbedding(cfg.x_embed_dim)
		self.y_embedding = ScalarSinusoidalEmbedding(cfg.y_embed_dim)

		# Numeric feature projections (scalar -> vector)
		self.depth_projection = nn.Linear(1, cfg.depth_embed_dim)
		self.orientation_sin_projection = nn.Linear(1, cfg.orientation_sin_embed_dim)
		self.orientation_cos_projection = nn.Linear(1, cfg.orientation_cos_embed_dim)
		self.size_projection = nn.Linear(1, cfg.size_embed_dim)

		token_input_dim = (
			cfg.type_embed_dim
			+ cfg.function_embed_dim
			+ cfg.role_embed_dim
			+ cfg.hole_id_embed_dim
			+ cfg.x_embed_dim
			+ cfg.y_embed_dim
			+ cfg.depth_embed_dim
			+ cfg.orientation_sin_embed_dim
			+ cfg.orientation_cos_embed_dim
			+ cfg.size_embed_dim
		)

		self.token_projection = nn.Sequential(
			nn.Linear(token_input_dim, cfg.d_model),
			nn.GELU(),
			nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps),
			nn.Dropout(cfg.dropout),
		)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=cfg.d_model,
			nhead=cfg.nhead,
			dim_feedforward=cfg.dim_feedforward,
			dropout=cfg.dropout,
			activation="gelu",
			layer_norm_eps=cfg.layer_norm_eps,
			batch_first=True,
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
		self.final_norm = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

	def _project_numeric(self, tensor_2d: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
		return proj(tensor_2d.unsqueeze(-1).to(torch.float32))

	@staticmethod
	def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
		# depth range: [0, 3]
		return (depth.to(torch.float32) / 3.0).clamp(0.0, 1.0)

	@staticmethod
	def _normalize_size(size: torch.Tensor) -> torch.Tensor:
		# size range: [2, 5]
		return ((size.to(torch.float32) - 2.0) / 3.0).clamp(0.0, 1.0)

	def build_token_embeddings(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
		type_emb = self.type_embedding(batch["type_id"])  # [B, L, d_type]
		function_emb = self.function_embedding(batch["function_id"])  # [B, L, d_func]
		role_emb = self.role_embedding(batch["role_id"])  # [B, L, d_role]
		hole_emb = self.hole_embedding(batch["hole_id"])  # [B, L, d_hole]

		x_emb = self.x_embedding(batch["x"].to(torch.float32))  # [B, L, d_x]
		y_emb = self.y_embedding(batch["y"].to(torch.float32))  # [B, L, d_y]

		depth_norm = self._normalize_depth(batch["depth"])
		size_norm = self._normalize_size(batch["size"])

		depth_emb = self._project_numeric(depth_norm, self.depth_projection)
		orientation_sin_emb = self._project_numeric(batch["orientation_sin"], self.orientation_sin_projection)
		orientation_cos_emb = self._project_numeric(batch["orientation_cos"], self.orientation_cos_projection)
		size_emb = self._project_numeric(size_norm, self.size_projection)

		# "embed all features, then concatenate embeddings"
		token_concat = torch.cat(
			[
				type_emb,
				function_emb,
				role_emb,
				hole_emb,
				x_emb,
				y_emb,
				depth_emb,
				orientation_sin_emb,
				orientation_cos_emb,
				size_emb,
			],
			dim=-1,
		)
		return self.token_projection(token_concat)

	@staticmethod
	def masked_mean_pool(token_embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
		"""Mean pool over non-padded tokens.

		Args:
			token_embeddings: [B, L, D]
			padding_mask: [B, L] where True means padded token
		"""
		valid_mask = (~padding_mask).unsqueeze(-1).to(token_embeddings.dtype)  # [B, L, 1]
		summed = (token_embeddings * valid_mask).sum(dim=1)  # [B, D]
		counts = valid_mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
		return summed / counts

	def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
		"""Encode hold-token sequences.

		Returns:
			{
			  "token_embeddings": [B, L, d_model],
			  "route_embedding": [B, d_model],  # mean pooled
			}
		"""
		padding_mask = batch["padding_mask"].bool()
		tokens = self.build_token_embeddings(batch)
		encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
		encoded = self.final_norm(encoded)
		route_embedding = self.masked_mean_pool(encoded, padding_mask)
		return {"token_embeddings": encoded, "route_embedding": route_embedding}


def collate_hold_token_batch(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
	"""Pad variable-length hold-token samples into a batch.

	Each sample should be a dict with keys:
	- type_id, function_id, role_id, hole_id, x, y, depth, orientation_sin, orientation_cos, size
	Each value is a 1D tensor of length L_i.
	"""
	if not samples:
		raise ValueError("samples must be non-empty")

	feature_keys = [
		"type_id",
		"function_id",
		"role_id",
		"hole_id",
		"x",
		"y",
		"depth",
		"orientation_sin",
		"orientation_cos",
		"size",
	]

	lengths = [int(sample["type_id"].shape[0]) for sample in samples]
	max_len = max(lengths)
	batch_size = len(samples)

	def _pad_1d(tensor: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
		if tensor.shape[0] == target_len:
			return tensor
		pad = torch.full((target_len - tensor.shape[0],), pad_value, dtype=tensor.dtype, device=tensor.device)
		return torch.cat([tensor, pad], dim=0)

	batch: dict[str, torch.Tensor] = {}
	for key in feature_keys:
		padded = []
		for sample in samples:
			t = sample[key]
			pad_value = 0 if key in {"type_id", "function_id", "role_id", "hole_id"} else 0.0
			padded.append(_pad_1d(t, max_len, pad_value))
		batch[key] = torch.stack(padded, dim=0)

	padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
	for i, length in enumerate(lengths):
		padding_mask[i, :length] = False
	batch["padding_mask"] = padding_mask

	# enforce dtypes
	batch["type_id"] = batch["type_id"].long()
	batch["function_id"] = batch["function_id"].long()
	batch["role_id"] = batch["role_id"].long()
	batch["hole_id"] = batch["hole_id"].long()
	for key in ["x", "y", "depth", "orientation_sin", "orientation_cos", "size"]:
		batch[key] = batch[key].to(torch.float32)

	return batch


__all__ = [
	"RouteTransformerConfig",
	"ScalarSinusoidalEmbedding",
	"RouteTransformerEncoder",
	"collate_hold_token_batch",
]
