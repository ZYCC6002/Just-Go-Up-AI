from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .route_vae_encoder import ScalarSinusoidalEmbedding


@dataclass
class RouteVAEDecoderConfig:
	"""Configuration for autoregressive route decoder."""

	# Feature vocab sizes
	type_vocab_size: int
	function_vocab_size: int
	role_vocab_size: int
	hole_id_vocab_size: int

	# Latent/conditioning sizes
	latent_dim: int
	condition_hidden_dim: int = 64

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

	# Decoder transformer sizes
	d_model: int = 128
	nhead: int = 8
	num_layers: int = 4
	dim_feedforward: int = 256
	dropout: float = 0.1
	layer_norm_eps: float = 1e-5
	max_seq_len: int = 128


class RouteTransformerDecoder(nn.Module):
	"""Autoregressive transformer decoder conditioned on latent, angle, and grade.

	Expected teacher-forcing input keys:
	- type_encoded_id, function_encoded_id, role_encoded_id, hole_encoded_id
	- x, y, depth, orientation_sin, orientation_cos, size
	- padding_mask (bool, True for padded tokens)
	"""

	def __init__(self, cfg: RouteVAEDecoderConfig) -> None:
		super().__init__()
		self.cfg = cfg

		# Categorical input embeddings
		self.type_embedding = nn.Embedding(cfg.type_vocab_size, cfg.type_embed_dim)
		self.function_embedding = nn.Embedding(cfg.function_vocab_size, cfg.function_embed_dim)
		self.role_embedding = nn.Embedding(cfg.role_vocab_size, cfg.role_embed_dim)
		self.hole_embedding = nn.Embedding(cfg.hole_id_vocab_size, cfg.hole_id_embed_dim)

		# Scalar positional embeddings for x/y
		self.x_embedding = ScalarSinusoidalEmbedding(cfg.x_embed_dim)
		self.y_embedding = ScalarSinusoidalEmbedding(cfg.y_embed_dim)

		# Scalar projections
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

		# Learned sequence position embedding for decoder time-step order
		self.sequence_position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

		# Conditioning memory tokens
		self.latent_to_memory = nn.Linear(cfg.latent_dim, cfg.d_model)
		self.cond_mlp = nn.Sequential(
			nn.Linear(3, cfg.condition_hidden_dim),  # [angle, grade, grade_missing]
			nn.GELU(),
			nn.Linear(cfg.condition_hidden_dim, cfg.d_model),
		)

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

		# Output heads (next-token predictions)
		self.type_head = nn.Linear(cfg.d_model, cfg.type_vocab_size)
		self.function_head = nn.Linear(cfg.d_model, cfg.function_vocab_size)
		self.role_head = nn.Linear(cfg.d_model, cfg.role_vocab_size)
		self.hole_head = nn.Linear(cfg.d_model, cfg.hole_id_vocab_size)

		self.x_head = nn.Linear(cfg.d_model, 1)
		self.y_head = nn.Linear(cfg.d_model, 1)
		self.depth_head = nn.Linear(cfg.d_model, 1)
		self.orientation_sin_head = nn.Linear(cfg.d_model, 1)
		self.orientation_cos_head = nn.Linear(cfg.d_model, 1)
		self.size_head = nn.Linear(cfg.d_model, 1)

	def _project_numeric(self, tensor_2d: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
		return proj(tensor_2d.unsqueeze(-1).to(torch.float32))

	@staticmethod
	def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
		return (depth.to(torch.float32) / 3.0).clamp(0.0, 1.0)

	@staticmethod
	def _normalize_size(size: torch.Tensor) -> torch.Tensor:
		return ((size.to(torch.float32) - 2.0) / 3.0).clamp(0.0, 1.0)

	@staticmethod
	def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
		# Shape [L, L], -inf above diagonal blocks future positions.
		mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
		return torch.triu(mask, diagonal=1)

	def _build_memory(
		self,
		z: torch.Tensor,
		angle: torch.Tensor,
		grade: torch.Tensor,
		grade_missing: torch.Tensor | None,
	) -> torch.Tensor:
		"""Create conditioning memory tokens for decoder cross-attention.

		Returns memory shape [B, 2, d_model]:
		- token 0: latent token
		- token 1: route condition token (angle/grade)
		"""
		if grade_missing is None:
			grade_missing = torch.zeros_like(grade, dtype=torch.float32)
		else:
			grade_missing = grade_missing.to(torch.float32)

		latent_token = self.latent_to_memory(z).unsqueeze(1)
		cond_input = torch.stack(
			[
				angle.to(torch.float32),
				grade.to(torch.float32),
				grade_missing,
			],
			dim=-1,
		)
		cond_token = self.cond_mlp(cond_input).unsqueeze(1)
		return torch.cat([latent_token, cond_token], dim=1)

	def build_decoder_inputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
		type_emb = self.type_embedding(batch["type_encoded_id"])
		function_emb = self.function_embedding(batch["function_encoded_id"])
		role_emb = self.role_embedding(batch["role_encoded_id"])
		hole_emb = self.hole_embedding(batch["hole_encoded_id"])

		x_emb = self.x_embedding(batch["x"].to(torch.float32))
		y_emb = self.y_embedding(batch["y"].to(torch.float32))

		depth_norm = self._normalize_depth(batch["depth"])
		size_norm = self._normalize_size(batch["size"])

		depth_emb = self._project_numeric(depth_norm, self.depth_projection)
		orientation_sin_emb = self._project_numeric(batch["orientation_sin"], self.orientation_sin_projection)
		orientation_cos_emb = self._project_numeric(batch["orientation_cos"], self.orientation_cos_projection)
		size_emb = self._project_numeric(size_norm, self.size_projection)

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

		tokens = self.token_projection(token_concat)

		# Add learned time-step embedding
		batch_size, seq_len, _ = tokens.shape
		if seq_len > self.cfg.max_seq_len:
			raise ValueError(
				f"Sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}."
			)
		positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
		tokens = tokens + self.sequence_position_embedding(positions)
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
			batch: tokenized decoder inputs (usually shifted-right targets)
			z: latent vector [B, latent_dim]
			angle: route angle condition [B]
			grade: route grade condition [B]
			grade_missing: optional mask [B] (1 if grade missing)
		"""
		padding_mask = batch["padding_mask"].bool()
		tgt = self.build_decoder_inputs(batch)
		memory = self._build_memory(z=z, angle=angle, grade=grade, grade_missing=grade_missing)

		seq_len = tgt.shape[1]
		tgt_mask = self._causal_mask(seq_len=seq_len, device=tgt.device)

		decoded = self.decoder(
			tgt=tgt,
			memory=memory,
			tgt_mask=tgt_mask,
			tgt_key_padding_mask=padding_mask,
		)
		decoded = self.final_norm(decoded)

		return {
			"hidden_states": decoded,
			"type_logits": self.type_head(decoded),
			"function_logits": self.function_head(decoded),
			"role_logits": self.role_head(decoded),
			"hole_logits": self.hole_head(decoded),
			"x_pred": self.x_head(decoded).squeeze(-1),
			"y_pred": self.y_head(decoded).squeeze(-1),
			"depth_pred": self.depth_head(decoded).squeeze(-1),
			"orientation_sin_pred": self.orientation_sin_head(decoded).squeeze(-1),
			"orientation_cos_pred": self.orientation_cos_head(decoded).squeeze(-1),
			"size_pred": self.size_head(decoded).squeeze(-1),
		}


__all__ = ["RouteVAEDecoderConfig", "RouteTransformerDecoder"]
