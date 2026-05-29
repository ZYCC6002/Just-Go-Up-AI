from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .route_vae_encoder import collate_hold_token_batch


FEATURE_KEYS = [
	"type_encoded_id",
	"role_encoded_id",
	"hole_encoded_id",
	"x",
	"y",
	"orientation_sin",
	"orientation_cos",
	"size",
	# depth excluded: physical hold property, not reconstructed by decoder
	# delta_x_prev / delta_y_prev excluded: derivable from predicted x/y; teacher-forcing
	# with ground-truth delta creates a train/inference mismatch
	# knn_features excluded: requires full sequence, unavailable during autoregressive generation
]


@dataclass(frozen=True)
class DecoderEOSIds:
	type_eos_id: int
	role_eos_id: int
	hole_eos_id: int



def prepare_teacher_forcing_batch(
	target_batch: dict[str, torch.Tensor],
	*,
	eos_ids: DecoderEOSIds,
	ignore_index: int = -100,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
	"""Build BOS-prepended decoder inputs and EOS-terminated categorical targets.

	Args:
		target_batch: Unshifted ground-truth tokens with `padding_mask` [B, L].
		eos_ids: EOS class ids for each categorical head.
		ignore_index: Label index used to ignore padded positions in CE loss.

	Returns:
		(decoder_input_batch, categorical_targets)
		- decoder_input_batch has length L+1 with BOS anchor at timestep 0.
		- categorical targets are length L+1 and include EOS at each sequence end.
	"""
	padding_mask = target_batch["padding_mask"].bool()
	batch_size, seq_len = padding_mask.shape

	decoder_input_batch: dict[str, torch.Tensor] = {}
	for key in FEATURE_KEYS:
		v = target_batch[key]
		shifted = torch.zeros(
			(batch_size, seq_len + 1),
			dtype=v.dtype,
			device=v.device,
		)
		shifted[:, 1:] = v
		decoder_input_batch[key] = shifted

	decoder_padding_mask = torch.ones(
		(batch_size, seq_len + 1),
		dtype=torch.bool,
		device=padding_mask.device,
	)
	decoder_padding_mask[:, 0] = False
	decoder_padding_mask[:, 1:] = padding_mask
	decoder_input_batch["padding_mask"] = decoder_padding_mask

	valid_lengths = (~padding_mask).sum(dim=1)
	batch_indices = torch.arange(batch_size, device=padding_mask.device)

	def _build_cat_target(key: str, eos_id: int) -> torch.Tensor:
		src = target_batch[key].long()
		tgt = torch.full(
			(batch_size, seq_len + 1),
			fill_value=ignore_index,
			dtype=torch.long,
			device=src.device,
		)
		tgt[:, :seq_len] = src
		tgt[:, :seq_len] = tgt[:, :seq_len].masked_fill(padding_mask, ignore_index)
		tgt[batch_indices, valid_lengths] = eos_id
		return tgt

	categorical_targets = {
		"type_target": _build_cat_target("type_encoded_id", eos_ids.type_eos_id),
		"role_target": _build_cat_target("role_encoded_id", eos_ids.role_eos_id),
		"hole_target": _build_cat_target("hole_encoded_id", eos_ids.hole_eos_id),
	}

	return decoder_input_batch, categorical_targets


def build_condition_tensors(
	samples: list[Any],
	*,
	device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Create angle and grade tensors from sample metadata."""
	angles = []
	grades = []

	def _read(sample: Any, key: str, default: Any = None) -> Any:
		if isinstance(sample, dict):
			return sample.get(key, default)
		return getattr(sample, key, default)

	for sample in samples:
		angles.append(float(_read(sample, "angle", 0.0)))
		grades.append(float(_read(sample, "grade", 0.0)))

	angle_t = torch.tensor(angles, dtype=torch.float32, device=device)
	grade_t = torch.tensor(grades, dtype=torch.float32, device=device)
	return angle_t, grade_t


def prepare_cvae_training_batch(
	samples: list[Any],
	*,
	eos_ids: DecoderEOSIds,
	device: torch.device | None = None,
	ignore_index: int = -100,
) -> dict[str, Any]:
	"""Prepare all tensors needed by the CVAE training step.

	Expected sample schema:
	- sample["tokens"]: token dict used by `collate_hold_token_batch`
	- sample["angle"]: float
	- sample["grade"]: float | None
	"""
	if not samples:
		raise ValueError("samples must be non-empty")

	def _tokens(sample: Any) -> dict[str, torch.Tensor]:
		if isinstance(sample, dict):
			return sample["tokens"]
		return getattr(sample, "tokens")

	token_samples = [_tokens(sample) for sample in samples]
	target_batch = collate_hold_token_batch(token_samples)
	if device is not None:
		target_batch = {k: v.to(device) for k, v in target_batch.items()}

	decoder_input_batch, categorical_targets = prepare_teacher_forcing_batch(
		target_batch,
		eos_ids=eos_ids,
		ignore_index=ignore_index,
	)

	angle, grade = build_condition_tensors(samples, device=device)

	batch_size, seq_len = target_batch["padding_mask"].shape

	def _shift_numeric_target(key: str, vmin: float = 0.0, vmax: float = 1.0) -> torch.Tensor:
		src = target_batch[key].to(torch.float32)
		src = (src - vmin) / max(vmax - vmin, 1e-6)
		src = src.clamp(0.0, 1.0)
		tgt = torch.zeros((batch_size, seq_len + 1), dtype=torch.float32, device=src.device)
		tgt[:, :seq_len] = src
		return tgt

	numeric_targets = {
		"x_target": _shift_numeric_target("x", 0.0, 140.0),
		"y_target": _shift_numeric_target("y", 0.0, 160.0),
		"orientation_sin_target": _shift_numeric_target("orientation_sin", -1.0, 1.0),
		"orientation_cos_target": _shift_numeric_target("orientation_cos", -1.0, 1.0),
		"size_target": _shift_numeric_target("size", 2.0, 5.0),
	}

	# Numeric reconstruction should be evaluated on real token targets only:
	# include positions predicting actual holds, exclude EOS and ignored pads.
	type_target = categorical_targets["type_target"]
	valid_numeric_mask = (type_target != ignore_index) & (type_target != eos_ids.type_eos_id)

	return {
		"encoder_batch": target_batch,
		"target_batch": target_batch,
		"decoder_input_batch": decoder_input_batch,
		"categorical_targets": categorical_targets,
		"numeric_targets": numeric_targets,
		"angle": angle,
		"grade": grade,
		"valid_numeric_mask": valid_numeric_mask,
	}


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	"""MSE loss over positions selected by a boolean mask."""
	if pred.shape != target.shape:
		raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
	mask_f = mask.to(torch.float32)
	denom = mask_f.sum().clamp(min=1.0)
	return (((pred - target) ** 2) * mask_f).sum() / denom


__all__ = [
	"DecoderEOSIds",
	"FEATURE_KEYS",
	"build_condition_tensors",
	"masked_mse",
	"prepare_cvae_training_batch",
	"prepare_teacher_forcing_batch",
]
