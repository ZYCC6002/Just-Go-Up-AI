from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .route_vae_encoder import collate_hold_token_batch


FEATURE_KEYS = [
	"type_encoded_id",
	"function_encoded_id",
	"role_encoded_id",
	"hole_encoded_id",
	"x",
	"y",
	"depth",
	"orientation_sin",
	"orientation_cos",
	"size",
]


@dataclass(frozen=True)
class DecoderEOSIds:
	type_eos_id: int
	function_eos_id: int
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
		"function_target": _build_cat_target("function_encoded_id", eos_ids.function_eos_id),
		"role_target": _build_cat_target("role_encoded_id", eos_ids.role_eos_id),
		"hole_target": _build_cat_target("hole_encoded_id", eos_ids.hole_eos_id),
	}

	return decoder_input_batch, categorical_targets


def build_condition_tensors(
	samples: list[Any],
	*,
	device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Create angle/grade/grade_missing tensors from sample metadata.

	Each sample is expected to provide:
	- `angle` (float)
	- `grade` (float | None)
	"""
	angles = []
	grades = []
	grade_missing = []

	def _read(sample: Any, key: str, default: Any = None) -> Any:
		if isinstance(sample, dict):
			return sample.get(key, default)
		return getattr(sample, key, default)

	for sample in samples:
		grade_value = _read(sample, "grade", None)
		angles.append(float(_read(sample, "angle", 0.0)))
		if grade_value is None:
			grades.append(0.0)
			grade_missing.append(1.0)
		else:
			grades.append(float(grade_value))
			grade_missing.append(0.0)

	angle_t = torch.tensor(angles, dtype=torch.float32, device=device)
	grade_t = torch.tensor(grades, dtype=torch.float32, device=device)
	grade_missing_t = torch.tensor(grade_missing, dtype=torch.float32, device=device)
	return angle_t, grade_t, grade_missing_t


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

	angle, grade, grade_missing = build_condition_tensors(samples, device=device)

	batch_size, seq_len = target_batch["padding_mask"].shape

	def _shift_numeric_target(key: str) -> torch.Tensor:
		src = target_batch[key].to(torch.float32)
		tgt = torch.zeros((batch_size, seq_len + 1), dtype=torch.float32, device=src.device)
		tgt[:, :seq_len] = src
		return tgt

	numeric_targets = {
		"x_target": _shift_numeric_target("x"),
		"y_target": _shift_numeric_target("y"),
		"depth_target": _shift_numeric_target("depth"),
		"orientation_sin_target": _shift_numeric_target("orientation_sin"),
		"orientation_cos_target": _shift_numeric_target("orientation_cos"),
		"size_target": _shift_numeric_target("size"),
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
		"grade_missing": grade_missing,
		"valid_numeric_mask": valid_numeric_mask,
	}


__all__ = [
	"DecoderEOSIds",
	"FEATURE_KEYS",
	"build_condition_tensors",
	"prepare_cvae_training_batch",
	"prepare_teacher_forcing_batch",
]
