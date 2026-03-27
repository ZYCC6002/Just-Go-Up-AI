from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from data_preprocessing.route_preprocessing import build_training_samples_from_db, split_route_samples
from model_training.route_vae_bottleneck import (
	RouteConditionalVAE,
	RouteVAEBottleneck,
	RouteVAEBottleneckConfig,
	kl_divergence_loss,
)
from model_training.route_vae_decoder import RouteTransformerDecoder, RouteVAEDecoderConfig
from model_training.route_vae_encoder import RouteTransformerEncoder
from model_training.training_utils import DecoderEOSIds, prepare_cvae_training_batch


@dataclass
class EpochMetrics:
	total_loss: float
	categorical_loss: float
	numeric_loss: float
	kl_loss: float
	num_batches: int


def _compute_kl_beta_for_epoch(*, epoch: int, target_kl_beta: float, kl_warmup_epochs: int) -> float:
	"""Linear KL annealing from 0 -> target_kl_beta over warmup epochs."""
	if kl_warmup_epochs <= 0:
		return float(target_kl_beta)
	ratio = min(max(epoch, 0), kl_warmup_epochs) / float(kl_warmup_epochs)
	return float(target_kl_beta) * ratio


def _save_loss_curve_plot(
	*,
	epochs: list[int],
	train_total_losses: list[float],
	val_total_losses: list[float],
	output_path: Path,
) -> None:
	if not epochs:
		return

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig = plt.figure(figsize=(9, 5))
	ax = fig.add_subplot(111)
	ax.plot(epochs, train_total_losses, label="Train Total Loss", linewidth=2)
	ax.plot(epochs, val_total_losses, label="Val Total Loss", linewidth=2)
	ax.set_title("Training Loss per Epoch")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.grid(alpha=0.3)
	ax.legend(loc="best")
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)
	print(f"Saved loss curve plot: {output_path}")



def _iter_minibatches(samples: list[Any], batch_size: int, shuffle=True):
	indices = list(range(len(samples)))
	if shuffle:
		random.shuffle(indices)
	for i in range(0, len(indices), batch_size):
		yield [samples[j] for j in indices[i:i + batch_size]]


def _sample_token_length(sample: Any) -> int:
	if isinstance(sample, dict):
		if "num_holds" in sample:
			return int(sample["num_holds"])
		return int(sample["tokens"]["type_encoded_id"].shape[0])
	if hasattr(sample, "num_holds"):
		return int(getattr(sample, "num_holds"))
	return int(getattr(sample, "tokens")["type_encoded_id"].shape[0])


def _filter_samples_by_decoder_max_len(samples: list[Any], *, max_seq_len: int) -> tuple[list[Any], int]:
	"""Drop samples that cannot fit in decoder sequence length.

	Decoder input length is target length + 1 due to BOS.
	Therefore valid target length is <= max_seq_len - 1.
	"""
	max_target_len = max_seq_len - 1
	kept: list[Any] = []
	skipped = 0
	for sample in samples:
		if _sample_token_length(sample) <= max_target_len:
			kept.append(sample)
		else:
			skipped += 1
	return kept, skipped



def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	if pred.shape != target.shape:
		raise ValueError(f"Shape mismatch for MSE: pred={pred.shape}, target={target.shape}")
	mask_f = mask.to(torch.float32)
	denom = mask_f.sum().clamp(min=1.0)
	return (((pred - target) ** 2) * mask_f).sum() / denom



def _compute_batch_losses(
	model: RouteConditionalVAE,
	batch_samples: list[Any],
	*,
	device: torch.device,
	eos_ids: DecoderEOSIds,
	kl_beta: float,
	numeric_weight: float,
	ignore_index: int = -100,
	sample_latent: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
	prepared = prepare_cvae_training_batch(
		batch_samples,
		eos_ids=eos_ids,
		device=device,
		ignore_index=ignore_index,
	)

	out = model(
		encoder_batch=prepared["encoder_batch"],
		decoder_batch=prepared["decoder_input_batch"],
		angle=prepared["angle"],
		grade=prepared["grade"],
		grade_missing=prepared["grade_missing"],
		sample_latent=sample_latent,
	)

	categorical_targets = prepared["categorical_targets"]
	cat_type = F.cross_entropy(
		out["type_logits"].reshape(-1, out["type_logits"].shape[-1]),
		categorical_targets["type_target"].reshape(-1),
		ignore_index=ignore_index,
	)
	cat_function = F.cross_entropy(
		out["function_logits"].reshape(-1, out["function_logits"].shape[-1]),
		categorical_targets["function_target"].reshape(-1),
		ignore_index=ignore_index,
	)
	cat_role = F.cross_entropy(
		out["role_logits"].reshape(-1, out["role_logits"].shape[-1]),
		categorical_targets["role_target"].reshape(-1),
		ignore_index=ignore_index,
	)
	cat_hole = F.cross_entropy(
		out["hole_logits"].reshape(-1, out["hole_logits"].shape[-1]),
		categorical_targets["hole_target"].reshape(-1),
		ignore_index=ignore_index,
	)
	categorical_loss = cat_type + cat_function + cat_role + cat_hole

	numeric_targets = prepared["numeric_targets"]
	valid_numeric_mask = prepared["valid_numeric_mask"]
	num_x = _masked_mse(out["x_pred"], numeric_targets["x_target"], valid_numeric_mask)
	num_y = _masked_mse(out["y_pred"], numeric_targets["y_target"], valid_numeric_mask)
	num_depth = _masked_mse(out["depth_pred"], numeric_targets["depth_target"], valid_numeric_mask)
	num_osin = _masked_mse(out["orientation_sin_pred"], numeric_targets["orientation_sin_target"], valid_numeric_mask)
	num_ocos = _masked_mse(out["orientation_cos_pred"], numeric_targets["orientation_cos_target"], valid_numeric_mask)
	num_size = _masked_mse(out["size_pred"], numeric_targets["size_target"], valid_numeric_mask)
	numeric_loss = num_x + num_y + num_depth + num_osin + num_ocos + num_size

	kl_loss = kl_divergence_loss(out["mu"], out["logvar"], reduction="mean")
	total = categorical_loss + numeric_weight * numeric_loss + kl_beta * kl_loss

	stats = {
		"total": float(total.detach().cpu()),
		"categorical": float(categorical_loss.detach().cpu()),
		"numeric": float(numeric_loss.detach().cpu()),
		"kl": float(kl_loss.detach().cpu()),
	}
	return total, stats



def _run_epoch(
	model: RouteConditionalVAE,
	samples: list[Any],
	*,
	optimizer: AdamW | None,
	device: torch.device,
	eos_ids: DecoderEOSIds,
	batch_size: int,
	kl_beta: float,
	numeric_weight: float,
	grad_clip_norm: float,
	sample_latent: bool,
) -> EpochMetrics:
	is_train = optimizer is not None
	model.train(is_train)

	loss_sum = 0.0
	cat_sum = 0.0
	num_sum = 0.0
	kl_sum = 0.0
	batches = 0

	for batch_samples in _iter_minibatches(samples, batch_size=batch_size):
		if is_train:
			optimizer.zero_grad(set_to_none=True)

		with torch.set_grad_enabled(is_train):
			loss, stats = _compute_batch_losses(
				model,
				batch_samples,
				device=device,
				eos_ids=eos_ids,
				kl_beta=kl_beta,
				numeric_weight=numeric_weight,
				sample_latent=sample_latent,
			)

			if is_train:
				loss.backward()
				nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
				optimizer.step()

		loss_sum += stats["total"]
		cat_sum += stats["categorical"]
		num_sum += stats["numeric"]
		kl_sum += stats["kl"]
		batches += 1

	if batches == 0:
		return EpochMetrics(0.0, 0.0, 0.0, 0.0, 0)

	return EpochMetrics(
		total_loss=loss_sum / batches,
		categorical_loss=cat_sum / batches,
		numeric_loss=num_sum / batches,
		kl_loss=kl_sum / batches,
		num_batches=batches,
	)



def _select_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")



def _build_model(vocab_bundle, device: torch.device, latent_dim: int) -> tuple[RouteConditionalVAE, DecoderEOSIds]:
	enc_cfg = vocab_bundle.to_transformer_config()
	dec_cfg = RouteVAEDecoderConfig(
		type_vocab_size=enc_cfg.type_vocab_size,
		function_vocab_size=enc_cfg.function_vocab_size,
		role_vocab_size=enc_cfg.role_vocab_size,
		hole_id_vocab_size=enc_cfg.hole_id_vocab_size,
		latent_dim=latent_dim,
		# Keep normalization ranges consistent across encoder and decoder.
		x_min=enc_cfg.x_min,
		x_max=enc_cfg.x_max,
		y_min=enc_cfg.y_min,
		y_max=enc_cfg.y_max,
		angle_min=enc_cfg.angle_min,
		angle_max=enc_cfg.angle_max,
		grade_min=enc_cfg.grade_min,
		grade_max=enc_cfg.grade_max,
	)
	bottleneck_cfg = RouteVAEBottleneckConfig(
		encoder_embedding_dim=enc_cfg.d_model,
		latent_dim=latent_dim,
	)

	encoder = RouteTransformerEncoder(enc_cfg)
	decoder = RouteTransformerDecoder(dec_cfg)
	bottleneck = RouteVAEBottleneck(bottleneck_cfg)
	model = RouteConditionalVAE(encoder=encoder, bottleneck=bottleneck, decoder=decoder).to(device)

	eos_ids = DecoderEOSIds(
		type_eos_id=decoder.type_eos_id,
		function_eos_id=decoder.function_eos_id,
		role_eos_id=decoder.role_eos_id,
		hole_eos_id=decoder.hole_eos_id,
	)
	return model, eos_ids


def _load_or_build_samples_and_vocabs(args: argparse.Namespace) -> tuple[list[Any], Any]:
	cache_path = Path(args.cache_path)
	cache_path.parent.mkdir(parents=True, exist_ok=True)

	cache_payload = None
	if cache_path.exists() and not args.rebuild_cache:
		cache_payload = torch.load(cache_path, map_location="cpu", weights_only=False)

	cache_key = {
		"db_path": str(Path(args.db_path).resolve()),
		"metadata_source": args.metadata_source,
		"metadata_product_id": args.metadata_product_id,
		"require_full_metadata": args.require_full_metadata,
		"max_routes": args.max_routes,
		"min_holds": args.min_holds,
	}

	if cache_payload is not None and cache_payload.get("cache_key") == cache_key:
		print(f"Loaded preprocessed dataset cache: {cache_path}")
		return cache_payload["samples"], cache_payload["vocabs"]

	print("Building dataset from DB (cache miss)...")
	samples, vocabs = build_training_samples_from_db(
		args.db_path,
		require_full_metadata=args.require_full_metadata,
		metadata_source=args.metadata_source,
		metadata_product_id=args.metadata_product_id,
		max_routes=args.max_routes,
		min_holds=args.min_holds,
	)

	torch.save(
		{
			"cache_key": cache_key,
			"samples": samples,
			"vocabs": vocabs,
		},
		cache_path,
	)
	print(f"Saved preprocessed dataset cache: {cache_path}")
	return samples, vocabs



def main() -> None:
	parser = argparse.ArgumentParser(description="Train route conditional VAE.")
	parser.add_argument("--db-path", type=str, default=str(PROJECT_ROOT / "data/raw/kilter_database.sqlite"))
	parser.add_argument("--metadata-source", type=str, default="kilter_board_csv")
	parser.add_argument("--metadata-product-id", type=int, default=1)
	parser.add_argument("--require-full-metadata", action="store_true")
	parser.add_argument("--max-routes", type=int, default=100000)
	parser.add_argument("--min-holds", type=int, default=1)
	parser.add_argument(
		"--cache-path",
		type=str,
		default=str(PROJECT_ROOT / "artifacts/preprocessed_routes_cache.pt"),
	)
	parser.add_argument("--rebuild-cache", action="store_true")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--latent-dim", type=int, default=32)
	parser.add_argument("--numeric-weight", type=float, default=0.25)
	parser.add_argument("--kl-beta", type=float, default=1.0)
	parser.add_argument(
		"--kl-warmup-epochs",
		type=int,
		default=None,
		help="Number of epochs to linearly warm KL from 0 to --kl-beta. Defaults to 25% of total epochs.",
	)
	parser.add_argument("--grad-clip-norm", type=float, default=1.0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--checkpoint-path", type=str, default=str(PROJECT_ROOT / "artifacts/route_cvae.pt"))
	parser.add_argument(
		"--loss-plot-path",
		type=str,
		default=str(PROJECT_ROOT / "artifacts/route_cvae_loss_curve.png"),
		help="Path to save training/validation total loss curve.",
	)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--resume-path", type=str, default=None)
	args = parser.parse_args()
	if args.kl_warmup_epochs is None:
		args.kl_warmup_epochs = max(1, int(math.ceil(0.25 * args.epochs)))
	if args.kl_warmup_epochs < 0:
		raise ValueError("--kl-warmup-epochs must be >= 0")

	torch.manual_seed(args.seed)
	device = _select_device()
	print(f"Using device: {device}")

	samples, vocabs = _load_or_build_samples_and_vocabs(args)
	train_samples, val_samples, test_samples = split_route_samples(samples, train_ratio=0.8, val_ratio=0.1, seed=args.seed)
	print(
		f"Loaded routes: total={len(samples)} train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
	)

	model, eos_ids = _build_model(vocabs, device=device, latent_dim=args.latent_dim)

	train_samples, skipped_train = _filter_samples_by_decoder_max_len(
		train_samples,
		max_seq_len=model.decoder.cfg.max_seq_len,
	)
	val_samples, skipped_val = _filter_samples_by_decoder_max_len(
		val_samples,
		max_seq_len=model.decoder.cfg.max_seq_len,
	)
	test_samples, skipped_test = _filter_samples_by_decoder_max_len(
		test_samples,
		max_seq_len=model.decoder.cfg.max_seq_len,
	)

	if skipped_train or skipped_val or skipped_test:
		print(
			"Skipped overlength routes "
			f"(max_seq_len={model.decoder.cfg.max_seq_len}, max_target_len={model.decoder.cfg.max_seq_len - 1}): "
			f"train={skipped_train}, val={skipped_val}, test={skipped_test}"
		)
		print(
			f"After filtering: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
		)

	if len(train_samples) == 0:
		raise ValueError(
			"No training samples remain after max_seq_len filtering. "
			"Increase decoder max_seq_len or reduce route length."
		)
	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	best_val = math.inf
	start_epoch = 1
	checkpoint_path = Path(args.checkpoint_path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

	if args.resume:
		resume_path = Path(args.resume_path) if args.resume_path else checkpoint_path
		if not resume_path.exists():
			raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

		resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
		model.load_state_dict(resume_ckpt["model_state_dict"])
		if "optimizer_state_dict" in resume_ckpt:
			optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])

		start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
		best_val = float(resume_ckpt.get("best_val", math.inf))
		print(
			f"Resumed from checkpoint: {resume_path} "
			f"(next_epoch={start_epoch}, best_val={best_val:.4f})"
		)

	epoch_history: list[int] = []
	train_total_history: list[float] = []
	val_total_history: list[float] = []

	for epoch in range(start_epoch, args.epochs + 1):
		epoch_kl_beta = _compute_kl_beta_for_epoch(
			epoch=epoch,
			target_kl_beta=args.kl_beta,
			kl_warmup_epochs=args.kl_warmup_epochs,
		)
		train_metrics = _run_epoch(
			model,
			train_samples,
			optimizer=optimizer,
			device=device,
			eos_ids=eos_ids,
			batch_size=args.batch_size,
			kl_beta=epoch_kl_beta,
			numeric_weight=args.numeric_weight,
			grad_clip_norm=args.grad_clip_norm,
			sample_latent=True,
		)

		with torch.no_grad():
			val_metrics = _run_epoch(
				model,
				val_samples,
				optimizer=None,
				device=device,
				eos_ids=eos_ids,
				batch_size=args.batch_size,
				kl_beta=epoch_kl_beta,
				numeric_weight=args.numeric_weight,
				grad_clip_norm=args.grad_clip_norm,
				sample_latent=False,
			)

		train_recon = train_metrics.categorical_loss + args.numeric_weight * train_metrics.numeric_loss
		train_weighted_kl = epoch_kl_beta * train_metrics.kl_loss
		val_recon = val_metrics.categorical_loss + args.numeric_weight * val_metrics.numeric_loss
		val_weighted_kl = epoch_kl_beta * val_metrics.kl_loss

		print(
			f"Epoch {epoch:03d} | "
			f"kl_beta={epoch_kl_beta:.6f} "
			f"train total={train_metrics.total_loss:.4f} recon={train_recon:.4f} w_kl={train_weighted_kl:.4f} "
			f"cat={train_metrics.categorical_loss:.4f} num={train_metrics.numeric_loss:.4f} kl={train_metrics.kl_loss:.4f} | "
			f"val total={val_metrics.total_loss:.4f} recon={val_recon:.4f} w_kl={val_weighted_kl:.4f} "
			f"cat={val_metrics.categorical_loss:.4f} num={val_metrics.numeric_loss:.4f} kl={val_metrics.kl_loss:.4f}"
		)

		epoch_history.append(epoch)
		train_total_history.append(train_metrics.total_loss)
		val_total_history.append(val_metrics.total_loss)

		if val_metrics.total_loss < best_val:
			best_val = val_metrics.total_loss
			torch.save(
				{
					"epoch": epoch,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"best_val": best_val,
					"args": vars(args),
				},
				checkpoint_path,
			)
			print(f"Saved checkpoint: {checkpoint_path}")

	_save_loss_curve_plot(
		epochs=epoch_history,
		train_total_losses=train_total_history,
		val_total_losses=val_total_history,
		output_path=Path(args.loss_plot_path),
	)
	
	best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
	model.load_state_dict(best_ckpt["model_state_dict"])

	with torch.no_grad():
		test_metrics = _run_epoch(
			model,
			test_samples,
			optimizer=None,
			device=device,
			eos_ids=eos_ids,
			batch_size=args.batch_size,
			kl_beta=args.kl_beta,
			numeric_weight=args.numeric_weight,
			grad_clip_norm=args.grad_clip_norm,
			sample_latent=False,
		)
	test_recon = test_metrics.categorical_loss + args.numeric_weight * test_metrics.numeric_loss
	test_weighted_kl = args.kl_beta * test_metrics.kl_loss
	print(
		f"Test | total={test_metrics.total_loss:.4f} recon={test_recon:.4f} w_kl={test_weighted_kl:.4f} "
		f"cat={test_metrics.categorical_loss:.4f} num={test_metrics.numeric_loss:.4f} kl={test_metrics.kl_loss:.4f}"
	)


if __name__ == "__main__":
	main()
