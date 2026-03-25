from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from data_preprocessing.route_preprocessing import RouteSample, build_training_samples_from_db
from model_training import (
	DecoderEOSIds,
	RouteConditionalVAE,
	RouteTransformerDecoder,
	RouteTransformerEncoder,
	RouteVAEBottleneck,
	RouteVAEBottleneckConfig,
	RouteVAEDecoderConfig,
	prepare_cvae_training_batch,
)
from route_visualizer import visualize_route



def _select_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")



def _load_samples_and_vocabs(
	*,
	db_path: str,
	cache_path: str,
	metadata_source: str,
	metadata_product_id: int,
	max_routes: int,
) -> tuple[list[RouteSample], Any]:
	cache_file = Path(cache_path)
	if cache_file.exists():
		try:
			payload = torch.load(cache_file, map_location="cpu", weights_only=False)
			samples = payload.get("samples")
			vocabs = payload.get("vocabs")
			if samples and vocabs is not None:
				print(f"Loaded preprocessed routes cache: {cache_file}")
				return samples[:max_routes], vocabs
		except Exception as exc:
			print(f"Warning: failed to load cache at {cache_file}; rebuilding from DB. Error: {exc}")

	print("Building route samples from DB...")
	samples, vocabs = build_training_samples_from_db(
		db_path,
		metadata_source=metadata_source,
		metadata_product_id=metadata_product_id,
		max_routes=max_routes,
	)
	return samples, vocabs



def _build_model_from_checkpoint(
	*,
	checkpoint_path: str,
	vocabs,
	device: torch.device,
	latent_dim_override: int | None = None,
) -> RouteConditionalVAE:
	ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
	ckpt_args: dict[str, Any] = dict(ckpt.get("args", {}))

	enc_cfg = vocabs.to_transformer_config()
	latent_dim = int(latent_dim_override if latent_dim_override is not None else ckpt_args.get("latent_dim", 32))

	dec_cfg = RouteVAEDecoderConfig(
		type_vocab_size=enc_cfg.type_vocab_size,
		function_vocab_size=enc_cfg.function_vocab_size,
		role_vocab_size=enc_cfg.role_vocab_size,
		hole_id_vocab_size=enc_cfg.hole_id_vocab_size,
		latent_dim=latent_dim,
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

	model = RouteConditionalVAE(
		encoder=RouteTransformerEncoder(enc_cfg),
		bottleneck=RouteVAEBottleneck(bottleneck_cfg),
		decoder=RouteTransformerDecoder(dec_cfg),
	).to(device)
	model.load_state_dict(ckpt["model_state_dict"], strict=True)
	model.eval()
	return model



def _iter_minibatches(items: list[Any], batch_size: int):
	for i in range(0, len(items), batch_size):
		yield items[i:i + batch_size]



def _sample_token_len(sample: RouteSample) -> int:
	return int(sample.tokens["type_encoded_id"].shape[0])



def _filter_samples_by_decoder_len(samples: list[RouteSample], *, max_seq_len: int) -> tuple[list[RouteSample], int]:
	max_target_len = max_seq_len - 1  # account for BOS
	kept = [s for s in samples if _sample_token_len(s) <= max_target_len]
	return kept, len(samples) - len(kept)



def _extract_latent_matrix(
	*,
	model: RouteConditionalVAE,
	samples: list[RouteSample],
	batch_size: int,
	device: torch.device,
) -> np.ndarray:
	eos_ids = DecoderEOSIds(
		type_eos_id=model.decoder.type_eos_id,
		function_eos_id=model.decoder.function_eos_id,
		role_eos_id=model.decoder.role_eos_id,
		hole_eos_id=model.decoder.hole_eos_id,
	)

	latents: list[np.ndarray] = []
	with torch.no_grad():
		for batch_samples in _iter_minibatches(samples, batch_size):
			prepared = prepare_cvae_training_batch(batch_samples, eos_ids=eos_ids, device=device)
			enc_out = model.encoder(
				prepared["encoder_batch"],
				angle=prepared["angle"],
				grade=prepared["grade"],
				grade_missing=prepared["grade_missing"],
			)
			# deterministic latent for analysis
			bottleneck_out = model.bottleneck(enc_out["route_embedding"], sample_latent=False)
			latents.append(bottleneck_out["z"].detach().cpu().numpy())

	if not latents:
		raise ValueError("No latent vectors extracted.")
	return np.vstack(latents)



def _standardize(features: np.ndarray) -> np.ndarray:
	mean = features.mean(axis=0, keepdims=True)
	std = features.std(axis=0, keepdims=True)
	std[std < 1e-6] = 1.0
	return (features - mean) / std



def _enable_hover_annotations(
	*,
	ax: Axes,
	fig: Figure,
	scatter,
	pca_2d: np.ndarray,
	samples: list[RouteSample],
	cluster_ids: np.ndarray,
) -> None:
	annotation = ax.annotate(
		"",
		xy=(0, 0),
		xytext=(10, 10),
		textcoords="offset points",
		bbox={"boxstyle": "round", "fc": "white", "ec": "black", "alpha": 0.9},
		arrowprops={"arrowstyle": "->", "lw": 0.8},
	)
	annotation.set_visible(False)

	def _label(i: int) -> str:
		name = samples[i].name.strip() if samples[i].name else "<unnamed route>"
		grade = samples[i].grade if samples[i].grade else "<ungraded route>"
		angle = samples[i].angle if samples[i].angle else "<no angle route>"
		if len(name) > 80:
			name = name[:77] + "..."
		return f"""{name}\ngrade={grade}\nangle={angle}\ncluster={int(cluster_ids[i])}"""

	def _on_move(event) -> None:
		if event.inaxes != ax:
			if annotation.get_visible():
				annotation.set_visible(False)
				fig.canvas.draw_idle()
			return

		contains, info = scatter.contains(event)
		if not contains:
			if annotation.get_visible():
				annotation.set_visible(False)
				fig.canvas.draw_idle()
			return

		idx = int(info["ind"][0])
		x, y = pca_2d[idx]
		annotation.xy = (x, y)
		annotation.set_text(_label(idx))
		annotation.set_visible(True)
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect("motion_notify_event", _on_move)



def _enable_click_to_visualize(*, ax: Axes, fig: Figure, scatter, samples: list[RouteSample]) -> None:
	def _on_click(event) -> None:
		if event.inaxes != ax:
			return
		contains, info = scatter.contains(event)
		if not contains:
			return
		idx = int(info["ind"][0])
		sample = samples[idx]
		print(f"Opening route visualizer for: {sample.name} ({sample.uuid})")
		try:
			visualize_route(sample.name)
		except Exception as exc:
			print(f"Failed to visualize route '{sample.name}': {exc}")

	fig.canvas.mpl_connect("button_press_event", _on_click)



def run_analysis(
	*,
	db_path: str,
	cache_path: str,
	checkpoint_path: str,
	metadata_source: str,
	metadata_product_id: int,
	max_routes: int,
	n_clusters: int,
	batch_size: int,
	latent_dim_override: int | None,
	seed: int,
	output_path: str,
	show_plot: bool,
	click_to_visualize: bool,
) -> None:
	samples, vocabs = _load_samples_and_vocabs(
		db_path=db_path,
		cache_path=cache_path,
		metadata_source=metadata_source,
		metadata_product_id=metadata_product_id,
		max_routes=max_routes,
	)
	if not samples:
		raise ValueError("No route samples available for analysis.")

	device = _select_device()
	print(f"Using device: {device}")

	model = _build_model_from_checkpoint(
		checkpoint_path=checkpoint_path,
		vocabs=vocabs,
		device=device,
		latent_dim_override=latent_dim_override,
	)

	samples, skipped = _filter_samples_by_decoder_len(samples, max_seq_len=model.decoder.cfg.max_seq_len)
	if skipped:
		print(
			f"Skipped {skipped} routes with length > {model.decoder.cfg.max_seq_len - 1} "
			f"(decoder max_seq_len={model.decoder.cfg.max_seq_len})."
		)
	if not samples:
		raise ValueError("All samples filtered out by decoder max_seq_len.")

	latent_matrix = _extract_latent_matrix(
		model=model,
		samples=samples,
		batch_size=batch_size,
		device=device,
	)
	latent_matrix = _standardize(latent_matrix)

	pca = PCA(n_components=2, random_state=seed)
	pca_2d = pca.fit_transform(latent_matrix)

	kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
	cluster_ids = kmeans.fit_predict(pca_2d)

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111)
	scatter = ax.scatter(
		pca_2d[:, 0],
		pca_2d[:, 1],
		c=cluster_ids,
		cmap="tab10",
		s=8,
		alpha=0.85,
	)
	ax.set_title(f"Route Latents PCA (2D) + KMeans (k={n_clusters})")
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
	ax.add_artist(legend)

	_enable_hover_annotations(
		ax=ax,
		fig=fig,
		scatter=scatter,
		pca_2d=pca_2d,
		samples=samples,
		cluster_ids=cluster_ids,
	)
	if click_to_visualize:
		_enable_click_to_visualize(ax=ax, fig=fig, scatter=scatter, samples=samples)

	fig.tight_layout()
	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out, dpi=180)
	print(f"Saved latent PCA+KMeans cluster plot to: {out}")
	print(
		f"Explained variance ratio: "
		f"PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}"
	)

	if show_plot:
		plt.show()
	else:
		plt.close(fig)



def main() -> None:
	parser = argparse.ArgumentParser(description="PCA + KMeans visualization from CVAE latent vectors.")
	parser.add_argument("--db-path", type=str, default=str(PROJECT_ROOT / "data/raw/kilter_database.sqlite"))
	parser.add_argument(
		"--cache-path",
		type=str,
		default=str(PROJECT_ROOT / "artifacts/preprocessed_routes_cache.pt"),
	)
	parser.add_argument(
		"--checkpoint-path",
		type=str,
		default=str(PROJECT_ROOT / "artifacts/route_cvae.pt"),
	)
	parser.add_argument("--metadata-source", type=str, default="kilter_board_csv")
	parser.add_argument("--metadata-product-id", type=int, default=1)
	parser.add_argument("--max-routes", type=int, default=5000)
	parser.add_argument("--n-clusters", type=int, default=6)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--latent-dim", type=int, default=None)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--output-path",
		type=str,
		default=str(PROJECT_ROOT / "artifacts/routes_pca_kmeans_latent.png"),
	)
	parser.add_argument("--show", action="store_true", help="Display interactive plot window.")
	parser.add_argument(
		"--disable-click-visualizer",
		action="store_true",
		help="Disable click-to-open route visualizer.",
	)
	args = parser.parse_args()

	run_analysis(
		db_path=args.db_path,
		cache_path=args.cache_path,
		checkpoint_path=args.checkpoint_path,
		metadata_source=args.metadata_source,
		metadata_product_id=args.metadata_product_id,
		max_routes=args.max_routes,
		n_clusters=args.n_clusters,
		batch_size=args.batch_size,
		latent_dim_override=args.latent_dim,
		seed=args.seed,
		output_path=args.output_path,
		show_plot=args.show,
		click_to_visualize=not args.disable_click_visualizer,
	)


if __name__ == "__main__":
	main()
