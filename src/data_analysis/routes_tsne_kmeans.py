from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from data_analysis.clustered_analysis_utils import (
	enable_click_to_visualize,
	enable_hover_annotations,
	load_cluster_cache,
	subset_cluster_cache,
)
from route_visualizer import visualize_route



def run_analysis(
	*,
	cluster_cache_path: str,
	max_routes: int,
	seed: int,
	output_path: str,
	show_plot: bool,
	click_to_visualize: bool,
	tsne_perplexity: float,
	tsne_init: Literal["pca", "random"],
	tsne_learning_rate: float | Literal["auto"],
	tsne_metric: str,
) -> None:
	samples, latent_matrix, cluster_ids, _payload = load_cluster_cache(cluster_cache_path)
	samples, latent_matrix, cluster_ids = subset_cluster_cache(
		samples,
		latent_matrix,
		cluster_ids,
		max_routes=max_routes,
	)
	if not samples:
		raise ValueError("No clustered samples available for t-SNE.")
	if tsne_perplexity >= len(samples):
		raise ValueError(
			f"--tsne-perplexity ({tsne_perplexity}) must be smaller than sample count ({len(samples)})."
		)

	tsne = TSNE(
		n_components=2,
		perplexity=tsne_perplexity,
		learning_rate=tsne_learning_rate,
		init=tsne_init,
		metric=tsne_metric,
		random_state=seed,
	)
	tsne_2d = tsne.fit_transform(latent_matrix)

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111)
	scatter = ax.scatter(
		tsne_2d[:, 0],
		tsne_2d[:, 1],
		c=cluster_ids,
		cmap="tab10",
		s=8,
		alpha=0.85,
	)
	ax.set_title("Clustered Latents t-SNE (2D)")
	ax.set_xlabel("t-SNE-1")
	ax.set_ylabel("t-SNE-2")
	legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
	ax.add_artist(legend)

	enable_hover_annotations(
		ax=ax,
		fig=fig,
		scatter=scatter,
		projected_2d=tsne_2d,
		samples=samples,
		cluster_ids=cluster_ids,
	)
	if click_to_visualize:
		enable_click_to_visualize(
			ax=ax,
			fig=fig,
			scatter=scatter,
			samples=samples,
			visualize_route=visualize_route,
		)

	fig.tight_layout()
	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out, dpi=180)
	print(f"Saved t-SNE plot to: {out}")

	if show_plot:
		plt.show()
	else:
		plt.close(fig)



def main() -> None:
	parser = argparse.ArgumentParser(description="t-SNE visualization from preclustered CVAE latent vectors.")
	parser.add_argument(
		"--cluster-cache-path",
		type=str,
		default=str(PROJECT_ROOT / "data/routes_kmeans_original.pt"),
		help="Path to the clustered latent cache produced by routes_kmeans_original.py",
	)
	parser.add_argument("--max-routes", type=int, default=5000)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--tsne-perplexity", type=float, default=30.0)
	parser.add_argument("--tsne-init", type=str, choices=["pca", "random"], default="pca")
	parser.add_argument(
		"--tsne-learning-rate",
		type=str,
		default="auto",
		help="t-SNE learning rate as float or 'auto'.",
	)
	parser.add_argument("--tsne-metric", type=str, default="euclidean")
	parser.add_argument(
		"--output-path",
		type=str,
		default=str(PROJECT_ROOT / "data/routes_tsne_kmeans_latent.png"),
	)
	parser.add_argument("--show", action="store_true", help="Display interactive plot window.")
	parser.add_argument(
		"--disable-click-visualizer",
		action="store_true",
		help="Disable click-to-open route visualizer.",
	)
	args = parser.parse_args()

	if args.tsne_learning_rate.lower() == "auto":
		tsne_learning_rate: float | Literal["auto"] = "auto"
	else:
		try:
			tsne_learning_rate = float(args.tsne_learning_rate)
		except ValueError as exc:
			raise ValueError("--tsne-learning-rate must be a float or 'auto'.") from exc

	tsne_init: Literal["pca", "random"] = args.tsne_init

	run_analysis(
		cluster_cache_path=args.cluster_cache_path,
		max_routes=args.max_routes,
		seed=args.seed,
		output_path=args.output_path,
		show_plot=args.show,
		click_to_visualize=not args.disable_click_visualizer,
		tsne_perplexity=args.tsne_perplexity,
		tsne_init=tsne_init,
		tsne_learning_rate=tsne_learning_rate,
		tsne_metric=args.tsne_metric,
	)


if __name__ == "__main__":
	main()
