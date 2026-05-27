from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from data_preprocessing.route_preprocessing import RouteSample



def load_cluster_cache(cluster_cache_path: str) -> tuple[list[RouteSample], np.ndarray, np.ndarray, dict[str, Any]]:
	cache_file = Path(cluster_cache_path)
	if not cache_file.exists():
		raise FileNotFoundError(f"Cluster cache not found: {cache_file}")

	payload = torch.load(cache_file, map_location="cpu", weights_only=False)
	samples = payload.get("samples")
	latent_matrix = payload.get("latent_matrix")
	cluster_ids = payload.get("cluster_ids")
	if samples is None or latent_matrix is None or cluster_ids is None:
		raise ValueError(f"Cluster cache is missing required fields: {cache_file}")

	return (
		list(samples),
		np.asarray(latent_matrix, dtype=np.float32),
		np.asarray(cluster_ids, dtype=np.int64),
		payload,
	)



def subset_cluster_cache(
	samples: list[RouteSample],
	features: np.ndarray,
	cluster_ids: np.ndarray,
	*,
	max_routes: int | None,
) -> tuple[list[RouteSample], np.ndarray, np.ndarray]:
	if max_routes is None:
		return samples, features, cluster_ids
	end = min(len(samples), int(max_routes))
	return samples[:end], features[:end], cluster_ids[:end]



def standardize_features(features: np.ndarray) -> np.ndarray:
	mean = features.mean(axis=0, keepdims=True)
	std = features.std(axis=0, keepdims=True)
	std[std < 1e-6] = 1.0
	return (features - mean) / std



def enable_hover_annotations(
	*,
	ax: Axes,
	fig: Figure,
	scatter,
	projected_2d: np.ndarray,
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
		return f"{name}\ngrade={grade}\nangle={angle}\ncluster={int(cluster_ids[i])}"

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
		x, y = projected_2d[idx]
		annotation.xy = (x, y)
		annotation.set_text(_label(idx))
		annotation.set_visible(True)
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect("motion_notify_event", _on_move)



def enable_click_to_visualize(*, ax: Axes, fig: Figure, scatter, samples: list[RouteSample], visualize_route) -> None:
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
			visualize_route(sample.name, sample=sample)
		except Exception as exc:
			print(f"Failed to visualize route '{sample.name}': {exc}")

	fig.canvas.mpl_connect("button_press_event", _on_click)



def reduce_dimensions(
	matrix: np.ndarray,
	*,
	method: str,
	n_components: int,
	seed: int = 42,
	umap_n_neighbors: int = 15,
	umap_min_dist: float = 0.1,
) -> np.ndarray:
	"""Reduce dimensionality of a feature matrix.

	Args:
		matrix:           Input matrix [N, D].
		method:           ``"pca"`` or ``"umap"``.
		n_components:     Target number of dimensions.
		seed:             Random seed for reproducibility.
		umap_n_neighbors: UMAP neighbour count (ignored for PCA).
		umap_min_dist:    UMAP minimum distance parameter (ignored for PCA).

	Returns:
		Reduced matrix [N, n_components].
	"""
	if method == "pca":
		from sklearn.decomposition import PCA

		return PCA(n_components=n_components, random_state=seed).fit_transform(matrix)
	if method == "umap":
		import umap as umap_lib

		reducer = umap_lib.UMAP(
			n_components=n_components,
			n_neighbors=umap_n_neighbors,
			min_dist=umap_min_dist,
			random_state=seed,
		)
		return reducer.fit_transform(matrix)
	raise ValueError(f"Unknown dimensionality-reduction method: {method!r}. Choose 'pca' or 'umap'.")



def plot_2d_latents(
	projected_2d: np.ndarray,
	cluster_ids: np.ndarray,
	samples: list[RouteSample],
	*,
	title: str,
	output_path: str,
	show_plot: bool,
	click_to_visualize: bool,
	extra_info: str = "",
	x_label: str = "Dim 1",
	y_label: str = "Dim 2",
) -> None:
	"""Render a 2-D scatter plot of latent vectors coloured by cluster.

	Handles HDBSCAN noise points (label ``-1``) as grey dots excluded from
	the legend.  Wires hover-annotation and (optionally) click-to-visualize
	interactions.

	Args:
		projected_2d:       2-D projection [N, 2].
		cluster_ids:        Cluster labels [N]; may include ``-1`` (noise).
		samples:            Corresponding :class:`RouteSample` objects.
		title:              Figure title.
		output_path:        Where to save the PNG.
		show_plot:          Whether to call ``plt.show()``.
		click_to_visualize: Whether to wire the click handler.
		extra_info:         Extra string printed to stdout after save (e.g.
		                    explained-variance ratios).
		x_label:            X-axis label (default ``"Dim 1"``).
		y_label:            Y-axis label (default ``"Dim 2"``).
	"""
	from route_visualizer import visualize_route

	unique_labels = np.unique(cluster_ids)
	noise_mask = cluster_ids == -1
	cluster_labels = unique_labels[unique_labels != -1]

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111)

	# Plot noise points first (grey, not in legend)
	if noise_mask.any():
		ax.scatter(
			projected_2d[noise_mask, 0],
			projected_2d[noise_mask, 1],
			c="lightgrey",
			s=6,
			alpha=0.5,
			label="_noise",
			zorder=1,
		)

	# Plot cluster points — use tab10 but only over the actual cluster labels
	cmap = plt.cm.get_cmap("tab10", max(len(cluster_labels), 1))
	label_to_color: dict[int, Any] = {int(lbl): cmap(i) for i, lbl in enumerate(cluster_labels)}

	# Build unified colour array for scatter (needed for hover)
	non_noise = ~noise_mask
	colors_non_noise = [label_to_color[int(c)] for c in cluster_ids[non_noise]]

	scatter = ax.scatter(
		projected_2d[non_noise, 0],
		projected_2d[non_noise, 1],
		c=colors_non_noise,
		s=8,
		alpha=0.85,
		zorder=2,
	)

	# Legend entries — one patch per cluster
	import matplotlib.patches as mpatches

	legend_handles = [
		mpatches.Patch(color=label_to_color[int(lbl)], label=f"Cluster {int(lbl)}")
		for lbl in cluster_labels
	]
	if noise_mask.any():
		legend_handles.append(
			mpatches.Patch(color="lightgrey", label=f"Noise ({int(noise_mask.sum())})")
		)
	ax.legend(handles=legend_handles, title="Cluster", loc="best")

	ax.set_title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)

	enable_hover_annotations(
		ax=ax,
		fig=fig,
		scatter=scatter,
		projected_2d=projected_2d,
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
	print(f"Saved plot to: {out}")
	if extra_info:
		print(extra_info)

	if show_plot:
		plt.show()
	else:
		plt.close(fig)
