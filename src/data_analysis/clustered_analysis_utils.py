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
	projected: np.ndarray,
	samples: list[RouteSample],
	cluster_ids: np.ndarray,
) -> None:
	n_dims = projected.shape[1]

	def _label(i: int) -> str:
		name = samples[i].name.strip() if samples[i].name else "<unnamed route>"
		grade = samples[i].grade if samples[i].grade is not None else "<ungraded>"
		angle = samples[i].angle if samples[i].angle is not None else "?"
		if len(name) > 60:
			name = name[:57] + "..."
		return f"{name} | grade={grade} | angle={angle} | cluster={int(cluster_ids[i])}"

	if n_dims == 3:
		# In 3D use a fixed status text at the bottom — avoids 3D annotation
		# placement issues while still showing route info on hover.
		from mpl_toolkits.mplot3d import proj3d
		xs_3d = projected[:, 0].astype(float)
		ys_3d = projected[:, 1].astype(float)
		zs_3d = projected[:, 2].astype(float)
		status = fig.text(
			0.5, 0.01, "", ha="center", va="bottom", fontsize=8.5,
			bbox={"boxstyle": "round", "fc": "white", "ec": "gray", "alpha": 0.85},
		)
		status.set_visible(False)

		def _on_move(event) -> None:
			if event.x is None or event.y is None:
				return
			try:
				renderer = fig.canvas.get_renderer()
				bbox_ax = ax.get_window_extent(renderer)
				x2d, y2d, _ = proj3d.proj_transform(xs_3d, ys_3d, zs_3d, ax.get_proj())
				# proj_transform output is in normalized [-1,1] projected space;
				# map to pixel coords using the axes bounding box.
				x_px = bbox_ax.x0 + (x2d + 1) / 2 * bbox_ax.width
				y_px = bbox_ax.y0 + (y2d + 1) / 2 * bbox_ax.height
				dists = np.hypot(x_px - event.x, y_px - event.y)
				idx = int(np.argmin(dists))
				if dists[idx] < 15:
					status.set_text(_label(idx))
					status.set_visible(True)
				else:
					status.set_visible(False)
				fig.canvas.draw_idle()
			except Exception:
				pass
	else:
		annotation = ax.annotate(
			"",
			xy=(0, 0),
			xytext=(10, 10),
			textcoords="offset points",
			bbox={"boxstyle": "round", "fc": "white", "ec": "black", "alpha": 0.9},
			arrowprops={"arrowstyle": "->", "lw": 0.8},
		)
		annotation.set_visible(False)

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
			x, y = projected[idx]
			annotation.xy = (x, y)
			annotation.set_text(_label(idx))
			annotation.set_visible(True)
			fig.canvas.draw_idle()

	fig.canvas.mpl_connect("motion_notify_event", _on_move)



def enable_click_to_visualize(
	*,
	ax: Axes,
	fig: Figure,
	scatter,
	samples: list[RouteSample],
	visualize_route,
	visualize_decoded=None,
	n_dims: int = 2,
) -> list[str]:
	"""Wire left-click to open the real or decoded route.

	Returns a mutable ``mode`` list so the caller can wire a toggle widget:
	``mode[0]`` is ``"Real"`` or ``"Decoded"``.

	In 3D mode the scatter must have ``picker=True`` set — click detection uses
	``pick_event`` because ``scatter.contains()`` is unreliable for 3D artists.
	"""
	mode: list[str] = ["Real"]

	def _dispatch(idx: int) -> None:
		sample = samples[idx]
		if mode[0] == "Decoded" and visualize_decoded is not None:
			print(f"Decoding route: {sample.name} ({sample.uuid})")
			try:
				visualize_decoded(sample)
			except Exception as exc:
				print(f"Failed to decode route '{sample.name}': {exc}")
		else:
			print(f"Opening route visualizer for: {sample.name} ({sample.uuid})")
			try:
				visualize_route(sample.name, sample=sample)
			except Exception as exc:
				print(f"Failed to visualize route '{sample.name}': {exc}")

	if n_dims == 3:
		def _on_pick(event) -> None:
			if event.artist is not scatter:
				return
			if event.mouseevent.button != 1:  # ignore scroll (button="up"/"down")
				return
			_dispatch(int(event.ind[0]))
		fig.canvas.mpl_connect("pick_event", _on_pick)
	else:
		def _on_click(event) -> None:
			if event.inaxes != ax:
				return
			contains, info = scatter.contains(event)
			if not contains:
				return
			_dispatch(int(info["ind"][0]))
		fig.canvas.mpl_connect("button_press_event", _on_click)

	return mode



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
	visualize_decoded=None,
	n_dims: int = 2,
	extra_info: str = "",
	x_label: str = "Dim 1",
	y_label: str = "Dim 2",
	z_label: str = "Dim 3",
) -> None:
	"""Render a 2-D or 3-D scatter plot of latent vectors coloured by cluster."""
	import matplotlib.patches as mpatches

	from route_visualizer import visualize_route

	unique_labels = np.unique(cluster_ids)
	noise_mask = cluster_ids == -1
	cluster_labels = unique_labels[unique_labels != -1]

	cmap = plt.cm.get_cmap("tab10", max(len(cluster_labels), 1))
	label_to_rgba: dict[int, np.ndarray] = {}
	for i, lbl in enumerate(cluster_labels):
		rgba = np.array(cmap(i))
		rgba[3] = 0.85
		label_to_rgba[int(lbl)] = rgba

	noise_rgba = np.array([0.827, 0.827, 0.827, 0.45])

	point_colors = np.array([
		noise_rgba if c == -1 else label_to_rgba[int(c)]
		for c in cluster_ids
	])
	point_sizes = np.where(noise_mask, 5, 8).astype(float)

	if n_dims == 3:
		fig = plt.figure(figsize=(12, 9))
		ax = fig.add_subplot(111, projection="3d")
		scatter = ax.scatter(
			projected_2d[:, 0],
			projected_2d[:, 1],
			projected_2d[:, 2],
			c=point_colors,
			s=point_sizes,
			picker=True,
			pickradius=5,
		)
		ax.set_axis_off()

		# Scroll-to-zoom: scale all three axis ranges around their centres.
		def _on_scroll(event) -> None:
			if event.inaxes != ax:
				return
			scale = 0.9 if event.button == "up" else 1.1
			for get, set_ in (
				(ax.get_xlim, ax.set_xlim),
				(ax.get_ylim, ax.set_ylim),
				(ax.get_zlim, ax.set_zlim),
			):
				lo, hi = get()
				mid = (lo + hi) / 2
				half = (hi - lo) / 2 * scale
				set_(mid - half, mid + half)
			fig.canvas.draw_idle()

		fig.canvas.mpl_connect("scroll_event", _on_scroll)
	else:
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111)
		scatter = ax.scatter(
			projected_2d[:, 0],
			projected_2d[:, 1],
			c=point_colors,
			s=point_sizes,
		)

	# Legend — one patch per cluster, plus noise count if present
	legend_handles = [
		mpatches.Patch(color=label_to_rgba[int(lbl)], label=f"Cluster {int(lbl)}")
		for lbl in cluster_labels
	]
	if noise_mask.any():
		legend_handles.append(
			mpatches.Patch(color=noise_rgba, label=f"Noise ({int(noise_mask.sum())})")
		)
	ax.legend(handles=legend_handles, title="Cluster", loc="best")

	ax.set_title(title)
	if n_dims == 2:
		ax.set_xlabel(x_label)
		ax.set_ylabel(y_label)

	enable_hover_annotations(
		ax=ax,
		fig=fig,
		scatter=scatter,
		projected=projected_2d,
		samples=samples,
		cluster_ids=cluster_ids,
	)
	mode = None
	if click_to_visualize:
		mode = enable_click_to_visualize(
			ax=ax,
			fig=fig,
			scatter=scatter,
			samples=samples,
			visualize_route=visualize_route,
			visualize_decoded=visualize_decoded,
			n_dims=n_dims,
		)

	fig.tight_layout()

	# Radio button to toggle Real / Decoded.
	if click_to_visualize and visualize_decoded is not None and mode is not None:
		from matplotlib.widgets import RadioButtons
		if n_dims == 2:
			fig.subplots_adjust(left=0.16)
		rax = fig.add_axes([0.01, 0.44, 0.13, 0.13])
		rax.set_facecolor("#f0f0f0")
		radio = RadioButtons(rax, ["Real", "Decoded"], active=0)
		radio.on_clicked(lambda label: mode.__setitem__(0, label))
		fig._click_mode_radio = radio  # keep alive — matplotlib GC's unreferenced widgets

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
