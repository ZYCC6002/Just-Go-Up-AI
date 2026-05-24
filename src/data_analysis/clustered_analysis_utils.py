from __future__ import annotations

from pathlib import Path
from typing import Any

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
