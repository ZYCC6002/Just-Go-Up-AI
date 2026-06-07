from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_analysis.clustered_analysis_utils import (
    load_cluster_cache,
    plot_2d_latents,
    reduce_dimensions,
    subset_cluster_cache,
)


def run_analysis(
    *,
    cluster_cache_path: str,
    method: str,
    max_routes: int,
    seed: int,
    output_path: str,
    show_plot: bool,
    click_to_visualize: bool,
    checkpoint_path: str | None,
    n_dims: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    tsne_perplexity: float,
    tsne_init: Literal["pca", "random"],
    tsne_learning_rate: float | Literal["auto"],
    tsne_metric: str,
) -> None:
    # Build decoded-route callback if a checkpoint was supplied.
    visualize_decoded = None
    if click_to_visualize and checkpoint_path is not None:
        import torch as _torch
        from generate_route import decode_z_to_holds, visualize_generated_route
        from model_training import DecoderEOSIds, build_model_from_checkpoint, prepare_cvae_training_batch

        _device = _torch.device("cpu")
        _model, _ = build_model_from_checkpoint(checkpoint_path, vocabs=None, device=_device)
        _model.eval()
        _ckpt = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _vocabs = _ckpt["vocabs"]
        _db_path = PROJECT_ROOT / "data" / "raw" / "kilter_database.sqlite"
        _images_root = PROJECT_ROOT / "data" / "raw" / "kilter_images"
        _eos_ids = DecoderEOSIds(
            type_eos_id=_model.decoder.type_eos_id,
            role_eos_id=_model.decoder.role_eos_id,
            hole_eos_id=_model.decoder.hole_eos_id,
        )

        def visualize_decoded(sample) -> None:
            batch = prepare_cvae_training_batch([sample], eos_ids=_eos_ids, device=_device)
            with _torch.no_grad():
                enc_out = _model.encoder(batch["encoder_batch"])
                bn_out = _model.bottleneck(enc_out["route_embedding"], sample_latent=False)
            z = bn_out["z"]  # [1, latent_dim] — posterior mean, no sampling
            holds = decode_z_to_holds(_model, _vocabs, z, sample.num_holds, _device)
            print(f"  → {len(holds)} holds decoded")
            visualize_generated_route(
                holds,
                db_path=_db_path,
                images_root=_images_root,
                title=f"Decoded: {sample.name}",
            )

        print(f"Loaded decoder checkpoint for click-to-decode: {checkpoint_path}")

    samples, latent_matrix, cluster_ids, _payload = load_cluster_cache(cluster_cache_path)
    samples, latent_matrix, cluster_ids = subset_cluster_cache(
        samples, latent_matrix, cluster_ids, max_routes=max_routes
    )
    if not samples:
        raise ValueError("No clustered samples available for visualization.")

    # If the cache stored a pre-reduced matrix (from --pre-reduce at cluster time),
    # subset it the same way so indices stay aligned.
    _pre_reduced_full = _payload.get("pre_reduced_matrix")
    pre_reduced_matrix = (
        np.asarray(_pre_reduced_full[:max_routes], dtype=np.float32)
        if _pre_reduced_full is not None and max_routes is not None
        else (np.asarray(_pre_reduced_full, dtype=np.float32) if _pre_reduced_full is not None else None)
    )

    extra_info = ""

    dim_label = f"{n_dims}D"

    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_dims, random_state=seed)
        projected_2d = pca.fit_transform(latent_matrix)
        ev = pca.explained_variance_ratio_
        ev_str = "  ".join(f"PC{i+1}={v:.4f}" for i, v in enumerate(ev))
        extra_info = f"Explained variance: {ev_str}"
        title = f"Clustered Latents PCA ({dim_label})"
        axis_labels = tuple(f"PC{i+1}" for i in range(n_dims))

    elif method == "umap":
        if pre_reduced_matrix is not None:
            pre_reduce_dims = int(_payload.get("pre_reduce_dims", pre_reduced_matrix.shape[1]))
            print(
                f"Projecting stored {pre_reduce_dims}D pre-reduced embedding → {dim_label} UMAP "
                f"(anchored to clustering space)."
            )
            umap_input = pre_reduced_matrix
        else:
            umap_input = latent_matrix

        projected_2d = reduce_dimensions(
            umap_input,
            method="umap",
            n_components=n_dims,
            seed=seed,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
        )
        title = f"Clustered Latents UMAP ({dim_label})"
        axis_labels = tuple(f"UMAP-{i+1}" for i in range(n_dims))

    elif method == "tsne":
        from sklearn.manifold import TSNE

        if tsne_perplexity >= len(samples):
            raise ValueError(
                f"--tsne-perplexity ({tsne_perplexity}) must be smaller than "
                f"sample count ({len(samples)})."
            )
        tsne = TSNE(
            n_components=n_dims,
            perplexity=tsne_perplexity,
            learning_rate=tsne_learning_rate,
            init=tsne_init,
            metric=tsne_metric,
            random_state=seed,
        )
        projected_2d = tsne.fit_transform(latent_matrix)
        title = f"Clustered Latents t-SNE ({dim_label})"
        axis_labels = tuple(f"t-SNE-{i+1}" for i in range(n_dims))

    else:
        raise ValueError(f"Unknown visualization method: {method!r}. Choose 'pca', 'umap', or 'tsne'.")

    # Resolve default output path
    if output_path == "__auto__":
        output_path = str(PROJECT_ROOT / f"data/routes_latent_{method}.png")

    plot_kwargs: dict = dict(
        title=title,
        output_path=output_path,
        show_plot=show_plot,
        click_to_visualize=click_to_visualize,
        visualize_decoded=visualize_decoded,
        n_dims=n_dims,
        extra_info=extra_info,
        x_label=axis_labels[0],
        y_label=axis_labels[1],
    )
    if n_dims == 3:
        plot_kwargs["z_label"] = axis_labels[2]

    plot_2d_latents(projected_2d, cluster_ids, samples, **plot_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pre-clustered CVAE latent vectors (PCA / UMAP / t-SNE)."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["pca", "umap", "tsne"],
        default="pca",
        help="Dimensionality-reduction method for 2D visualization.",
    )
    parser.add_argument(
        "--cluster-cache-path",
        type=str,
        default=str(PROJECT_ROOT / "data/routes_kmeans_original.pt"),
    )
    parser.add_argument("--max-routes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=str,
        default="__auto__",
        help="PNG save path. Defaults to data/routes_latent_{method}.png.",
    )
    parser.add_argument("--show", action="store_true", help="Display interactive plot window.")
    parser.add_argument(
        "--disable-click-visualizer",
        action="store_true",
        help="Disable click-to-open route visualizer.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to a VAE checkpoint (.pt). When provided, a Real/Decoded radio button "
             "appears on the scatter plot — clicking 'Decoded' shows the VAE reconstruction "
             "of the selected route instead of the real climb.",
    )
    parser.add_argument(
        "--n-dims",
        type=int,
        choices=[2, 3],
        default=2,
        help="Projection dimensionality: 2 for a flat scatter, 3 for a rotatable 3D plot. Default: 2.",
    )

    # UMAP-specific
    parser.add_argument(
        "--umap-n-neighbors", type=int, default=50,
        help="UMAP n_neighbors for 2D projection. Higher = more global structure. Default: 50.",
    )
    parser.add_argument(
        "--umap-min-dist", type=float, default=0.1,
        help="UMAP min_dist for 2D projection. Higher spreads points out for visual clarity. Default: 0.1.",
    )

    # t-SNE-specific
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-init", type=str, choices=["pca", "random"], default="pca")
    parser.add_argument(
        "--tsne-learning-rate",
        type=str,
        default="auto",
        help="t-SNE learning rate as float or 'auto'.",
    )
    parser.add_argument("--tsne-metric", type=str, default="euclidean")

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
        method=args.method,
        max_routes=args.max_routes,
        seed=args.seed,
        output_path=args.output_path,
        show_plot=args.show,
        click_to_visualize=not args.disable_click_visualizer,
        checkpoint_path=args.checkpoint_path,
        n_dims=args.n_dims,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        tsne_perplexity=args.tsne_perplexity,
        tsne_init=tsne_init,
        tsne_learning_rate=tsne_learning_rate,
        tsne_metric=args.tsne_metric,
    )


if __name__ == "__main__":
    main()
