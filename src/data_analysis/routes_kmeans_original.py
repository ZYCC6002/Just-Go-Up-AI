from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_preprocessing.route_preprocessing import RouteSample, build_training_samples_from_db
from data_analysis.clustered_analysis_utils import standardize_features
from model_training import (
    DecoderEOSIds,
    build_model_from_checkpoint,
    filter_samples_by_decoder_max_len,
    iter_minibatches,
    select_device,
    prepare_cvae_training_batch,
)


def _load_samples_and_vocabs(
    *,
    db_path: str,
    cache_path: str,
    metadata_source: str,
    metadata_product_id: int,
) -> tuple[list[RouteSample], Any]:
    cache_file = Path(cache_path)
    if cache_file.exists():
        try:
            payload = torch.load(cache_file, map_location="cpu", weights_only=False)
            samples = payload.get("samples")
            vocabs = payload.get("vocabs")
            if samples and vocabs is not None:
                print(f"Loaded preprocessed routes cache: {cache_file}")
                return samples, vocabs
        except Exception as exc:
            print(f"Warning: failed to load cache ({exc}); rebuilding from DB.")

    print("Building route samples from DB...")
    return build_training_samples_from_db(
        db_path,
        metadata_source=metadata_source,
        metadata_product_id=metadata_product_id,
        max_routes=None,
    )


def _load_or_build_angle_grade_map(samples: list[RouteSample], *, cache_path: str) -> dict[str, Any]:
    map_path = Path(cache_path).with_name(f"{Path(cache_path).stem}_angle_grade_map.pt")
    if map_path.exists():
        try:
            payload = torch.load(map_path, map_location="cpu", weights_only=False)
            if (
                int(payload.get("count", -1)) == len(samples)
                and payload.get("first_uuid") == (samples[0].uuid if samples else None)
                and payload.get("last_uuid") == (samples[-1].uuid if samples else None)
            ):
                return payload
        except Exception:
            pass

    angles = np.array([float(s.angle) for s in samples], dtype=np.float32)
    grades = np.array(
        [float(s.grade) if s.grade is not None else np.nan for s in samples],
        dtype=np.float32,
    )
    payload = {
        "count": len(samples),
        "first_uuid": samples[0].uuid if samples else None,
        "last_uuid": samples[-1].uuid if samples else None,
        "angles": angles,
        "grades": grades,
    }
    torch.save(payload, map_path)
    return payload


def _filter_samples_by_grade_angle(
    samples: list[RouteSample],
    *,
    angle_grade_map: dict[str, Any],
    min_grade: float | None,
    max_grade: float | None,
    min_angle: float | None,
    max_angle: float | None,
    include_ungraded: bool,
) -> tuple[np.ndarray, int]:
    angles = np.asarray(angle_grade_map["angles"], dtype=np.float32)
    grades = np.asarray(angle_grade_map["grades"], dtype=np.float32)

    mask = np.ones(len(samples), dtype=bool)
    if min_angle is not None:
        mask &= angles >= float(min_angle)
    if max_angle is not None:
        mask &= angles <= float(max_angle)

    if min_grade is not None or max_grade is not None:
        non_missing = ~np.isnan(grades)
        # Use floor so that integer grade filters (e.g. --min-grade 22 --max-grade 22)
        # capture all routes in that grade band.  Kilter grades are stored as
        # floating-point community averages (e.g. 22.013, 22.94 for V6), so an
        # exact float comparison would match almost nothing.
        grade_band = np.floor(grades)
        in_range = np.ones(len(samples), dtype=bool)
        if min_grade is not None:
            in_range &= grade_band >= float(min_grade)
        if max_grade is not None:
            in_range &= grade_band <= float(max_grade)
        if include_ungraded:
            mask &= (~non_missing) | (non_missing & in_range)
        else:
            mask &= non_missing & in_range

    kept_indices = np.flatnonzero(mask)
    return kept_indices, int(len(samples) - len(kept_indices))


def _extract_latent_matrix(
    *,
    model: Any,
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
        for batch_samples in iter_minibatches(samples, batch_size):
            prepared = prepare_cvae_training_batch(batch_samples, eos_ids=eos_ids, device=device)
            enc_out = model.encoder(
                prepared["encoder_batch"],
                angle=prepared["angle"],
                grade=prepared["grade"],
                grade_missing=prepared["grade_missing"],
            )
            bottleneck_out = model.bottleneck(enc_out["route_embedding"], sample_latent=False)
            latents.append(bottleneck_out["z"].detach().cpu().numpy())

    if not latents:
        raise ValueError("No latent vectors extracted.")
    return np.vstack(latents)


def run_analysis(
    *,
    db_path: str,
    cache_path: str,
    checkpoint_path: str,
    cluster_cache_path: str,
    metadata_source: str,
    metadata_product_id: int,
    max_routes: int,
    n_clusters: int,
    batch_size: int,
    latent_dim_override: int | None,
    min_grade: float | None,
    max_grade: float | None,
    min_angle: float | None,
    max_angle: float | None,
    include_ungraded: bool,
    seed: int,
) -> None:
    samples, vocabs = _load_samples_and_vocabs(
        db_path=db_path,
        cache_path=cache_path,
        metadata_source=metadata_source,
        metadata_product_id=metadata_product_id,
    )
    if not samples:
        raise ValueError("No route samples available for analysis.")

    angle_grade_map = _load_or_build_angle_grade_map(samples, cache_path=cache_path)
    kept_indices, n_filtered = _filter_samples_by_grade_angle(
        samples,
        angle_grade_map=angle_grade_map,
        min_grade=min_grade,
        max_grade=max_grade,
        min_angle=min_angle,
        max_angle=max_angle,
        include_ungraded=include_ungraded,
    )
    if n_filtered:
        print(f"Filtered out {n_filtered} routes by grade/angle constraints.")
    if kept_indices.size == 0:
        raise ValueError("All samples were filtered out by grade/angle constraints.")

    if max_routes is not None:
        kept_indices = kept_indices[:max_routes]
    samples = [samples[int(i)] for i in kept_indices]

    device = select_device()
    print(f"Using device: {device}")

    model, _ = build_model_from_checkpoint(
        checkpoint_path, vocabs, device, latent_dim_override=latent_dim_override
    )

    samples, skipped = filter_samples_by_decoder_max_len(
        samples, max_seq_len=model.decoder.cfg.max_seq_len
    )
    if skipped:
        print(f"Skipped {skipped} routes exceeding decoder max_seq_len={model.decoder.cfg.max_seq_len}.")
    if not samples:
        raise ValueError("All samples filtered out by decoder max_seq_len.")

    latent_matrix = _extract_latent_matrix(
        model=model, samples=samples, batch_size=batch_size, device=device
    )
    latent_matrix_std = standardize_features(latent_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    cluster_ids = kmeans.fit_predict(latent_matrix_std)

    cluster_cache = Path(cluster_cache_path)
    cluster_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "samples": samples,
            "latent_matrix": latent_matrix_std,
            "cluster_ids": cluster_ids,
            "cluster_centers": kmeans.cluster_centers_,
            "inertia": float(kmeans.inertia_),
            "n_clusters": int(n_clusters),
            "seed": int(seed),
        },
        cluster_cache,
    )
    print(f"Saved clustered latent dataset: {cluster_cache}")

    unique, counts = np.unique(cluster_ids, return_counts=True)
    print("Cluster distribution:")
    for c, n in zip(unique.tolist(), counts.tolist()):
        print(f"  cluster {c}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KMeans clustering on CVAE latent vectors.")
    parser.add_argument("--db-path", type=str, default=str(PROJECT_ROOT / "data/raw/kilter_database.sqlite"))
    parser.add_argument("--cache-path", type=str, default=str(PROJECT_ROOT / "data/preprocessed_routes_cache.pt"))
    parser.add_argument("--checkpoint-path", type=str, default=str(PROJECT_ROOT / "data/route_cvae.pt"))
    parser.add_argument("--cluster-cache-path", type=str, default=str(PROJECT_ROOT / "data/routes_kmeans_original.pt"))
    parser.add_argument("--metadata-source", type=str, default="kilter_board_csv")
    parser.add_argument("--metadata-product-id", type=int, default=1)
    parser.add_argument("--max-routes", type=int, default=5000)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--min-grade", type=float, default=None)
    parser.add_argument("--max-grade", type=float, default=None)
    parser.add_argument("--min-angle", type=float, default=None)
    parser.add_argument("--max-angle", type=float, default=None)
    parser.add_argument("--include-ungraded", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_analysis(
        db_path=args.db_path,
        cache_path=args.cache_path,
        checkpoint_path=args.checkpoint_path,
        cluster_cache_path=args.cluster_cache_path,
        metadata_source=args.metadata_source,
        metadata_product_id=args.metadata_product_id,
        max_routes=args.max_routes,
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        latent_dim_override=args.latent_dim,
        min_grade=args.min_grade,
        max_grade=args.max_grade,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        include_ungraded=args.include_ungraded,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
