"""Style analysis of VAE latent clusters.

Loads a pre-computed cluster cache and produces a diagnostic figure.

Diagnostic panels (is style being learned?):
  · Ridge R² probe — grade/angle leakage and style feature recoverability from z
  · PCA scree + parallel analysis — genuine style dimension count (Horn's method)

Style characterisation panels (what is the style?):
  · UMAP/PCA 2D scatter coloured by cluster
  · PC1 vs PC2 scatter (latent PCA space, coloured by cluster)
  · Feature heatmap per cluster (9 interpretable features, row-normalised)
  · Grade distribution per cluster (confirms grade control in micro-clusters)
  · move_size vs foot_frac scatter — primary style-axis cross-plot

Architecture context (unconditional VAE):
  · z encodes grade + angle + style simultaneously (no AdaLN conditioning)
  · Global cache: grade R² and angle R² should be HIGH (model encoded difficulty)
  · Micro-cluster cache (single grade + angle): grade/angle R² → ~0 (no variance),
    remaining z variation is pure style; style feature R² should be moderate–high
  · With use_absolute_pos=False: x_spread/y_spread will have LOW R² (expected)
  · With use_type_feature=False: type-derived features not in z (expected)

Usage:
    uv run src/data_analysis/style_analysis.py \\
        --cluster-cache-path data/routes_v6_40deg.pt \\
        --output-path data/style_analysis.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Feature vector layout produced by extract_features()
# ---------------------------------------------------------------------------
# Indices 0-4: shape_desc[:5] (identical in both 5D old and 9D new caches)
FEAT_X_SPREAD     = 0   # normalised x std of all holds
FEAT_Y_SPREAD     = 1   # normalised y std of all holds
FEAT_FOOT_FRAC    = 2   # fraction of holds that are feet
FEAT_HAND_DENSITY = 3   # hand count / max_holds (normalised)
FEAT_MOVE_SIZE    = 4   # mean normalised move distance (shape_desc[4])
# Indices 5-8: type fractions from shape_desc[5:9] (9D; use_type_feature=True)
#              zeros for old 5D caches or when use_type_feature=False
FEAT_JUG_FRAC     = 5   # fraction of holds that are jugs
FEAT_SLOPER_FRAC  = 6   # fraction of holds that are slopers
FEAT_CRIMP_FRAC   = 7   # fraction of holds that are crimps
FEAT_PINCH_FRAC   = 8   # fraction of holds that are pinches
# Indices 9-14: derived from tokens
FEAT_GRADE        = 9   # difficulty_average (raw float, e.g. 22.x for V6)
FEAT_NUM_HOLDS    = 10  # total hold count
FEAT_STEP_HEIGHT  = 11  # mean Δy of consecutive hand-hold pairs (upward reach)
FEAT_STEP_X_VAR   = 12  # std Δx of consecutive hand-hold pairs (lateral variance)
FEAT_CROWDING       = 13  # mean kNN distance to nearby holds (hold spacing, all holds)
FEAT_MAX_MOVE       = 14  # max L2 consecutive move magnitude in y-sorted order (artifact of sort)
FEAT_KNN_MOVE_DIST  = 15  # mean dist to 3 nearest hand holds with strictly greater y, per hand hold
                           # permutation-invariant proxy for upward movement difficulty on an
                           # unordered hold set; unlike max_move this is not an artefact of y-sort

FEATURE_LABELS = [
    "x_spread", "y_spread", "foot_frac", "hand_density", "move_size",
    "jug_frac", "sloper_frac", "crimp_frac", "pinch_frac",
    "grade", "num_holds", "step_height", "step_x_var", "crowding", "max_move",
    "knn_move_dist",
]
N_FEATURES = len(FEATURE_LABELS)

# Features shown in the heatmap (grade excluded — it has its own panel)
HEATMAP_INDICES = [
    FEAT_X_SPREAD, FEAT_Y_SPREAD, FEAT_FOOT_FRAC, FEAT_HAND_DENSITY, FEAT_MOVE_SIZE,
    FEAT_JUG_FRAC, FEAT_SLOPER_FRAC, FEAT_CRIMP_FRAC, FEAT_PINCH_FRAC,
    FEAT_NUM_HOLDS, FEAT_STEP_HEIGHT, FEAT_STEP_X_VAR, FEAT_CROWDING, FEAT_KNN_MOVE_DIST,
]
HEATMAP_LABELS = [FEATURE_LABELS[i] for i in HEATMAP_INDICES]

# Style features probed for recoverability from z
STYLE_PROBE_INDICES = [
    FEAT_KNN_MOVE_DIST, FEAT_MOVE_SIZE, FEAT_STEP_X_VAR,
    FEAT_FOOT_FRAC, FEAT_JUG_FRAC, FEAT_CRIMP_FRAC,
    FEAT_STEP_HEIGHT, FEAT_CROWDING, FEAT_NUM_HOLDS,
]
STYLE_PROBE_NAMES = [
    "knn_move_dist", "move_size", "step_x_var",
    "foot_frac", "jug_frac", "crimp_frac",
    "step_height", "crowding", "num_holds",
]

CLUSTER_PALETTE = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231",
    "#911EB4", "#42D4F4", "#F032E6", "#BFEF45",
    "#FABED4", "#469990",
]
NOISE_COLOR = "#AAAAAA"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _grade_to_vgrade(grade: float) -> str:
    return f"V{max(0, int(grade) // 2 - 5)}" if grade >= 10 else f"V0"


def _filter_desc(min_grade, max_grade, min_angle, max_angle) -> str:
    parts = []
    if min_grade is not None or max_grade is not None:
        if min_grade is not None and min_grade == max_grade:
            parts.append(_grade_to_vgrade(float(min_grade)))
        else:
            lo = f"≥{_grade_to_vgrade(float(min_grade))}" if min_grade is not None else ""
            hi = f"≤{_grade_to_vgrade(float(max_grade))}" if max_grade is not None else ""
            parts.append(f"grade {lo}{hi}".strip())
    if min_angle is not None or max_angle is not None:
        if min_angle is not None and min_angle == max_angle:
            parts.append(f"{int(min_angle)}°")
        else:
            lo = f"≥{int(min_angle)}°" if min_angle is not None else ""
            hi = f"≤{int(max_angle)}°" if max_angle is not None else ""
            parts.append(f"angle {lo}{hi}".strip())
    return ", ".join(parts) if parts else "all routes"


def filter_by_grade_angle(
    samples, latent_matrix: np.ndarray, cluster_ids: np.ndarray,
    *, min_grade, max_grade, min_angle, max_angle,
):
    angles = np.array([float(s.angle) for s in samples], dtype=np.float32)
    grades = np.array(
        [float(s.grade) if s.grade is not None else np.nan for s in samples],
        dtype=np.float32,
    )
    mask = np.ones(len(samples), dtype=bool)
    if min_angle is not None:
        mask &= angles >= float(min_angle)
    if max_angle is not None:
        mask &= angles <= float(max_angle)
    if min_grade is not None or max_grade is not None:
        grade_band = np.floor(grades)
        if min_grade is not None:
            mask &= grade_band >= float(min_grade)
        if max_grade is not None:
            mask &= grade_band <= float(max_grade)
    idx = np.flatnonzero(mask)
    return (
        [samples[i] for i in idx],
        latent_matrix[idx],
        cluster_ids[idx],
        int(len(samples) - len(idx)),
    )


def load_cache(path: str):
    import torch
    p = torch.load(path, map_location="cpu", weights_only=False)
    return (
        p["samples"],
        np.asarray(p["latent_matrix"], dtype=np.float32),
        np.asarray(p["cluster_ids"]),
        p,
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(samples) -> np.ndarray:
    """Extract interpretable feature matrix [N, 16] from route samples.

    Features:
        0–4:  shape_desc[:5] — x_spread, y_spread, foot_frac, hand_density, move_size
        5–8:  type fractions  — jug_frac, sloper_frac, crimp_frac, pinch_frac
              (zeros if old 5D cache or use_type_feature=False)
        9:    grade             (raw difficulty_average; 0 if ungraded)
        10:   num_holds
        11:   step_height       (mean Δy between consecutive hand holds)
        12:   step_x_var        (std Δx between consecutive hand holds)
        13:   crowding          (mean kNN distance to nearby holds)
        14:   max_move          (max L2 consecutive move magnitude)
        15:   knn_move_dist     (mean dist to 3 nearest hand holds with greater y)
    """
    rows = []
    for s in samples:
        tok = s.tokens
        sd = tok["shape_desc"].numpy().astype(np.float32)
        # Pad to 9D so old 5D caches still work; type fractions will be 0
        if len(sd) < 9:
            sd = np.concatenate([sd, np.zeros(9 - len(sd), dtype=np.float32)])

        # Movement deltas — mask hand-hold moves (either Δx or Δy non-zero;
        # foot holds and BOS are always 0 on both axes).
        dy_all = tok["delta_y_prev"].numpy()
        dx_all = tok["delta_x_prev"].numpy()
        move_mask   = (dx_all != 0.0) | (dy_all != 0.0)
        dx_moves    = dx_all[move_mask]
        dy_moves    = dy_all[move_mask]
        # Mean L2 consecutive move magnitude (normalised board units) — what the
        # attention pool can derive from hold tokens; this replaces shape_desc[4]
        # (which is all-pairwise distance, not sequential, and lives in the CLS
        # token that is excluded from the attention pool).
        move_l2       = np.sqrt(dx_moves**2 + dy_moves**2) if move_mask.any() else np.array([0.0])
        mean_move_mag = float(move_l2.mean()) if move_mask.any() else 0.0
        max_move_mag  = float(move_l2.max())  if move_mask.any() else 0.0
        mean_delta_y  = float(dy_moves.mean()) if move_mask.any() else 0.0
        std_delta_x   = float(dx_moves.std())  if move_mask.sum() > 1 else 0.0

        # Hold crowding: mean kNN distance (knn_features cols 0:3 = distances, ALL holds)
        if "knn_features" in tok:
            knn = tok["knn_features"].numpy()
            mean_dist = float(knn[:, :3].mean())
        else:
            mean_dist = 0.0

        # knn_move_dist: for each hand hold, mean distance to the 3 closest hand holds
        # with strictly greater y. Averaged over all hand holds that have at least one
        # above them. Normalised by board diagonal (~251.9 units).
        # Permutation-invariant — not an artefact of y-sort ordering.
        # Uses move_mask to identify hand holds (foot holds always have delta=0).
        # Note: the first hand hold (also delta=0) is excluded; minor approximation.
        _BOARD_DIAG = np.sqrt(184.0 ** 2 + 172.0 ** 2)
        x_all = tok["x"].numpy().astype(np.float32)
        y_all = tok["y"].numpy().astype(np.float32)
        xh = x_all[move_mask]
        yh = y_all[move_mask]
        knn_move_dist = 0.0
        if len(xh) >= 2:
            _K = 3
            per_hold = []
            for _i in range(len(xh)):
                above = yh > yh[_i]
                if not above.any():
                    continue
                d = np.sqrt((xh[above] - xh[_i]) ** 2 + (yh[above] - yh[_i]) ** 2)
                per_hold.append(np.sort(d)[: min(_K, len(d))].mean() / _BOARD_DIAG)
            knn_move_dist = float(np.mean(per_hold)) if per_hold else 0.0

        # Build row — move_size (index 4) is now mean consecutive L2 move magnitude
        # from hold tokens, overriding the CLS-only shape_desc[4] value.
        sd_row = sd[:9].copy()
        sd_row[4] = mean_move_mag  # replace pairwise-spread with sequential move magnitude

        row = np.concatenate([
            sd_row,   # x_spread, y_spread, foot_frac, hand_density, move_size(fixed), jug, sloper, crimp, pinch
            [float(s.grade) if s.grade is not None else 0.0],
            [float(s.num_holds)],
            [mean_delta_y, std_delta_x, mean_dist, max_move_mag, knn_move_dist],
        ])
        rows.append(row)
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_ridge_probe(
    latent_matrix: np.ndarray,
    samples,
    features: np.ndarray,
) -> dict:
    """Probe z for grade, angle, and style feature recoverability (5-fold CV).

    Uses Ridge regression for most features (measures linear recoverability).
    Uses a small MLP for features that may be encoded non-linearly in z (max_move):
    Ridge near-zero R² on those features would be misleading if the information is
    present but on a curved manifold — MLP catches both linear and non-linear encoding.

    Returns dict:
        grade_r2, grade_std   — None if grade has no variance (grade-fixed micro-cluster)
        angle_r2, angle_std   — None if angle has no variance (single-angle cache)
        style_r2s             — dict[feature_name → float | None]
                                 names suffixed with " (MLP)" where MLP was used
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score, KFold

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    grades = np.array(
        [float(s.grade) if s.grade is not None else np.nan for s in samples],
        dtype=np.float64,
    )
    angles = np.array([float(s.angle) for s in samples], dtype=np.float64)

    def _probe_ridge(X: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
        valid = np.isfinite(y)
        if valid.sum() < 20:
            return None, None
        y_v = y[valid]
        X_v = X[valid]
        if y_v.var() < 1e-4:
            return None, None  # no variance — probe is meaningless
        cv_k = min(5, valid.sum() // 4)
        kf = KFold(n_splits=cv_k, shuffle=True, random_state=42)
        ridge = RidgeCV(alphas=alphas)
        scores = cross_val_score(ridge, X_v, y_v, cv=kf, scoring="r2")
        return float(scores.mean()), float(scores.std())

    def _probe_mlp(X: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
        """5-fold CV R² using a small MLP — catches non-linear encoding in z."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        valid = np.isfinite(y)
        if valid.sum() < 20:
            return None, None
        y_v = y[valid]
        X_v = X[valid]
        if y_v.var() < 1e-4:
            return None, None
        cv_k = min(5, valid.sum() // 4)
        kf = KFold(n_splits=cv_k, shuffle=True, random_state=42)
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                random_state=42,
                alpha=0.001,
            ),
        )
        scores = cross_val_score(mlp, X_v, y_v, cv=kf, scoring="r2")
        return float(scores.mean()), float(scores.std())

    # Features where non-linear encoding is plausible (max/argmax statistics that
    # soft attention pooling may encode on a curved manifold rather than linearly).
    MLP_PROBE_FEATURES = {"max_move"}

    grade_r2, grade_std = _probe_ridge(latent_matrix, grades)
    angle_r2, angle_std = _probe_ridge(latent_matrix, angles)

    style_r2s: dict[str, float | None] = {}
    for idx, name in zip(STYLE_PROBE_INDICES, STYLE_PROBE_NAMES):
        feat = features[:, idx].astype(np.float64)
        if name in MLP_PROBE_FEATURES:
            r2, _ = _probe_mlp(latent_matrix, feat)
            style_r2s[f"{name} (MLP)"] = r2
        else:
            r2, _ = _probe_ridge(latent_matrix, feat)
            style_r2s[name] = r2

    return {
        "grade_r2": grade_r2, "grade_std": grade_std,
        "angle_r2": angle_r2, "angle_std": angle_std,
        "style_r2s": style_r2s,
    }


def parallel_analysis(
    latent_matrix: np.ndarray,
    n_permutations: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Horn's parallel analysis: compare real PCA eigenvalues to permutation noise.

    Returns:
        real_ev      [n_pcs]  actual eigenvalues
        real_evr     [n_pcs]  actual explained variance ratios
        noise_p95    [n_pcs]  95th-percentile eigenvalue from permuted data
        n_genuine    int      number of PCs exceeding noise p95
    """
    from sklearn.decomposition import PCA

    n, d = latent_matrix.shape
    n_pcs = min(d, n - 1)
    rng = np.random.default_rng(seed)

    pca = PCA(n_components=n_pcs)
    pca.fit(latent_matrix)
    real_ev  = pca.explained_variance_
    real_evr = pca.explained_variance_ratio_

    noise_evs = np.zeros((n_permutations, n_pcs))
    for i in range(n_permutations):
        perm = np.column_stack([rng.permutation(latent_matrix[:, j]) for j in range(d)])
        pca_n = PCA(n_components=n_pcs)
        pca_n.fit(perm)
        noise_evs[i] = pca_n.explained_variance_

    noise_p95  = np.percentile(noise_evs, 95, axis=0)
    n_genuine  = int(np.sum(real_ev > noise_p95))

    return real_ev, real_evr, noise_p95, n_genuine


def compute_pc_analysis(
    latent_matrix: np.ndarray,
    features: np.ndarray,
    n_pcs: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project z to PCA space and compute Pearson correlations with interpretable features.

    Returns:
        pc_scores    [N, n_pcs]            PC scores per route
        ev_ratio     [n_pcs]               Explained variance ratio
        cors         [n_pcs, N_FEATURES]   Pearson r between PC_i and feature_j
    """
    from sklearn.decomposition import PCA

    n_pcs = min(n_pcs, latent_matrix.shape[1], latent_matrix.shape[0] - 1)
    pca = PCA(n_components=n_pcs)
    pc_scores = pca.fit_transform(latent_matrix)
    ev_ratio  = pca.explained_variance_ratio_

    cors = np.zeros((n_pcs, N_FEATURES))
    for i in range(n_pcs):
        for j in range(N_FEATURES):
            feat = features[:, j].astype(float)
            valid = np.isfinite(feat) & np.isfinite(pc_scores[:, i])
            if valid.sum() > 10 and feat[valid].std() > 1e-8:
                r = np.corrcoef(pc_scores[valid, i], feat[valid])[0, 1]
                cors[i, j] = 0.0 if np.isnan(r) else r

    return pc_scores, ev_ratio, cors


# ---------------------------------------------------------------------------
# Cluster stats
# ---------------------------------------------------------------------------

def cluster_stats(features: np.ndarray, cluster_ids: np.ndarray) -> dict[int, dict]:
    unique = sorted(int(c) for c in np.unique(cluster_ids) if c >= 0)
    stats = {}
    for c in unique:
        mask = cluster_ids == c
        f = features[mask]
        stats[c] = {
            "n":      int(mask.sum()),
            "mean":   f.mean(axis=0),
            "median": np.median(f, axis=0),
            "std":    f.std(axis=0),
            "q25":    np.percentile(f, 25, axis=0),
            "q75":    np.percentile(f, 75, axis=0),
        }
    return stats


# ---------------------------------------------------------------------------
# 2-D projection for scatter
# ---------------------------------------------------------------------------

def project_2d(
    latent_matrix: np.ndarray,
    *,
    use_umap: bool = True,
    n_neighbors: int = 50,
    min_dist: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, str]:
    """Project latent matrix to 2D; falls back to PCA if umap is unavailable."""
    if use_umap:
        try:
            from umap import UMAP
            proj = UMAP(
                n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                random_state=seed,
            ).fit_transform(latent_matrix)
            return proj, "UMAP"
        except ImportError:
            print("Warning: umap-learn not installed; using PCA for 2D projection.")

    from sklearn.decomposition import PCA
    pca  = PCA(n_components=2, random_state=seed)
    proj = pca.fit_transform(latent_matrix)
    ev   = pca.explained_variance_ratio_
    return proj, f"PCA  (PC1 {ev[0]:.1%} · PC2 {ev[1]:.1%})"


# ---------------------------------------------------------------------------
# Individual panel helpers
# ---------------------------------------------------------------------------

def _cluster_colors(unique_clusters: list[int]) -> dict[int, str]:
    return {c: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, c in enumerate(unique_clusters)}


def _plot_scatter(
    ax,
    projected_2d: np.ndarray,
    cluster_ids: np.ndarray,
    stats: dict,
    colors: dict,
    proj_label: str,
    method: str,
) -> None:
    noise_mask = cluster_ids == -1
    if noise_mask.any():
        ax.scatter(
            projected_2d[noise_mask, 0], projected_2d[noise_mask, 1],
            c=NOISE_COLOR, s=6, alpha=0.25, linewidths=0,
            label=f"noise (n={noise_mask.sum()})", rasterized=True,
        )
    for c, st in stats.items():
        mask = cluster_ids == c
        ax.scatter(
            projected_2d[mask, 0], projected_2d[mask, 1],
            c=colors[c], s=12, alpha=0.55, linewidths=0,
            label=f"C{c} (n={st['n']})",
            rasterized=True,
        )
        cx, cy = projected_2d[mask, 0].mean(), projected_2d[mask, 1].mean()
        ax.text(
            cx, cy, f"C{c}", fontsize=9, fontweight="bold", color=colors[c],
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=colors[c], alpha=0.7),
        )
    ax.set_title(f"{proj_label} — {method.upper()} clusters in latent space", fontsize=11)
    ax.set_xlabel(proj_label.split()[0] + "-1")
    ax.set_ylabel(proj_label.split()[0] + "-2")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.8, ncol=2)
    ax.tick_params(labelsize=8)


def _plot_diagnostic(
    ax,
    ridge_results: dict,
    n_genuine: int,
    latent_dim: int,
    n_samples: int,
    n_clusters: int,
) -> None:
    """Render the style-learning diagnostic summary panel."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def txt(x, y, s, **kw):
        ax.text(x, y, s, transform=ax.transAxes, va="top", **kw)

    def _r2_color(r2: float | None, *, high_is_good: bool) -> str:
        if r2 is None:
            return "#888888"
        if high_is_good:
            return "#2CA02C" if r2 > 0.3 else ("#FF7F0E" if r2 > 0.1 else "#AAAAAA")
        else:
            return "#2CA02C" if r2 < 0.15 else ("#FF7F0E" if r2 < 0.5 else "#D62728")

    def _r2_tag(r2: float | None, *, high_is_good: bool) -> str:
        if r2 is None:
            return "N/A"
        if high_is_good:
            return f"{r2:.3f}  {'★★' if r2 > 0.3 else ('★' if r2 > 0.1 else '~')}"
        else:
            return f"{r2:.3f}  {'✓' if r2 < 0.15 else ('!' if r2 < 0.5 else '✗✗')}"

    y = 0.97
    dh = 0.068  # line height

    txt(0.5, y, "Style Learning Diagnostics", ha="center", fontsize=11,
        fontweight="bold"); y -= dh * 0.7
    txt(0.5, y, "unconditional VAE — z encodes grade + angle + style", ha="center",
        fontsize=7.5, color="#666666", style="italic"); y -= dh * 0.55
    ax.axhline(y + dh * 0.2, color="#CCCCCC", lw=0.8)
    y -= dh * 0.1

    # Header info
    txt(0.02, y, f"N routes: {n_samples}   latent dim: {latent_dim}   clusters: {n_clusters}",
        fontsize=8, color="#333333"); y -= dh * 1.1

    # --- Confound probes ---
    txt(0.02, y, "Confound probes  (Ridge R², 5-fold CV):", fontsize=8.5,
        fontweight="bold", color="#222222"); y -= dh

    grade_r2 = ridge_results.get("grade_r2")
    angle_r2 = ridge_results.get("angle_r2")
    # In a micro-cluster grade/angle have no variance → None means "fixed, good"
    grade_note = "  (fixed ✓)" if grade_r2 is None else ""
    angle_note = "  (fixed ✓)" if angle_r2 is None else ""

    txt(0.04, y, f"Grade R² from z:", fontsize=8, color="#333333")
    txt(0.98, y, _r2_tag(grade_r2, high_is_good=False) + grade_note, ha="right",
        fontsize=8, fontweight="bold",
        color=_r2_color(grade_r2, high_is_good=False) if grade_r2 is not None else "#2CA02C")
    y -= dh

    txt(0.04, y, f"Angle R² from z:", fontsize=8, color="#333333")
    txt(0.98, y, _r2_tag(angle_r2, high_is_good=False) + angle_note, ha="right",
        fontsize=8, fontweight="bold",
        color=_r2_color(angle_r2, high_is_good=False) if angle_r2 is not None else "#2CA02C")
    y -= dh * 0.4
    ax.axhline(y + dh * 0.15, color="#EEEEEE", lw=0.6)
    y -= dh * 0.4

    # --- Genuine PCs ---
    txt(0.02, y, f"Genuine style dims  (parallel analysis p95):", fontsize=8.5,
        fontweight="bold", color="#222222"); y -= dh
    dim_color = "#2CA02C" if n_genuine >= 2 else ("#FF7F0E" if n_genuine == 1 else "#D62728")
    txt(0.04, y, f"PCs > noise:", fontsize=8, color="#333333")
    txt(0.98, y, f"{n_genuine} of {latent_dim}", ha="right", fontsize=8,
        fontweight="bold", color=dim_color)
    y -= dh * 0.4
    ax.axhline(y + dh * 0.15, color="#EEEEEE", lw=0.6)
    y -= dh * 0.4

    # --- Style recoverability ---
    txt(0.02, y, "Style recoverability  (z → feature R²):", fontsize=8.5,
        fontweight="bold", color="#222222"); y -= dh
    txt(0.02, y, "★★ >0.3  ★ >0.1  ~ weak  (high_is_good)", fontsize=7,
        color="#888888"); y -= dh * 0.9
    for name, r2 in ridge_results.get("style_r2s", {}).items():
        txt(0.04, y, f"{name}:", fontsize=8, color="#333333")
        txt(0.98, y, _r2_tag(r2, high_is_good=True), ha="right", fontsize=8,
            fontweight="bold", color=_r2_color(r2, high_is_good=True))
        y -= dh * 0.88
    ax.set_facecolor("#FAFAFA")

    # Attribute context note
    if y > 0.04:
        txt(0.02, max(y, 0.02),
            "Note: use_absolute_pos=False → x/y_spread low R² is expected.\n"
            "use_type_feature=True → jug/crimp/sloper recoverability expected.\n"
            "move_size, foot_frac, step_height, crowding should be HIGH.",
            fontsize=6.5, color="#AAAAAA", wrap=True)


def _plot_scree(
    ax,
    real_evr: np.ndarray,
    noise_p95_evr: np.ndarray,
    n_genuine: int,
) -> None:
    """PCA scree plot with parallel-analysis noise threshold."""
    n = len(real_evr)
    x = np.arange(1, n + 1)
    colors_bar = ["#2CA02C" if i < n_genuine else "#CCCCCC" for i in range(n)]
    ax.bar(x, real_evr * 100, color=colors_bar, alpha=0.85, zorder=2)

    # Noise p95 line (convert eigenvalue to equivalent evr for plotting)
    # noise_p95 is in eigenvalue units; convert to fraction of total variance
    noise_evr = noise_p95_evr / noise_p95_evr.sum() if noise_p95_evr.sum() > 0 else noise_p95_evr
    ax.plot(x, noise_evr * 100, "r--", lw=1.5, alpha=0.75, label="noise p95", zorder=3)
    ax.scatter(x, noise_evr * 100, c="red", s=14, alpha=0.6, zorder=4)

    ax.axvline(n_genuine + 0.5, color="#2CA02C", lw=1.2, ls=":", alpha=0.8)
    ax.text(n_genuine + 0.6, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 5,
            f"{n_genuine} genuine", fontsize=7.5, color="#2CA02C")

    ax.set_xlabel("PC", fontsize=8)
    ax.set_ylabel("Variance explained (%)", fontsize=8)
    ax.set_title("PCA scree  +  parallel-analysis noise\n(green bars = genuine style PCs)", fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.8)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0.4, min(n, 16) + 0.6)
    ax.grid(axis="y", alpha=0.3)


def _plot_heatmap(
    ax,
    stats: dict,
    colors: dict,
) -> None:
    """Feature heatmap: clusters × 13 interpretable features (row-normalised)."""
    unique = sorted(stats.keys())
    n_cl = len(unique)
    hm = np.array([stats[c]["mean"][HEATMAP_INDICES] for c in unique])  # [K, 9]
    hm_min = hm.min(axis=0, keepdims=True)
    hm_max = hm.max(axis=0, keepdims=True)
    hm_norm = (hm - hm_min) / np.maximum(hm_max - hm_min, 1e-8)

    im = ax.imshow(hm_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(HEATMAP_LABELS)))
    ax.set_xticklabels(HEATMAP_LABELS, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(n_cl))
    ax.set_yticklabels([f"C{c}" for c in unique], fontsize=7.5)
    ax.set_title("Feature profile per cluster\n(column-normalised: green=high, red=low)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="relative value")
    for i, c in enumerate(unique):
        for j, fi in enumerate(HEATMAP_INDICES):
            ax.text(j, i, f"{stats[c]['mean'][fi]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")


def _plot_pc_scatter(
    ax,
    pc_scores: np.ndarray,
    pc_ev_ratio: np.ndarray,
    cluster_ids: np.ndarray,
    stats: dict,
    colors: dict,
) -> None:
    """PC1 vs PC2 scatter colored by cluster (latent PCA space)."""
    noise_mask = cluster_ids == -1
    if noise_mask.any():
        ax.scatter(pc_scores[noise_mask, 0], pc_scores[noise_mask, 1],
                   c=NOISE_COLOR, s=4, alpha=0.2, linewidths=0, rasterized=True)
    for c, st in stats.items():
        mask = cluster_ids == c
        ax.scatter(pc_scores[mask, 0], pc_scores[mask, 1],
                   c=colors[c], s=8, alpha=0.5, linewidths=0,
                   label=f"C{c}", rasterized=True)
        cx, cy = pc_scores[mask, 0].mean(), pc_scores[mask, 1].mean()
        ax.text(cx, cy, f"C{c}", fontsize=8, fontweight="bold", color=colors[c],
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=colors[c], alpha=0.7))

    ev1 = pc_ev_ratio[0] * 100 if len(pc_ev_ratio) > 0 else 0
    ev2 = pc_ev_ratio[1] * 100 if len(pc_ev_ratio) > 1 else 0
    ax.set_xlabel(f"PC1  ({ev1:.1f}% var)", fontsize=8)
    ax.set_ylabel(f"PC2  ({ev2:.1f}% var)", fontsize=8)
    ax.set_title("PC1 vs PC2 (latent PCA space)\nClusters should separate if style is learned", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)


def _plot_grade_box(
    ax,
    features: np.ndarray,
    cluster_ids: np.ndarray,
    stats: dict,
    colors: dict,
) -> None:
    """Grade distribution box plot per cluster."""
    unique = sorted(stats.keys())
    grade_data = []
    for c in unique:
        g = features[cluster_ids == c, FEAT_GRADE]
        g = g[g > 0]  # exclude ungraded (stored as 0)
        grade_data.append(g if len(g) > 0 else np.array([0.0]))

    bp = ax.boxplot(grade_data, patch_artist=True, notch=False,
                    flierprops=dict(marker=".", markersize=2, alpha=0.4))
    for patch, c in zip(bp["boxes"], unique):
        patch.set_facecolor(colors[c]); patch.set_alpha(0.75)

    ax.set_xticklabels([f"C{c}" for c in unique], fontsize=7)
    ax.set_title("Grade distribution per cluster\n(ideal: overlapping in micro-cluster)", fontsize=9)
    ax.set_ylabel("difficulty_average", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    # V-grade reference lines
    for gval, vlabel in [(10, "V0"), (14, "V2"), (18, "V4"),
                         (22, "V6"), (26, "V8"), (30, "V10")]:
        ax.axhline(gval, color="grey", lw=0.6, ls="--", alpha=0.45)
        ax.text(len(unique) + 0.15, gval, vlabel, fontsize=6.5, color="grey", va="center")


def _plot_style_scatter(
    ax,
    features: np.ndarray,
    cluster_ids: np.ndarray,
    stats: dict,
    colors: dict,
) -> None:
    """move_size vs foot_frac scatter — the two primary style axes."""
    for c, st in stats.items():
        mask = cluster_ids == c
        ax.scatter(features[mask, FEAT_MOVE_SIZE], features[mask, FEAT_FOOT_FRAC],
                   c=colors[c], s=5, alpha=0.15, linewidths=0, rasterized=True)
    for c, st in stats.items():
        mv = st["mean"][FEAT_MOVE_SIZE]
        ff = st["mean"][FEAT_FOOT_FRAC]
        ax.scatter(mv, ff, s=180, color=colors[c], zorder=5, edgecolors="black", lw=0.8)
        ax.text(mv + 0.003, ff + 0.005, f"C{c}",
                fontsize=6.5, fontweight="bold", color=colors[c])
    ax.set_xlabel("move_size  (tight=0 → full-board reach=1)", fontsize=8)
    ax.set_ylabel("foot_fraction  (power=0 → technical=1)", fontsize=8)
    ax.set_title("Primary style axes\n(move geometry vs. foot usage)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def make_figure(
    projected_2d: np.ndarray,
    proj_label: str,
    pc_scores: np.ndarray,
    pc_ev_ratio: np.ndarray,
    real_evr: np.ndarray,
    noise_p95_evr: np.ndarray,
    n_genuine: int,
    features: np.ndarray,
    cluster_ids: np.ndarray,
    stats: dict,
    ridge_results: dict,
    *,
    latent_dim: int,
    method: str = "kmeans",
    title_suffix: str = "",
    output_path: str,
) -> None:
    unique_clusters = sorted(stats.keys())
    colors = _cluster_colors(unique_clusters)
    n_samples = len(cluster_ids)

    fig = plt.figure(figsize=(24, 20), constrained_layout=True)
    fig.suptitle(
        f"VAE Style Analysis{title_suffix}",
        fontsize=15, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.8, 1.3, 1.1])

    ax_scatter  = fig.add_subplot(gs[0, :2])   # UMAP/PCA scatter (wide)
    ax_diag     = fig.add_subplot(gs[0, 2])    # diagnostic summary
    ax_scree    = fig.add_subplot(gs[1, 0])    # scree + parallel analysis
    ax_heatmap  = fig.add_subplot(gs[1, 1])    # feature heatmap
    ax_pcscat   = fig.add_subplot(gs[1, 2])    # PC1 vs PC2 scatter
    ax_grade    = fig.add_subplot(gs[2, :2])   # grade box plots (wide, fills gap left by removed panel)
    ax_style    = fig.add_subplot(gs[2, 2])    # move_size vs foot_frac

    _plot_scatter(ax_scatter, projected_2d, cluster_ids, stats, colors,
                  proj_label, method)

    _plot_diagnostic(ax_diag, ridge_results,
                     n_genuine, latent_dim, n_samples, len(unique_clusters))

    _plot_scree(ax_scree, real_evr, noise_p95_evr, n_genuine)

    _plot_heatmap(ax_heatmap, stats, colors)

    _plot_pc_scatter(ax_pcscat, pc_scores, pc_ev_ratio, cluster_ids, stats, colors)

    _plot_grade_box(ax_grade, features, cluster_ids, stats, colors)

    _plot_style_scatter(ax_style, features, cluster_ids, stats, colors)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_cluster_report(
    stats: dict,
    ridge_results: dict,
    n_genuine: int,
    n_noise: int,
    pc_cors: np.ndarray,
    pc_ev_ratio: np.ndarray | None = None,
) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("VAE STYLE ANALYSIS REPORT")
    print(sep)
    print(f"Noise points (label=-1): {n_noise}")
    print()

    # --- Diagnostics ---
    print("LEARNING DIAGNOSTICS")
    print("-" * 40)
    grade_r2 = ridge_results.get("grade_r2")
    angle_r2 = ridge_results.get("angle_r2")
    print(f"  Grade R² from z : {f'{grade_r2:.4f}' if grade_r2 is not None else 'N/A (no grade variance — grade fixed)'}")
    print(f"  Angle R² from z : {f'{angle_r2:.4f}' if angle_r2 is not None else 'N/A (no angle variance — angle fixed)'}")
    print(f"  Genuine PCs     : {n_genuine}")
    print()
    print("  Style feature recoverability (z → feature R²):")
    for name, r2 in ridge_results.get("style_r2s", {}).items():
        flag = "★★" if (r2 or 0) > 0.3 else ("★" if (r2 or 0) > 0.1 else "~")
        print(f"    {name:<14}: {f'{r2:.4f}  {flag}' if r2 is not None else 'N/A'}")

    # --- Cluster table ---
    print()
    print("CLUSTER SUMMARY")
    print("-" * 40)
    hdr = (
        f"{'C':<4} {'N':>5}  "
        f"{'Grade':>7} {'Holds':>6} {'Move':>6} "
        f"{'Foot%':>6} {'StepH':>6} {'Crowd':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for c, st in sorted(stats.items()):
        m = st["mean"]
        print(
            f"C{c:<3} {st['n']:>5}  "
            f"{m[FEAT_GRADE]:>7.1f} {m[FEAT_NUM_HOLDS]:>6.1f} {m[FEAT_MOVE_SIZE]:>6.3f} "
            f"{m[FEAT_FOOT_FRAC]*100:>5.1f}% {m[FEAT_STEP_HEIGHT]:>6.3f} {m[FEAT_CROWDING]:>6.3f}"
        )

    # --- PC interpretation ---
    print()
    # Show all PCs with >5% explained variance; fall back to first 4 if ev_ratio unavailable
    if pc_ev_ratio is not None:
        pc_indices = [i for i, ev in enumerate(pc_ev_ratio) if ev >= 0.05]
        if not pc_indices:
            pc_indices = list(range(min(4, pc_cors.shape[0])))
        ev_str = "PCs > 5% variance"
    else:
        pc_indices = list(range(min(4, pc_cors.shape[0])))
        ev_str = "first 4 PCs"
    print(f"PC FEATURE CORRELATIONS  (|r| > 0.20 shown, {ev_str})")
    print("-" * 40)
    for i in pc_indices:
        if i >= pc_cors.shape[0]:
            break
        ev_pct = f"  {pc_ev_ratio[i]*100:.1f}% var" if pc_ev_ratio is not None else ""
        genuine_tag = "  [genuine]" if i < n_genuine else ""
        sig = [(FEATURE_LABELS[j], pc_cors[i, j])
               for j in range(N_FEATURES) if abs(pc_cors[i, j]) > 0.20]
        sig.sort(key=lambda x: -abs(x[1]))
        desc = "  ".join(f"{n}({r:+.2f})" for n, r in sig[:6])
        print(f"  PC{i+1}{ev_pct}{genuine_tag}: {desc if desc else '(no strong correlations)'}")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VAE cluster style analysis."
    )
    parser.add_argument("--cluster-cache-path", type=str,
                        default=str(PROJECT_ROOT / "data/routes_clustered.pt"))
    parser.add_argument("--output-path", type=str,
                        default=str(PROJECT_ROOT / "data/style_analysis.png"))
    parser.add_argument("--max-routes", type=int, default=None)
    parser.add_argument("--min-grade", type=float, default=None,
                        help="Keep routes with floor(grade) >= this (e.g. 22 for V6).")
    parser.add_argument("--max-grade", type=float, default=None,
                        help="Keep routes with floor(grade) <= this (e.g. 22 for V6).")
    parser.add_argument("--min-angle", type=float, default=None)
    parser.add_argument("--max-angle", type=float, default=None)
    parser.add_argument("--no-umap", action="store_true",
                        help="Use PCA for 2D scatter instead of UMAP.")
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--n-permutations", type=int, default=200,
                        help="Number of permutations for parallel analysis (default 200).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading cache: {args.cluster_cache_path}")
    samples, latent_matrix, cluster_ids, payload = load_cache(args.cluster_cache_path)

    if args.max_routes is not None:
        samples       = samples[:args.max_routes]
        latent_matrix = latent_matrix[:args.max_routes]
        cluster_ids   = cluster_ids[:args.max_routes]

    any_filter = any(x is not None for x in [
        args.min_grade, args.max_grade, args.min_angle, args.max_angle,
    ])
    if any_filter:
        samples, latent_matrix, cluster_ids, n_filtered = filter_by_grade_angle(
            samples, latent_matrix, cluster_ids,
            min_grade=args.min_grade, max_grade=args.max_grade,
            min_angle=args.min_angle, max_angle=args.max_angle,
        )
        if n_filtered:
            print(f"  Filtered out {n_filtered} routes by grade/angle → {len(samples)} remaining")
        if not samples:
            raise SystemExit("No routes remain after grade/angle filter.")

    print(f"  {len(samples)} routes, {latent_matrix.shape[1]}D latent space")
    unique, counts = np.unique(cluster_ids, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        print(f"  {'noise' if u == -1 else f'cluster {u}'}: {c}")

    print("Extracting interpretable features...")
    features = extract_features(samples)

    print(f"Running parallel analysis ({args.n_permutations} permutations)...")
    real_ev, real_evr, noise_p95_ev, n_genuine = parallel_analysis(
        latent_matrix, n_permutations=args.n_permutations, seed=args.seed,
    )
    # Convert noise p95 eigenvalues to explained-variance-ratio scale for plotting
    total_var = real_ev.sum()
    noise_p95_evr = noise_p95_ev / total_var if total_var > 0 else noise_p95_ev
    print(f"  Genuine style PCs (p95): {n_genuine} of {latent_matrix.shape[1]}")

    print("Computing PC feature correlations (all PCs >5% variance)...")
    # Compute all PCs so we can filter by the >5% threshold in the plot/report
    n_pcs_to_compute = min(latent_matrix.shape[1], latent_matrix.shape[0] - 1)
    pc_scores, pc_ev_ratio, pc_cors = compute_pc_analysis(
        latent_matrix, features, n_pcs=n_pcs_to_compute,
    )

    print("Computing cluster statistics...")
    stats = cluster_stats(features, cluster_ids)

    print("Running Ridge R² probes...")
    ridge_results = compute_ridge_probe(latent_matrix, samples, features)
    grade_r2 = ridge_results.get("grade_r2")
    angle_r2 = ridge_results.get("angle_r2")
    print(f"  Grade R²: {f'{grade_r2:.4f}' if grade_r2 is not None else 'N/A (no grade variance)'}")
    print(f"  Angle R²: {f'{angle_r2:.4f}' if angle_r2 is not None else 'N/A (no angle variance)'}")

    n_noise = int((cluster_ids == -1).sum())
    print_cluster_report(
        stats, ridge_results,
        n_genuine, n_noise, pc_cors, pc_ev_ratio=pc_ev_ratio,
    )

    print("Computing 2D projection for scatter...")
    projected_2d, proj_label = project_2d(
        latent_matrix,
        use_umap=not args.no_umap,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )

    method = payload.get("method", "kmeans")
    fdesc  = _filter_desc(args.min_grade, args.max_grade, args.min_angle, args.max_angle)
    title_suffix = f"  —  {method.upper()}, {len(stats)} clusters  |  {fdesc}"

    print("Generating figure...")
    make_figure(
        projected_2d, proj_label,
        pc_scores, pc_ev_ratio,
        real_evr, noise_p95_evr, n_genuine,
        features, cluster_ids, stats,
        ridge_results,
        latent_dim=latent_matrix.shape[1],
        method=method,
        title_suffix=title_suffix,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
