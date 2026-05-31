# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All scripts are run with `uv run`. The `src/` directory is not an installed package — every script inserts `src/` into `sys.path` at the top.

```bash
# Train CVAE (locally — usually done on Colab via notebooks/train_route_cvae.ipynb)
uv run src/model_training/train_route_cvae.py \
  --latent-dim 16 --free-bits 0.5 --kl-warmup-epochs 5 \
  --grade-adversary-weight 2.0 --grade-adversary-alpha 2.0 --decoder-z-memory-tokens 8

# Build a KMeans cluster cache (required before visualization)
uv run src/data_analysis/routes_cluster.py \
  --method kmeans --n-clusters 6 --cluster-cache-path data/routes_kmeans_original.pt

# Build a filtered cluster cache (e.g. V6 at 40°)
uv run src/data_analysis/routes_cluster.py \
  --method kmeans --n-clusters 6 --min-grade 22 --max-grade 22 --min-angle 40 --max-angle 40 \
  --cluster-cache-path data/routes_v6_40deg_kmeans.pt

# HDBSCAN with UMAP pre-reduction (better for arc-shaped manifolds)
uv run src/data_analysis/routes_cluster.py \
  --method hdbscan --pre-reduce --pre-reduce-method umap --pre-reduce-dims 5 \
  --umap-n-neighbors 50 --umap-min-dist 0.0 \
  --hdbscan-min-cluster-size 200 --hdbscan-min-samples 15 --no-noise \
  --cluster-cache-path data/routes_hdbscan_umap.pt

# Visualize from an existing cluster cache (PCA / UMAP / t-SNE)
uv run src/data_analysis/routes_visualize.py --method pca  --cluster-cache-path data/routes_kmeans_original.pt --show
uv run src/data_analysis/routes_visualize.py --method umap --cluster-cache-path data/routes_hdbscan_umap.pt --show
uv run src/data_analysis/routes_visualize.py --method tsne --cluster-cache-path data/routes_kmeans_original.pt --show

# Visualize a single route
uv run src/route_visualizer.py --climb-name "Alberts dream"
```

There is no test suite. Smoke-test changes by running the affected script or with a short `uv run python3 -c "..."` import check.

**Grade filtering note:** Kilter grades are stored as floats (e.g., `22.013` for V6). Use integer bounds with `--min-grade` / `--max-grade`; the filtering code applies `np.floor()` so `--min-grade 22 --max-grade 22` correctly captures all V6 routes.

## Architecture

The pipeline has five layers:

```
SQLite DB (kilter_database.sqlite)
    └─ database_interfaces/board_lib_interface.py   ← thin DB wrapper
        └─ data_preprocessing/route_preprocessing.py  ← builds RouteSample list + vocabs
            └─ model_training/train_route_cvae.py      ← trains CVAE, saves checkpoint
                └─ data_analysis/routes_cluster.py   ← extracts latents, KMeans or HDBSCAN
                    └─ routes_visualize.py             ← PCA / UMAP / t-SNE visualisation
src/route_visualizer.py  ← standalone board image renderer (also called on click from PCA/t-SNE)
```

### Data flow and caches

`build_training_samples_from_db()` queries the DB, filters by quality/ascensionist count, and returns `(list[RouteSample], RouteVocabBundle)`. This is expensive; results are cached to `data/preprocessed_routes_cache.pt`. Pass `--rebuild-cache` to regenerate.

**UUID deduplication**: The DB join (`climbs × climb_stats`) produces one row per (UUID, angle). Since the encoder does not see angle, all angle variants of the same route produce identical z vectors. `_load_raw_routes` deduplicates to one entry per UUID, keeping the (UUID, angle) pair with the highest `quality_average × ascensionist_count`. Any existing `preprocessed_routes_cache.pt` built before this change contains duplicates and **must be rebuilt** (`--rebuild-cache`) before the next training run.

The cluster cache (`.pt` file produced by `routes_cluster.py`) contains standardised latent vectors + cluster labels and is the sole input to `routes_visualize.py`. Visualization does **not** load the model — it only reads the pre-computed cache. This keeps clustering consistent across views. HDBSCAN caches include a `method` key and may have noise points (label `−1`). The cluster script also deduplicates by UUID at extraction time, so old caches can still be visualised correctly even if rebuilt with the old preprocessing.

### CVAE model (`src/model_training/`)

```
RouteConditionalVAE
  ├── RouteTransformerEncoder   — bidirectional transformer, outputs route_embedding [B, d_model]
  ├── RouteVAEBottleneck        — MLP → (mu, logvar) → reparameterised z [B, latent_dim]
  └── RouteTransformerDecoder   — autoregressive transformer, cross-attends z via memory tokens
```

Key design decisions:
- **`HoldTokenEmbedder`** (in `model_utils.py`) is instantiated inside both encoder and decoder configs but is logically identical — each hold becomes one token by concatenating independent embeddings for type/role/hole_id (learned tables) and x/y (sinusoidal) then projecting to `d_model`. The `function` feature was removed (it was 100% derivable from `type`: `foot` type → function `foot`, everything else → `both`).
- **Encoder sees condition** (`encoder_use_condition=True`): grade/angle are injected into both encoder (via AdaLN) and decoder. The encoder uses grade context to normalise hold-token representations so z learns grade-relative style ("big move for this grade") rather than absolute spatial features that are grade-confounded. The adversary ensures residual grade doesn't leak into z. If z collapses onto grade, increase `--grade-adversary-weight` / `--grade-adversary-alpha`.
- **Post-only AdaLN**: the decoder has a single AdaLN after the transformer stack (not before). A pre-decoder AdaLN was removed because it let the decoder reconstruct from angle/grade alone, causing posterior collapse.
- **Free bits** (`--free-bits 0.5`): each latent dimension's KL is clamped from below at 0.5 nats. With `latent_dim=16` this forces KL ≥ 8.0 nats total, making full posterior collapse impossible.
- **Adversarial disentanglement**: `GradeAngleAdversaryHead` + gradient reversal layer (GRL) in `route_vae_bottleneck.py` pushes both grade AND angle out of z. Two heads share a GRL-reversed trunk: (1) grade_head [B] — MSE against the route's own grade at its stored angle; (2) angle_head [B] — MSE against normalised angle. Grade is predicted at the route's stored angle only (not all 15 listed angles): with `encoder_use_condition=True`, z = f(route, angle_A), so only grade_A is directly encoded in z — grades at other angles are only weakly correlated via route difficulty and do not add meaningful adversarial signal. Combined adversary loss = grade_adv_loss + angle_adv_loss, weighted by `--grade-adversary-weight`. Requires free_bits to be effective.
- **Encoder post-AdaLN scope**: `post_encoder_adaln` is applied only to hold tokens (indices 1:), NOT the CLS token at index 0. The CLS token becomes `route_embedding → z`; conditioning it post-transformer would bake grade/angle directly into z, making the adversary fight a hard-coded injection. The `pre_encoder_adaln` (applied before the transformer) still modulates all tokens including CLS, normalising hold features by grade/angle context before attention runs.
- **Route embedding pooling** (`--route-pool-mode`): controls how the encoder produces the single `route_embedding` vector fed to the bottleneck. `"attention"` (default) — a single learned query vector (`pool_query`, `pool_attn`) attends over all hold token outputs via multi-head attention; the model learns which holds are most style-discriminating and produces a soft weighted summary. Hold tokens only — the shape/CLS token at position 0 is deliberately excluded, preventing shortcutting via pre-computed `shape_desc` statistics. `"cls"` — shape/CLS token at position 0; kept only for backward compatibility with checkpoints trained before attention pooling was added (those have no `pool_query`/`pool_attn` weights, so `build_model_from_checkpoint` defaults to `"cls"` for them).

### `RouteSample` fields

`uuid`, `name`, `angle`, `grade` (float | None), `layout_id`, `num_holds`, `metadata_coverage` (0–1), `tokens` (dict of per-hold tensors), `angle_grades` (dict[float, float] — maps listed angle → grade for all angle variants with quality-filtered data; populated by `_load_raw_routes`). Grades are `difficulty_average` floats; integer bands are recovered with `int(grade)` or `np.floor(grade)`.

### `BoardLibInterface` (`src/database_interfaces/board_lib_interface.py`)

Context-manager DB wrapper. Key methods: `get_climb_by_name`, `get_hold_positions_for_climb`, `get_climb_stats` (returns quality, ascents, setter, difficulty). Hold metadata (type/function/depth/orientation/size) lives in `external_hold_metadata` and is joined automatically when `include_metadata=True`.

### Checkpoint format

Saved by `train_route_cvae.py` as:
```python
{"epoch": int, "model_state_dict": ..., "optimizer_state_dict": ..., "best_val": float, "args": dict}
```
`build_model_from_checkpoint()` in `model_utils.py` reconstructs the full model from `args` + vocabs — no need to manually reconstruct configs.

## Training (Colab)

Training runs on GPU via `notebooks/train_route_cvae.ipynb`. The notebook cell `train_cfg` holds all hyperparameters and `build_train_command` assembles the CLI call. Current recommended config:

```python
"latent_dim": 16,
"free_bits": 2.0,               # 2.0 nats/dim → KL floor = 32 nats total; prevents collapse
"kl_beta": 0.5,
"kl_warmup_epochs": 0,
"encoder_use_condition": True,  # Encoder sees grade/angle via AdaLN → grade-normalised style in z
"grade_adversary_weight": 4.0,  # Strong adversary needed since encoder now has direct grade access
"grade_adversary_alpha": 3.0,
"batch_size": 32,
"decoder_z_memory_tokens": 8,
# Route embedding pooling: learned query attends over hold tokens — avoids CLS shortcutting
"route_pool_mode": "attention",
# Feature flags:
#   use_absolute_pos=False  → no sinusoidal x/y in hold tokens; spatial info via deltas + knn only
#   use_type_feature=True   → hold-type embedding (jug/sloper/crimp/pinch) in tokens + type
#                             fractions in shape_desc (9D CLS token)
#   delta features (Δx, Δy) are always ON in the encoder (no flag; hardcoded use_delta=True)
# IMPORTANT: changing use_type_feature requires --rebuild-cache (shape_desc changes 5D ↔ 9D)
"use_absolute_pos": False,
"use_type_feature": True,
# Encoder architecture (all exposed as CLI args, saved in checkpoint)
"encoder_d_model": 256,          # 32 dims/head, richer per-hold representations
"encoder_nhead": 8,
"encoder_num_layers": 6,
"encoder_dim_feedforward": 1024, # 4× d_model (standard ratio)
# Decoder architecture (lighter — only needs to be good enough to force encoder to encode style)
"decoder_d_model": 128,
"decoder_num_layers": 4,
"decoder_dim_feedforward": 512,
```

**Encoder architecture rationale**: with only 128d/4 layers, style dimensions competed for the same 16 dims/head and "hold→pair→move pattern→route" required all 4 layers leaving no headroom. At 256d/6 layers/ffn=1024: each head gets 32 dims (can simultaneously encode grip type + spatial proximity), and there are 2 spare layers for higher-level synthesis (shoulder moves = horizontal traverse detected by δx pattern + knn bearing angles → regional style → global z). The decoder is deliberately kept lighter at 128d/4 layers — it only needs to reconstruct well enough that the encoder is forced to encode meaningful style signals into z.

**Before training**: rebuild the preprocessed cache on Colab with `--rebuild-cache` to get UUID deduplication, `angle_grades`, and fully normalised features. All board-coordinate-dependent features in `_encode_route_tokens` are now normalised per-route using the board size from `select_product_size`:
- `x_std`, `y_std` in `shape_desc` → divided by `board_x_span` / `board_y_span`
- `delta_x_prev`, `delta_y_prev` → hand-hold deltas only (foot holds always get 0); divided by `board_x_span` / `board_y_span` → range ≈ [−1, 1]
- `knn_features` (encoder-only) → k=3 nearest-neighbour distances normalised by board diagonal + bearing (sin/cos); shape `[L, 9]`
- `mean_move_norm` in `shape_desc` → divided by half the board diagonal → range [0, 1]
- `x`, `y` raw coordinates are stored as-is; normalised at model time by `RouteVAEEncoderConfig.x_min/x_max/y_min/y_max` (corrected to [−20, 164] and [4, 176] from the actual `holes` table range).

**Per-hold token projection dims** (current config: `use_type_feature=True`, `use_absolute_pos=False`):
- Encoder: type(32) + role(8) + hole(16) + depth(8) + ori_sin(8) + ori_cos(8) + size(8) + Δx(8) + Δy(8) + knn(16) = **120 → projected to d_model**
- Decoder: same minus knn and Δx/Δy = **72 → projected to d_model**
- `x`/`y` sinusoidal embeddings omitted (`use_absolute_pos=False`); spatial info comes from Δx/Δy move deltas and knn distances/bearings only.
- shape_desc = [x_std, y_std, foot_frac, hand_density, move_size, jug_frac, sloper_frac, crimp_frac, pinch_frac] → **9D**
  - Indices 0–4 are identical to the old 5D layout; old checkpoints truncate `[:5]` safely.

After training, verify:
- **KL** per epoch should remain ≥ `free_bits × latent_dim` (≥ 16.0 with free_bits=1.0, latent_dim=16). If KL collapses, increase free_bits.
- **No adversary** in this config — grade/angle structure in z is intentional. Verify via Ridge probe in style_analysis.py: global cache should show moderate-high grade/angle R² (model encoded difficulty); micro-cluster cache should show near-zero grade/angle R² (grade/angle controlled).
- **Style feature R²** in micro-cluster analysis: foot_frac, move_size, crowding should be >0.1 (★) or >0.3 (★★) indicating z captures style.
