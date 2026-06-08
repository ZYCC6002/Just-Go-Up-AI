# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All scripts are run with `uv run`. The `src/` directory is not an installed package — every script inserts `src/` into `sys.path` at the top.

```bash
# Train CVAE (locally — usually done on Colab via notebooks/train_route_cvae.ipynb)
uv run src/model_training/train_route_cvae.py \
  --latent-dim 16 --free-bits 1.0 --kl-beta 0.1 \
  --no-encoder-use-cond-adaln --no-decoder-use-cond-adaln \
  --decoder-token-dropout 0.5

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
- **Fully unconditional VAE**: grade/angle are **not** injected into the encoder or decoder (`encoder_use_cond_adaln=False`, `decoder_use_cond_adaln=False`). The AdaLN machinery exists in the code but is disabled. z is the sole conditioning signal for the decoder, delivered via cross-attention on the projected latent vector. This means z must capture all route-level structure; grade/angle are encoded implicitly if the route tokens carry enough signal.
- **Free bits** (`--free-bits 1.0`): each latent dimension's KL is clamped from below at 1.0 nats. With `latent_dim=16` this forces KL ≥ 16.0 nats total, making full posterior collapse impossible.
- **Token dropout** (`--decoder-token-dropout 0.5`): during training, each decoder input token at positions 1+ is replaced with a learned `mask_token` embedding with probability 0.5. Forces the decoder to rely on z for hold structure rather than exploiting teacher-forced categorical/spatial context. Disabled at inference (`self.training=False`). BOS token at position 0 is never masked.
- **Route embedding pooling** (`--route-pool-mode mean_max`): the encoder produces `route_embedding` by concatenating the mean and max of all hold token outputs → `Linear(2×d_model, d_model)`. The shape/CLS token at position 0 is excluded from pooling.

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
"free_bits": 1.0,               # 1.0 nats/dim → KL floor = 16.0 nats total; prevents collapse
"kl_beta": 0.1,
"kl_warmup_epochs": 0,
"batch_size": 32,
"val_batch_size": 256,          # larger OK: autoregressive val has no backprop memory pressure
"numeric_weight": 10,
"pairwise_weight": 5.0,
"hole_loss_weight": 0.1,
"decoder_token_dropout": 0.5,
# Fully unconditional VAE — no grade/angle injected into encoder or decoder
"encoder_use_cond_adaln": False,
"decoder_use_cond_adaln": False,
# Adversary disabled
"grade_adversary_weight": 0.0,
# Route embedding pooling: mean+max concatenated → Linear(2*d_model, d_model)
"route_pool_mode": "mean_max",
# Feature flags:
#   use_absolute_pos=True   → sinusoidal x/y embeddings in hold tokens
#   use_type_feature=True   → hold-type embedding (jug/sloper/crimp/pinch) in tokens + type
#                             fractions in shape_desc (9D CLS token)
#   use_delta_features=False → Δx/Δy excluded from encoder tokens
#   use_knn_features=False   → knn distance/bearing features excluded from encoder tokens
# IMPORTANT: changing use_type_feature requires --rebuild-cache (shape_desc changes 5D ↔ 9D)
"use_absolute_pos": True,
"use_type_feature": True,
"use_delta_features": False,
"use_knn_features": False,
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

**Per-hold token projection dims** (current config: `use_type_feature=True`, `use_absolute_pos=True`, `use_delta_features=False`, `use_knn_features=False`):
- Encoder: type(32) + role(8) + hole(16) + depth(8) + ori_sin(8) + ori_cos(8) + size(8) + x_sin(16) + y_sin(16) = **120 → projected to d_model**
- Decoder: same minus depth = type(32) + role(8) + hole(16) + ori_sin(8) + ori_cos(8) + size(8) + x_sin(16) + y_sin(16) = **112 → projected to d_model**
- `x`/`y` sinusoidal embeddings included (`use_absolute_pos=True`); no delta (Δx/Δy) or knn features.
- shape_desc = [x_std, y_std, foot_frac, hand_density, move_size, jug_frac, sloper_frac, crimp_frac, pinch_frac] → **9D**
  - Indices 0–4 are identical to the old 5D layout; old checkpoints truncate `[:5]` safely.

After training, verify:
- **KL** per epoch should remain ≥ `free_bits × latent_dim` (≥ 16.0 with free_bits=1.0, latent_dim=16). If KL collapses, increase free_bits.
- **Grade/angle not injected**: neither encoder nor decoder receives grade/angle; z encodes whatever route-level structure is present in the hold tokens. Grade/angle R² in a Ridge probe of z reflects implicit encoding via hold features (harder routes tend to have different hold distributions), not explicit conditioning.
- **Style feature R²** in micro-cluster analysis: foot_frac, move_size, crowding should be >0.1 (★) or >0.3 (★★) indicating z captures style.
