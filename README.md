# JustGoUpAI

JustGoUpAI is a machine learning project built for climbers to augment their Kilter board training!

This project aims to:
1. Learn a route "style space" so climbs can be grouped by movement feel (dynamic, crimpy, powerful, etc.)
2. Generate route ideas based on angle/grade preferences or similar existing climbs
3. Eventually connect to send history for weakness analysis and anti-style recommendations

---

## Data sources

Primary data source:

https://github.com/lemeryfertitta/BoardLib
Unofficial wrapper to interact with Board APIs 

Auxiliary metadata source:

https://github.com/Rundstedtzz/climbology/tree/main?tab=readme-ov-file
Contains labelled data for the standard 12x12 Kilter board holds

---

## Feature engineering

Quality filters

Only publicly available routes are used. Routes are also filtered by quality_average and ascenscionist_count. This intentionally biases training toward established routes with stronger community consensus.

Per-Hold Token Construction

Each hold is embedded as a single token by independently embedding all features and concatenating before projection.

Categorical features (type, function, role, hole_id) each get their own learned embedding table, allowing the model to learn dense geometric relationships between categories from co-occurrence in routes.

Coordinates use sinusoidal positional encoding on normalised values, providing multi-scale spatial representation across both fine-grained proximity and coarse board zones. 

Variable Length Handling

Routes on Kilter range from a few holds to upwards of 20. Variable-length sequences are handled by padding all routes in a batch to the length of the longest route with a null token, combined with a boolean padding_mask that prevents padded positions from contributing to attention and loss computation.

---

## Model selection and considerations

### Why CVAE

A CVAE was chosen to perform analysis and clustering of route data.

Reasoning:

- Routes are **sequences**, not independent points.
- We want a meaningful **latent space** for clustering/search.
- We want generation to be **controllable** by context (angle/grade).
- We want some tolerance to ambiguity in route style, which VAEs handle naturally.

The CVAE supports:

- compact latent embeddings for clustering/search,
- conditional decoding for controlled generation,
- uncertainty-aware training through KL regularization.

### Architecture highlights

- **Encoder**: A Transformer encoder over route tokens was chosen over FFN.
- **Bottleneck**: latent `mu/logvar` with reparameterization.
- **Decoder**: autoregressive Transformer decoder with BOS/EOS handling.
- **Conditioning**: angle/grade and latent tokens via decoder conditioning path.

### Training considerations

- Training balances three needs:
	- correctness of discrete hold decisions,
	- accuracy of continuous hold attributes,
	- a smooth latent space that stays useful for sampling.
- Overly long routes are filtered so training remains stable.
- Preprocessing is cached to speed up experiment loops.

### Cache/checkpoint compatibility

Checkpoint weights depend on vocabulary sizes. If preprocessing filters change, cached vocabularies can change too. Rebuild cache and retrain (or use matching cache/checkpoint pairs) to avoid shape mismatch errors.

---

## Setup

Requirements:

- Python >= 3.13
- Dependencies listed in [pyproject.toml](pyproject.toml)

Install with your preferred workflow (for example `uv` or `pip`) using the project dependencies.

---

## Usage

### 1) Train CVAE

Run:

`python src/model_training/train_route_cvae.py`

Useful options:

- `--rebuild-cache`
- `--checkpoint-path data/route_cvae.pt`
- `--resume`
- `--resume-path <path>`
- `--max-routes`, `--epochs`, `--batch-size`, `--latent-dim`

### 2) Latent PCA + KMeans analysis

First build the clustered latent cache once:

`python src/data_analysis/routes_kmeans_original.py --n-clusters 6 --cluster-cache-path data/routes_kmeans_original.pt`

Then run PCA or t-SNE on that cached clustered data:

`python src/data_analysis/routes_pca_kmeans.py --cluster-cache-path data/routes_kmeans_original.pt --show`

`python src/data_analysis/routes_tsne_kmeans.py --cluster-cache-path data/routes_kmeans_original.pt --show`

Common filters:

- `--max-routes`
- `--disable-click-visualizer`

### 3) Clustered latent cache

The PCA and t-SNE analysis scripts now read from a precomputed clustered latent cache. This keeps clustering consistent across views and avoids reclustering from scratch each time.

Default cache path:

`data/routes_kmeans_original.pt`

### 4) Route visualization

Run:

`python src/route_visualizer.py --climb-name "Alberts dream"`

---

## Current status

- End-to-end flow is in place (data -> train -> latent analysis -> visualization).
- PCA and t-SNE views now share a precomputed clustered latent cache.
- Preprocessing now prioritizes route quality and cleaner supervision by default.

---

## Future goals

1. **Generation quality**: make decoded routes feel more climbable and intentional.
2. **Recommendations**: suggest routes by style similarity and training goals.
3. **Evaluation**: build practical metrics for novelty, feasibility, and difficulty calibration.
4. **Coverage**: improve performance across underrepresented angle/grade slices.
5. **Reproducibility**: tighten experiment tracking for model/data/version comparisons.
6. **Productization**: expose route search/generation through a simple CLI or API.

---

## Notes

This is an active project. Data assumptions, filters, and model interfaces may evolve as experiments continue.
