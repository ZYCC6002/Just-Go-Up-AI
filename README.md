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

Quality filters: Only publicly available routes are used. Routes are also filtered by quality_average and ascenscionist_count. This intentionally biases training toward established routes with stronger community consensus.

Per-Hold Token Construction: Each hold is embedded as a single token by independently embedding all features and concatenating before projection.

Categorical features (type, function, role, hole_id) each get their own learned embedding table, allowing the model to learn dense geometric relationships between categories from co-occurrence in routes.

Coordinates use sinusoidal positional encoding on normalised values, providing multi-scale spatial representation across both fine-grained proximity and coarse board zones.

---

## Model selection and considerations

### Why CVAE with transformer encoder/decoder

A CVAE was chosen to perform analysis and clustering of route data.

Reasoning:

- Latent Space analysis: We want a meaningful latent space for clustering/search.
- Sequential Modelling: Routes are sequences, and transformers can model hold order and relationships between holds instead of treating each hold independently.
- Context-conditioned routes: We want the learned representation to depend on context, since angle and grade change the route distribution.
- Non-linear structure: The CVAE can capture non-linear relationships in route data that linear methods like PCA may miss.
- Generative Capabilities: Because the latent space is continuous, we can sample from it to generate new routes as well as cluster existing ones.

### Challenges Faced

Posterior collapse: The decoder is often too strong and ignores the latent variable, causing the learned latent space to contain little to no information.

Grade/angle entanglement: Even when angle and grade are provided as explicit conditions, the encoder still embeds them into z because it helps the decoder reconstruct routes. This means the dominant axis of the latent space tracks difficulty rather than style — KMeans clustering then just slices the difficulty gradient into bands instead of discovering movement patterns.

Latent space quality is hard to evaluate: good reconstruction loss does not necessarily mean the latent space is useful for clustering or generation.

### Steps taken

KL annealing: KL pressure is increased gradually during training so the model first learns to reconstruct routes well, then is pushed toward a smooth, informative latent space. This reduces the chance of posterior collapse. Beta is kept at 0.25 (rather than 1.0) to give z more capacity for style information.

AdaLN conditioning: Adaptive LayerNorm injects angle and grade into the encoder and decoder without routing them through z. This is a necessary but not sufficient step — without additional pressure, the encoder still finds it useful to encode grade/angle into z as well.

Adversarial disentanglement: A small MLP adversary head is trained alongside the model to predict normalised grade and angle from z. A gradient reversal layer (GRL) sits between z and the adversary: the adversary's own weights are updated normally (it learns to predict grade/angle), but the gradient that flows back through z into the encoder is negated. This causes the encoder to actively hide grade/angle information from z, freeing the latent dimensions to capture style instead. The strength of this pressure is controlled by `--grade-adversary-weight` (default 0.5).

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
- `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--latent-dim`
- `--numeric-weight`, `--kl-beta`, `--kl-warmup-epochs`
- `--encoder-use-condition`, `--encoder-use-cond-adaln`, `--decoder-use-cond-adaln`
- `--decoder-z-memory-tokens`, `--grad-clip-norm`, `--seed`
- `--max-routes`
- `--grade-adversary-weight` — adversarial disentanglement strength (0 = disabled, 0.5 recommended)
- `--grade-adversary-alpha` — gradient reversal layer scale factor

Tip: if you change preprocessing filters, rebuild the cache and retrain so the checkpoint vocabularies still match.

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

- End-to-end flow is in place (data → train → latent analysis → visualization).
- PCA and t-SNE views share a precomputed clustered latent cache.
- Preprocessing filters by route quality and ascensionist count by default.
- Adversarial disentanglement added to prevent grade/angle from dominating the latent space.

---

## Future goals

<!-- 1. **Generation quality**: make decoded routes feel more climbable and intentional.
2. **Recommendations**: suggest routes by style similarity and training goals.
3. **Evaluation**: build practical metrics for novelty, feasibility, and difficulty calibration.
4. **Coverage**: improve performance across underrepresented angle/grade slices.
5. **Reproducibility**: tighten experiment tracking for model/data/version comparisons.
6. **Productization**: expose route search/generation through a simple CLI or API. -->

---

## Notes

This is an active project. Data assumptions, filters, and model interfaces may evolve as experiments continue.