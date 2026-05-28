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

### Encoder feature improvements

**Problem solved**: The original encoder embedded each hold in isolation — it had no signal about how far apart holds are, or what the overall route shape looks like. Without these signals, the latent space had no way to distinguish a "deadpoint jug route" from a "technical footwork route" even when grade and angle were identical.

**Hold sequence ordering**: Holds are sorted bottom-to-top by y-coordinate before encoding, creating a canonical climbing sequence. This makes the transformer's attention over the sequence correspond to the actual movement order on the wall.

**Move delta features** (`delta_x_prev`, `delta_y_prev`): The horizontal and vertical distance from each hold to the previous one in the sorted sequence. These give the encoder a direct signal for move size — large deltas indicate dynamic/reachy moves; small deltas indicate technical, close-distance footwork. Available to both encoder and decoder (only prior context required).

**Nearest-neighbour distance** (`dist_to_nearest`): The distance from each hold to the closest other hold on the route. This captures local density — a cluster of foot holds near a hand hold has a very different nearest-neighbour profile than an isolated deadpoint target.

**Type embedding capacity** (`type_embed_dim` 16→32): The hold type is the most style-discriminating categorical feature, so its embedding dimension was doubled to give the model more representational capacity for distinguishing jug routes from crimp routes from sloper routes.

**Route-level shape descriptor (CLS token)**: A 9-dimensional vector is computed from the full hold set and prepended to the hold sequence as a learned "shape token." The transformer attends to this token at every layer, giving each per-hold representation access to global route context. The 9 dimensions are:

| Dim | Feature | Style signal |
|-----|---------|-------------|
| 0 | x-spread (std of x coords) | Compression / wide traverses |
| 1 | y-spread (std of y coords) | Long vertical routes |
| 2 | foot fraction (holds with foot role) | Technical footwork density |
| 3 | hand count norm (hand holds / 20) | Endurance / route length |
| 4 | jug fraction | Power / pump routes |
| 5 | sloper fraction | Friction / balance routes |
| 6 | crimp fraction | Crimp-intensive routes |
| 7 | pinch fraction | Pinch-dominant routes |
| 8 | mean pairwise move norm | Dynamic / reachy vs. technical |

The shape token's output at position 0 is used as the `route_embedding` passed to the VAE bottleneck, replacing the previous masked mean-pool. This means the bottleneck encodes a global shape summary rather than an average of per-hold states.

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

Posterior collapse: The decoder can reconstruct routes from the angle/grade AdaLN condition alone without reading z via cross-attention. Once it discovers this shortcut, z receives no gradient signal and collapses to the prior N(0, I) — the latent space becomes pure noise.

Grade/angle entanglement: Even when angle and grade are provided as explicit conditions, the encoder can still embed them into z because it improves reconstruction. This means the dominant axis of the latent space tracks difficulty rather than style — KMeans clustering then just slices the difficulty gradient into bands instead of discovering movement patterns.

Latent space quality is hard to evaluate: good reconstruction loss does not necessarily mean the latent space is useful for clustering or generation.

C-shaped manifold structure: PCA on the extracted latent vectors consistently revealed curved, C-shaped arcs rather than a compact spherical blob. The arcs are smooth, continuous manifold where nearby points correspond to similar climbing styles. However, it exposed a fundamental limitation of K-means clustering: the two tips of a C-arc can land in the same cluster simply because they're close in straight-line distance, even though they're far apart along the manifold and represent very different movement styles.

### Steps taken

KL annealing: KL pressure is increased gradually during training so the model first learns to reconstruct routes well, then is pushed toward a smooth, informative latent space. Beta is kept at 0.25 (rather than 1.0) to give z more capacity for style information.

Free bits (minimum KL per dimension): Each latent dimension's KL is clamped from below at a minimum threshold (λ = 0.5 nats) before summing. This prevents any dimension from fully collapsing to the prior — the encoder is forced to keep every z dimension active. With latent_dim=16 and λ=0.5, z must carry at least 8 nats of information, making the collapsed solution infeasible. Controlled by `--free-bits` (default 0.5).

Post-only AdaLN: Adaptive LayerNorm injects angle and grade into the decoder **after** the transformer stack (not before it). An earlier design applied AdaLN both before and after — this gave the decoder a powerful shortcut to satisfy reconstruction without ever reading z through cross-attention. Removing the pre-decoder AdaLN weakened this shortcut enough that the decoder must attend to z to reconstruct routes accurately, which keeps z informative.

Adversarial disentanglement: A small MLP adversary head is trained alongside the model to predict normalised grade and angle from z. A gradient reversal layer (GRL) sits between z and the adversary: the adversary's own weights are updated normally (it learns to predict grade/angle), but the gradient that flows back through z into the encoder is negated. This causes the encoder to actively hide grade/angle information from z, freeing the latent dimensions to capture style instead. Free bits are required for this to be effective — without them, z collapses to noise before the adversary can apply any pressure. Controlled by `--grade-adversary-weight` (default 0.5).

UMAP + HDBSCAN clustering: To address the C-shaped manifold problem, UMAP is applied as a pre-reduction step before clustering. UMAP is a manifold learning algorithm that "unfolds" curved structures and preserves the topological relationships between points, so that distances in the reduced space better reflect distances along the route style manifold. HDBSCAN is then applied to the UMAP-reduced representation. Unlike K-means, HDBSCAN finds clusters as regions of high density without assuming any particular cluster shape or count, and can label sparse inter-cluster routes as noise (label −1) — a route that doesn't cleanly fit any style archetype.

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
- `--free-bits` — minimum KL per latent dimension in nats (0 = disabled, 0.5 recommended)

Tip: if you change preprocessing filters, rebuild the cache and retrain so the checkpoint vocabularies still match.

### 2) Cluster the latent space

First build a cluster cache. K-means is simple and fast; UMAP + HDBSCAN is better for the curved arc geometry:

```bash
# K-means (6 clusters, all routes)
python src/data_analysis/routes_cluster.py --method kmeans --n-clusters 6

# HDBSCAN with UMAP pre-reduction (recommended for manifold data)
python src/data_analysis/routes_cluster.py \
  --method hdbscan --pre-reduce --pre-reduce-method umap --pre-reduce-dims 5 \
  --hdbscan-min-cluster-size 200 --hdbscan-min-samples 15 \
  --cluster-cache-path data/routes_hdbscan_umap.pt

# Filter by grade/angle before clustering (e.g. V6 at 40°)
python src/data_analysis/routes_cluster.py \
  --method kmeans --n-clusters 6 --min-grade 22 --max-grade 22 --min-angle 40 --max-angle 40 \
  --cluster-cache-path data/routes_v6_40deg_kmeans.pt
```

### 3) Visualize clustered latents

Visualization reads the precomputed cluster cache — it does not reload the model. PCA, UMAP, and t-SNE projections are all available:

```bash
python src/data_analysis/routes_visualize.py --method pca  --cluster-cache-path data/routes_hdbscan_umap.pt --show
python src/data_analysis/routes_visualize.py --method umap --cluster-cache-path data/routes_hdbscan_umap.pt --show
python src/data_analysis/routes_visualize.py --method tsne --cluster-cache-path data/routes_hdbscan_umap.pt --show
```

Common options: `--max-routes`, `--disable-click-visualizer`.

### 4) Route visualization

Run:

`python src/route_visualizer.py --climb-name "Alberts dream"`

---

## Current status

- End-to-end flow is in place (data → train → cluster → visualize).
- Clustering and visualization are separated into two scripts (`routes_cluster.py`, `routes_visualize.py`) sharing a precomputed cache — clustering is done once and all projection methods read the same labels.
- Supports K-means and HDBSCAN clustering, with optional UMAP/PCA pre-reduction; PCA, UMAP, and t-SNE visualization.
- Preprocessing filters by route quality and ascensionist count by default.
- Adversarial disentanglement (GRL) added to prevent grade/angle from dominating z.
- Free bits (λ=0.5 nats/dim) added to prevent posterior collapse.
- Encoder feature improvements (delta moves, grip category, shape CLS token) added to capture style signals beyond hold position.
- Post-only AdaLN in decoder removes the conditioning shortcut that caused posterior collapse in earlier runs.

### Latent space quality (current model)

After the encoder feature improvements and architectural fixes, PCA on 5,000 extracted latents shows:

| Metric | Before | After |
|--------|--------|-------|
| PC1 explained variance | 73% | 42.5% |
| Effective dimensionality (95% variance) | ~1–2 dims | ~4 dims |
| t-SNE cluster structure | Scattered islands | Contiguous regions |

With grade and angle held constant (V6 @ 40°, n=256 routes), the 6 KMeans clusters differentiate on style axes rather than difficulty:

- **Foot-type hold density** (0% vs 41% foot-type holds across clusters) — technical footwork vs. power route
- **Move size** (mean pairwise move norm 0.558 vs 0.612) — tight/compressed vs. dynamic/reachy
- **Route length** (8.4 vs 13.8 mean holds) — short power problems vs. sustained endurance routes
- **Grip composition** (jug-dominant vs. mixed hand/foot) — juggy pull routes vs. coordinated movement routes

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