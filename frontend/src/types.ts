export interface ClusterParams {
  method: "kmeans" | "hdbscan";
  n_clusters: number;
  max_routes: number;
  pre_reduce: boolean;
  pre_reduce_method: "umap" | "pca";
  pre_reduce_dims: number;
  umap_n_neighbors: number;
  umap_min_dist: number;
  hdbscan_min_cluster_size: number;
  hdbscan_min_samples: number | null;
  min_grade: number | null;
  max_grade: number | null;
  min_angle: number | null;
  max_angle: number | null;
  seed: number;
}

export interface ClusterJob {
  job_id: string;
  status: "pending" | "running" | "done" | "error";
  cache_key?: string;
  n_routes?: number;
  n_clusters?: number;
  error?: string;
}

export interface ScatterPoint {
  x: number;
  y: number;
  z?: number;
  name: string;
  grade: number | null;
  grade_label: string | null;
  angle: number | null;
  cluster: number;
  uuid: string;
}

export interface VizResponse {
  points: ScatterPoint[];
  axis_labels: string[];
  n_clusters: number;
  method: string;
  n_dims: number;
}

export interface VizParams {
  cache_key: string;
  method: "pca" | "umap" | "tsne";
  n_dims: 2 | 3;
  max_routes?: number;
  seed?: number;
  umap_n_neighbors?: number;
  umap_min_dist?: number;
  tsne_perplexity?: number;
}

export interface RouteInfo {
  name: string;
  grade: number | null;
  grade_label: string;
  angle: number | null;
  quality: number | null;
  ascents: number | null;
  n_holds: number;
  role_counts: Record<string, number>;
  type_counts: Record<string, number>;
}

export interface SimilarRoute {
  rank: number;
  name: string;
  grade: number | null;
  angle: number | null;
  cluster: number;
  z_distance: number;
}
