import type {
  ClusterJob,
  ClusterParams,
  RouteInfo,
  SimilarRoute,
  VizParams,
  VizResponse,
} from "../types";

const BASE = "/api";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail?.detail ?? res.statusText);
  }
  return res.json();
}

async function get<T>(path: string, params?: Record<string, unknown>): Promise<T> {
  const url = new URL(path, window.location.origin);
  url.pathname = `${BASE}${path}`;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== null && v !== undefined) {
        url.searchParams.set(k, String(v));
      }
    }
  }
  const res = await fetch(url.toString());
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail?.detail ?? res.statusText);
  }
  return res.json();
}

export function getDefaultCluster(): Promise<ClusterJob> {
  return get<ClusterJob>("/cluster/default");
}

export function startCluster(params: ClusterParams): Promise<ClusterJob> {
  return post<ClusterJob>("/cluster", params);
}

export function getClusterStatus(jobId: string): Promise<ClusterJob> {
  return get<ClusterJob>(`/cluster/${jobId}`);
}

export function getVisualization(params: VizParams): Promise<VizResponse> {
  return get<VizResponse>("/visualize", params as unknown as Record<string, unknown>);
}

export function getRouteImageUrl(name: string): string {
  return `/api/route/image?name=${encodeURIComponent(name)}`;
}

export function getRouteInfo(name: string): Promise<RouteInfo> {
  return get<RouteInfo>("/route/info", { name });
}

export function getSimilarity(
  cacheKey: string,
  name: string,
  k = 20
): Promise<{ query: string; results: SimilarRoute[] }> {
  return get("/similarity", { cache_key: cacheKey, name, k });
}
