import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import * as api from "./api/client";
import ControlPanel from "./components/ControlPanel";
import RoutePanel from "./components/RoutePanel";
import ScatterPlot from "./components/ScatterPlot";
import type {
  ClusterJob,
  ClusterParams,
  SimilarRoute,
  VizResponse,
} from "./types";

const DEFAULT_CLUSTER_PARAMS: ClusterParams = {
  method: "kmeans",
  n_clusters: 6,
  max_routes: 5000,
  pre_reduce: true,
  pre_reduce_method: "umap",
  pre_reduce_dims: 5,
  umap_n_neighbors: 15,
  umap_min_dist: 0.0,
  hdbscan_min_cluster_size: 50,
  hdbscan_min_samples: null,
  min_grade: null,
  max_grade: null,
  min_angle: null,
  max_angle: null,
  seed: 42,
};

export default function App() {
  const [clusterParams, setClusterParams] = useState<ClusterParams>(DEFAULT_CLUSTER_PARAMS);
  const [clusterJob, setClusterJob] = useState<ClusterJob | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [vizMethod, setVizMethod] = useState<"pca" | "umap" | "tsne">("umap");
  const [nDims, setNDims] = useState<2 | 3>(2);
  const [vizData, setVizData] = useState<VizResponse | null>(null);
  const [vizLoading, setVizLoading] = useState(false);
  const [vizError, setVizError] = useState<string | null>(null);

  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);
  const [similarRoutes, setSimilarRoutes] = useState<SimilarRoute[] | null>(null);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [expandedRoute, setExpandedRoute] = useState<string | null>(null);

  const [routePanelWidth, setRoutePanelWidth] = useState(380);
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragRef.current = { startX: e.clientX, startWidth: routePanelWidth };

    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return;
      const delta = dragRef.current.startX - ev.clientX;
      setRoutePanelWidth(Math.max(200, Math.min(600, dragRef.current.startWidth + delta)));
    };
    const onUp = () => {
      dragRef.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [routePanelWidth]);

  // ── Polling ───────────────────────────────────────────────────────────────
  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (jobId: string) => {
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const status = await api.getClusterStatus(jobId);
          setClusterJob(status);
          if (status.status === "done" || status.status === "error") {
            stopPolling();
          }
        } catch {
          stopPolling();
        }
      }, 2000);
    },
    [stopPolling]
  );

  useEffect(() => () => stopPolling(), [stopPolling]);

  // ── Cluster ───────────────────────────────────────────────────────────────
  const handleCluster = useCallback(async () => {
    stopPolling();
    setVizData(null);
    setVizError(null);
    try {
      const job = await api.startCluster(clusterParams);
      setClusterJob(job);
      if (job.status !== "done" && job.status !== "error") {
        startPolling(job.job_id);
      }
    } catch (e: unknown) {
      setClusterJob({
        job_id: "",
        status: "error",
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }, [clusterParams, startPolling, stopPolling]);

  // ── Visualize ─────────────────────────────────────────────────────────────
  const handleVisualize = useCallback(async () => {
    if (!clusterJob?.cache_key) return;
    setVizLoading(true);
    setVizError(null);
    setVizData(null);
    try {
      const result = await api.getVisualization({
        cache_key: clusterJob.cache_key,
        method: vizMethod,
        n_dims: nDims,
        max_routes: clusterParams.max_routes,
      });
      setVizData(result);
    } catch (e: unknown) {
      setVizError(e instanceof Error ? e.message : String(e));
    } finally {
      setVizLoading(false);
    }
  }, [clusterJob, vizMethod, nDims, clusterParams.max_routes]);

  // ── Select route ──────────────────────────────────────────────────────────
  const handleSelectRoute = useCallback(
    async (name: string) => {
      setSelectedRoute(name);
      setSimilarRoutes(null);

      if (!clusterJob?.cache_key) return;
      setSimilarLoading(true);
      try {
        const resp = await api.getSimilarity(clusterJob.cache_key, name, 20);
        setSimilarRoutes(resp.results);
      } catch {
        setSimilarRoutes([]);
      } finally {
        setSimilarLoading(false);
      }
    },
    [clusterJob]
  );

  return (
    <div className="app-root">
      <ControlPanel
        clusterParams={clusterParams}
        setClusterParams={setClusterParams}
        vizMethod={vizMethod}
        setVizMethod={setVizMethod}
        nDims={nDims}
        setNDims={setNDims}
        clusterJob={clusterJob}
        onCluster={handleCluster}
        onVisualize={handleVisualize}
        vizLoading={vizLoading}
      />

      <div className="main-area" style={{ position: "relative" }}>
        {vizError && (
          <div className="error-banner">
            Visualization error: {vizError}
          </div>
        )}
        <ScatterPlot
          data={vizData}
          loading={vizLoading}
          onSelectRoute={handleSelectRoute}
          sidebarWidth={260 + routePanelWidth + 6}
        />
        {expandedRoute && (
          <div
            onClick={() => setExpandedRoute(null)}
            style={{
              position: "absolute",
              inset: 0,
              background: "rgba(0,0,0,0.88)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 12,
              cursor: "zoom-out",
              zIndex: 10,
            }}
          >
            <div style={{ fontSize: 13, color: "#aaa", fontWeight: 600 }}>
              {expandedRoute}
            </div>
            <img
              src={api.getRouteImageUrl(expandedRoute)}
              alt={expandedRoute}
              style={{ maxWidth: "90%", maxHeight: "85%", objectFit: "contain" }}
            />
            <div style={{ fontSize: 11, color: "#555" }}>click to close</div>
          </div>
        )}
      </div>

      {/* Drag handle */}
      <div
        onMouseDown={handleResizeStart}
        style={{
          width: 6,
          flexShrink: 0,
          cursor: "col-resize",
          background: "#1a1a1a",
          borderLeft: "1px solid #222",
          borderRight: "1px solid #222",
          transition: "background 0.15s",
        }}
        onMouseEnter={(e) => ((e.currentTarget as HTMLDivElement).style.background = "#2a4a7a")}
        onMouseLeave={(e) => ((e.currentTarget as HTMLDivElement).style.background = "#1a1a1a")}
      />

      <RoutePanel
        selectedRoute={selectedRoute}
        cacheKey={clusterJob?.cache_key ?? null}
        similarRoutes={similarRoutes}
        similarLoading={similarLoading}
        onSelectRoute={handleSelectRoute}
        onExpandImage={setExpandedRoute}
        width={routePanelWidth}
      />
    </div>
  );
}
