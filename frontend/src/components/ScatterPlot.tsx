import { useEffect, useRef, useMemo } from "react";
import type { VizResponse } from "../types";

interface Props {
  data: VizResponse | null;
  loading: boolean;
  onSelectRoute: (name: string) => void;
  sidebarWidth: number;
}

const CLUSTER_COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444",
  "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
  "#f97316", "#6366f1", "#14b8a6", "#e11d48",
];

function clusterColor(id: number): string {
  return id < 0 ? "#555" : CLUSTER_COLORS[id % CLUSTER_COLORS.length];
}

function getChartDims(sidebarWidth: number) {
  return {
    width: Math.max(200, window.innerWidth - sidebarWidth),
    height: Math.max(200, window.innerHeight),
  };
}

const centeredFill: React.CSSProperties = {
  flex: 1,
  minHeight: 0,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  background: "#0d0d0d",
  color: "#444",
  fontSize: 13,
  gap: 8,
};

export default function ScatterPlot({ data, loading, onSelectRoute, sidebarWidth }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const onSelectRef = useRef(onSelectRoute);
  onSelectRef.current = onSelectRoute;

  const traces = useMemo(() => {
    if (!data) return [];

    const byCluster = new Map<number, typeof data.points>();
    for (const p of data.points) {
      const arr = byCluster.get(p.cluster) ?? [];
      arr.push(p);
      byCluster.set(p.cluster, arr);
    }

    const is3d = data.n_dims === 3;

    return Array.from(byCluster.entries())
      .sort(([a], [b]) => a - b)
      .map(([cluster, pts]) => {
        const color = clusterColor(cluster);
        const label = cluster < 0 ? "noise" : `C:${cluster}`;
        const hoverText = pts.map(
          (p) =>
            `<b>${p.name}</b><br>` +
            `Grade: ${p.grade_label ?? "ungraded"}<br>` +
            `Angle: ${p.angle !== null ? p.angle + "°" : "?"}<br>` +
            `Cluster: ${cluster < 0 ? "noise" : cluster}`
        );

        if (is3d) {
          return {
            type: "scatter3d",
            mode: "markers",
            name: label,
            x: pts.map((p) => p.x),
            y: pts.map((p) => p.y),
            z: pts.map((p) => p.z ?? 0),
            text: hoverText,
            hovertemplate: "%{text}<extra></extra>",
            customdata: pts.map((p) => p.name),
            marker: { size: 4, color, opacity: 0.8 },
          };
        }

        return {
          type: "scatter",
          mode: "markers",
          name: label,
          x: pts.map((p) => p.x),
          y: pts.map((p) => p.y),
          text: hoverText,
          hovertemplate: "%{text}<extra></extra>",
          customdata: pts.map((p) => p.name),
          marker: { size: 5, color, opacity: 0.75 },
        };
      });
  }, [data]);

  const layout = useMemo(() => {
    if (!data) return {};
    const { width, height } = getChartDims(sidebarWidth);
    const is3d = data.n_dims === 3;
    const base = {
      width,
      height,
      paper_bgcolor: "#0d0d0d",
      plot_bgcolor: "#111",
      font: { color: "#ccc", size: 11 },
      legend: {
        bgcolor: "#1a1a1a",
        bordercolor: "#333",
        borderwidth: 1,
        font: { size: 10, color: "#aaa" },
      },
      margin: { l: 50, r: 20, t: 30, b: 50 },
      hovermode: "closest",
    };

    if (is3d) {
      return {
        ...base,
        scene: {
          xaxis: { title: data.axis_labels[0], gridcolor: "#333", zerolinecolor: "#444", backgroundcolor: "#111" },
          yaxis: { title: data.axis_labels[1], gridcolor: "#333", zerolinecolor: "#444", backgroundcolor: "#111" },
          zaxis: { title: data.axis_labels[2] ?? "", gridcolor: "#333", zerolinecolor: "#444", backgroundcolor: "#111" },
          bgcolor: "#111",
        },
      };
    }

    return {
      ...base,
      xaxis: { title: data.axis_labels[0], gridcolor: "#222", zerolinecolor: "#444", showgrid: true, color: "#888" },
      yaxis: { title: data.axis_labels[1], gridcolor: "#222", zerolinecolor: "#444", showgrid: true, color: "#888" },
    };
  }, [data]);

  useEffect(() => {
    if (!containerRef.current || !data) return;

    const el = containerRef.current;
    let cancelled = false;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    import("plotly.js-dist-min").then((mod: any) => {
      if (cancelled || !containerRef.current) return;
      const Plotly = mod.default ?? mod;

      Plotly.react(el, traces, layout, {
        displayModeBar: true,
        scrollZoom: true,
        responsive: false,
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const handleClick = (event: any) => {
        const name = event?.points?.[0]?.customdata as string | undefined;
        if (name) onSelectRef.current(name);
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (el as any).on("plotly_click", handleClick);

      const handleResize = () => {
        if (!containerRef.current) return;
        const { width, height } = getChartDims(sidebarWidth);
        Plotly.relayout(el, { width, height });
      };

      window.addEventListener("resize", handleResize);

      // Store cleanup on element so the return below can find it
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (el as any).__plotlyCleanup = () => {
        window.removeEventListener("resize", handleResize);
        Plotly.purge(el);
      };
    });

    return () => {
      cancelled = true;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const cleanup = (el as any).__plotlyCleanup;
      if (cleanup) cleanup();
    };
  }, [traces, layout, data]);

  if (loading) {
    return <div style={centeredFill}><span style={{ color: "#555" }}>Projecting latents…</span></div>;
  }

  if (!data) {
    return (
      <div style={centeredFill}>
        <div style={{ fontSize: 36 }}>🧗</div>
        <div>Cluster routes, then click Visualize</div>
      </div>
    );
  }

  return (
    <div style={{ flex: 1, minHeight: 0, overflow: "hidden", background: "#0d0d0d" }}>
      <div ref={containerRef} />
    </div>
  );
}
