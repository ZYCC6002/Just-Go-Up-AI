import React, { useState } from "react";
import type { ClusterJob, ClusterParams } from "../types";

interface Props {
  clusterParams: ClusterParams;
  setClusterParams: (p: ClusterParams) => void;
  vizMethod: "pca" | "umap" | "tsne";
  setVizMethod: (m: "pca" | "umap" | "tsne") => void;
  nDims: 2 | 3;
  setNDims: (n: 2 | 3) => void;
  clusterJob: ClusterJob | null;
  onCluster: () => void;
  onVisualize: () => void;
  vizLoading: boolean;
}

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  color: "#888",
  textTransform: "uppercase",
  letterSpacing: "0.04em",
  marginBottom: 2,
  display: "block",
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "4px 7px",
  background: "#1e1e1e",
  border: "1px solid #333",
  borderRadius: 4,
  color: "#eee",
  fontSize: 13,
  boxSizing: "border-box",
};

const rowStyle: React.CSSProperties = { marginBottom: 10 };

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={rowStyle}>
      <label style={labelStyle}>{label}</label>
      {children}
    </div>
  );
}

function GradeInput({
  value,
  onChange,
  placeholder,
}: {
  value: number | null;
  onChange: (v: number | null) => void;
  placeholder: string;
}) {
  return (
    <input
      type="number"
      min={10}
      max={33}
      placeholder={placeholder}
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}
      style={{ ...inputStyle, width: "48%" }}
    />
  );
}

function AngleInput({
  value,
  onChange,
  placeholder,
}: {
  value: number | null;
  onChange: (v: number | null) => void;
  placeholder: string;
}) {
  return (
    <input
      type="number"
      min={0}
      max={70}
      step={5}
      placeholder={placeholder}
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}
      style={{ ...inputStyle, width: "48%" }}
    />
  );
}

const VGRADE: Record<number, string> = {
  10: "V0", 11: "V1", 12: "V1", 13: "V2", 14: "V2",
  15: "V3", 16: "V3", 17: "V4", 18: "V4", 19: "V5",
  20: "V5", 21: "V6", 22: "V6", 23: "V7", 24: "V8",
  25: "V9", 26: "V10", 27: "V11", 28: "V12", 29: "V13",
};

function gradeHint(g: number | null): string {
  if (g === null) return "";
  const v = VGRADE[g];
  return v ? ` (${v})` : "";
}

type SetParam = <K extends keyof ClusterParams>(k: K, v: ClusterParams[K]) => void;

function UmapAdvanced({ params, set }: { params: ClusterParams; set: SetParam }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginBottom: 10 }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          width: "100%",
          padding: "4px 7px",
          background: "transparent",
          border: "1px solid #2a2a2a",
          borderRadius: 4,
          color: "#666",
          fontSize: 11,
          cursor: "pointer",
          textAlign: "left",
          display: "flex",
          alignItems: "center",
          gap: 5,
        }}
      >
        <span style={{ transition: "transform 0.15s", display: "inline-block", transform: open ? "rotate(90deg)" : "none" }}>▶</span>
        Advanced UMAP params
      </button>

      {open && (
        <div style={{ paddingTop: 8, display: "flex", flexDirection: "column", gap: 0 }}>
          <Row label="Dims">
            <input
              type="number"
              min={2}
              max={16}
              value={params.pre_reduce_dims}
              onChange={(e) => set("pre_reduce_dims", Number(e.target.value))}
              style={inputStyle}
            />
          </Row>
          <Row label="n_neighbors">
            <input
              type="number"
              min={2}
              max={200}
              value={params.umap_n_neighbors}
              onChange={(e) => set("umap_n_neighbors", Number(e.target.value))}
              style={inputStyle}
            />
          </Row>
          <Row label="min_dist">
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={params.umap_min_dist}
              onChange={(e) => set("umap_min_dist", Number(e.target.value))}
              style={inputStyle}
            />
          </Row>
        </div>
      )}
    </div>
  );
}

const sectionStyle: React.CSSProperties = {
  borderTop: "1px solid #2a2a2a",
  paddingTop: 12,
  marginTop: 12,
};

const sectionTitle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 700,
  color: "#aaa",
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  marginBottom: 10,
};

export default function ControlPanel({
  clusterParams,
  setClusterParams,
  vizMethod,
  setVizMethod,
  nDims,
  setNDims,
  clusterJob,
  onCluster,
  onVisualize,
  vizLoading,
}: Props) {
  const set = <K extends keyof ClusterParams>(k: K, v: ClusterParams[K]) =>
    setClusterParams({ ...clusterParams, [k]: v });

  const isRunning =
    clusterJob?.status === "pending" || clusterJob?.status === "running";
  const hasCacheKey = !!clusterJob?.cache_key;
  const jobDone = clusterJob?.status === "done";

  return (
    <div
      style={{
        width: 260,
        minWidth: 260,
        background: "#141414",
        padding: "16px 14px",
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
        gap: 0,
        borderRight: "1px solid #222",
      }}
    >
      <div style={{ fontSize: 15, fontWeight: 700, color: "#fff", marginBottom: 14 }}>
        🧗 JustGoUp AI
      </div>

      {/* ── CLUSTERING ── */}
      <div style={sectionTitle}>Clustering</div>

      <Row label="Method">
        <select
          style={inputStyle}
          value={clusterParams.method}
          onChange={(e) => set("method", e.target.value as "kmeans" | "hdbscan")}
        >
          <option value="kmeans">KMeans</option>
          <option value="hdbscan">HDBSCAN</option>
        </select>
      </Row>

      {clusterParams.method === "kmeans" && (
        <Row label={`Clusters: ${clusterParams.n_clusters}`}>
          <input
            type="range"
            min={2}
            max={12}
            value={clusterParams.n_clusters}
            onChange={(e) => set("n_clusters", Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </Row>
      )}

      {clusterParams.method === "hdbscan" && (
        <Row label="Min cluster size">
          <input
            type="number"
            min={10}
            value={clusterParams.hdbscan_min_cluster_size}
            onChange={(e) => set("hdbscan_min_cluster_size", Number(e.target.value))}
            style={inputStyle}
          />
        </Row>
      )}

      <Row label="Max routes">
        <input
          type="number"
          min={100}
          max={10000}
          value={clusterParams.max_routes}
          onChange={(e) => set("max_routes", Number(e.target.value))}
          style={inputStyle}
        />
      </Row>

      <Row label="UMAP pre-reduction">
        <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={clusterParams.pre_reduce}
            onChange={(e) => set("pre_reduce", e.target.checked)}
          />
          <span style={{ fontSize: 13, color: "#ccc" }}>Enabled</span>
        </label>
      </Row>

      {clusterParams.pre_reduce && <UmapAdvanced params={clusterParams} set={set} />}

      {/* ── GRADE / ANGLE FILTER ── */}
      <div style={sectionStyle}>
        <div style={sectionTitle}>Grade / Angle filter</div>

        <Row
          label={`Grade range${clusterParams.min_grade !== null ? gradeHint(clusterParams.min_grade) : ""}${clusterParams.max_grade !== null ? "–" + gradeHint(clusterParams.max_grade) : ""}`}
        >
          <div style={{ display: "flex", gap: 6 }}>
            <GradeInput
              value={clusterParams.min_grade}
              onChange={(v) => set("min_grade", v)}
              placeholder="min"
            />
            <GradeInput
              value={clusterParams.max_grade}
              onChange={(v) => set("max_grade", v)}
              placeholder="max"
            />
          </div>
          <div style={{ fontSize: 10, color: "#555", marginTop: 2 }}>
            10=V0, 22=V6, 26=V10 (integer)
          </div>
        </Row>

        <Row label="Angle range (°)">
          <div style={{ display: "flex", gap: 6 }}>
            <AngleInput
              value={clusterParams.min_angle}
              onChange={(v) => set("min_angle", v)}
              placeholder="min"
            />
            <AngleInput
              value={clusterParams.max_angle}
              onChange={(v) => set("max_angle", v)}
              placeholder="max"
            />
          </div>
        </Row>
      </div>

      {/* ── CLUSTER BUTTON ── */}
      <button
        onClick={onCluster}
        disabled={isRunning}
        style={{
          marginTop: 4,
          padding: "8px 0",
          background: isRunning ? "#2a2a2a" : "#3b82f6",
          color: isRunning ? "#666" : "#fff",
          border: "none",
          borderRadius: 5,
          fontWeight: 600,
          fontSize: 13,
          cursor: isRunning ? "not-allowed" : "pointer",
          transition: "background 0.2s",
        }}
      >
        {isRunning ? "Clustering…" : "Cluster"}
      </button>

      {clusterJob && (
        <div
          style={{
            marginTop: 6,
            fontSize: 11,
            padding: "4px 7px",
            borderRadius: 4,
            background:
              clusterJob.status === "done"
                ? "#1a3a1a"
                : clusterJob.status === "error"
                ? "#3a1a1a"
                : "#2a2a1a",
            color:
              clusterJob.status === "done"
                ? "#6fdb6f"
                : clusterJob.status === "error"
                ? "#f87171"
                : "#d4a",
          }}
        >
          {clusterJob.status === "done" &&
            `✓ ${clusterJob.n_routes} routes · ${clusterJob.n_clusters} clusters`}
          {clusterJob.status === "pending" && "⏳ Queued…"}
          {clusterJob.status === "running" && "⚙ Running…"}
          {clusterJob.status === "error" && `✗ ${clusterJob.error}`}
        </div>
      )}

      {/* ── VISUALIZATION ── */}
      <div style={sectionStyle}>
        <div style={sectionTitle}>Visualization</div>

        <Row label="Projection">
          <div style={{ display: "flex", gap: 4 }}>
            {(["pca", "umap", "tsne"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setVizMethod(m)}
                style={{
                  flex: 1,
                  padding: "4px 0",
                  background: vizMethod === m ? "#3b82f6" : "#1e1e1e",
                  color: vizMethod === m ? "#fff" : "#888",
                  border: "1px solid #333",
                  borderRadius: 4,
                  fontSize: 11,
                  fontWeight: vizMethod === m ? 700 : 400,
                  cursor: "pointer",
                  textTransform: "uppercase",
                }}
              >
                {m}
              </button>
            ))}
          </div>
        </Row>

        <Row label="Dimensions">
          <div style={{ display: "flex", gap: 4 }}>
            {([2, 3] as const).map((n) => (
              <button
                key={n}
                onClick={() => setNDims(n)}
                style={{
                  flex: 1,
                  padding: "4px 0",
                  background: nDims === n ? "#3b82f6" : "#1e1e1e",
                  color: nDims === n ? "#fff" : "#888",
                  border: "1px solid #333",
                  borderRadius: 4,
                  fontSize: 11,
                  fontWeight: nDims === n ? 700 : 400,
                  cursor: "pointer",
                }}
              >
                {n}D
              </button>
            ))}
          </div>
        </Row>
      </div>

      <button
        onClick={onVisualize}
        disabled={!hasCacheKey || vizLoading || !jobDone}
        style={{
          marginTop: 4,
          padding: "8px 0",
          background: hasCacheKey && jobDone && !vizLoading ? "#10b981" : "#2a2a2a",
          color: hasCacheKey && jobDone && !vizLoading ? "#fff" : "#555",
          border: "none",
          borderRadius: 5,
          fontWeight: 600,
          fontSize: 13,
          cursor: hasCacheKey && jobDone && !vizLoading ? "pointer" : "not-allowed",
          transition: "background 0.2s",
        }}
      >
        {vizLoading ? "Projecting…" : "Visualize"}
      </button>
    </div>
  );
}
