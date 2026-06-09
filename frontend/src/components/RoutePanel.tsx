import { useEffect, useRef, useState } from "react";
import type { RouteInfo, SimilarRoute } from "../types";
import { getRouteImageUrl, getRouteInfo } from "../api/client";

interface Props {
  selectedRoute: string | null;
  cacheKey: string | null;
  similarRoutes: SimilarRoute[] | null;
  similarLoading: boolean;
  onSelectRoute: (name: string) => void;
  onExpandImage: (name: string) => void;
  width: number;
}

function InfoRow({ label, value, indent }: { label: string; value: string; indent?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", paddingLeft: indent ? 10 : 0, color: indent ? "#aaa" : "#ccc" }}>
      <span>{label}</span>
      <span style={{ color: "#eee", fontVariantNumeric: "tabular-nums" }}>{value}</span>
    </div>
  );
}

const VGRADE: Record<number, string> = {
  10: "V0", 11: "V1", 12: "V1", 13: "V2", 14: "V2",
  15: "V3", 16: "V3", 17: "V4", 18: "V4", 19: "V5",
  20: "V5", 21: "V6", 22: "V6", 23: "V7", 24: "V8",
  25: "V9", 26: "V10", 27: "V11", 28: "V12", 29: "V13",
};

function gradeLabel(g: number | null): string {
  if (g === null) return "—";
  const v = VGRADE[Math.round(g)];
  return v ? `${g.toFixed(1)} (${v})` : g.toFixed(1);
}

const CLUSTER_COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444",
  "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
  "#f97316", "#6366f1",
];

function clusterBadge(c: number) {
  const color = c < 0 ? "#555" : CLUSTER_COLORS[c % CLUSTER_COLORS.length];
  return (
    <span
      style={{
        background: color + "33",
        border: `1px solid ${color}`,
        color,
        borderRadius: 3,
        fontSize: 10,
        padding: "1px 5px",
        fontWeight: 600,
      }}
    >
      {c < 0 ? "noise" : `C:${c}`}
    </span>
  );
}

export default function RoutePanel({
  selectedRoute,
  cacheKey,
  similarRoutes,
  similarLoading,
  onSelectRoute,
  onExpandImage,
  width,
}: Props) {
  const [similarHeight, setSimilarHeight] = useState(200);
  const [infoOpen, setInfoOpen] = useState(false);
  const [routeInfo, setRouteInfo] = useState<RouteInfo | null>(null);
  const [infoLoading, setInfoLoading] = useState(false);
  const dragRef = useRef<{ startY: number; startHeight: number } | null>(null);

  useEffect(() => {
    setRouteInfo(null);
    setInfoOpen(false);
  }, [selectedRoute]);

  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault();
    dragRef.current = { startY: e.clientY, startHeight: similarHeight };

    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return;
      // dragging up (negative delta) → expand similar routes; down → shrink
      const delta = dragRef.current.startY - ev.clientY;
      setSimilarHeight(Math.max(40, Math.min(600, dragRef.current.startHeight + delta)));
    };
    const onUp = () => {
      dragRef.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  return (
    <div
      style={{
        width,
        minWidth: width,
        background: "#141414",
        borderLeft: "1px solid #222",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Board image — takes all remaining space */}
      <div
        style={{
          flex: 1,
          minHeight: 0,
          background: "#0d0d0d",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "hidden",
        }}
      >
        {selectedRoute ? (
          <img
            key={selectedRoute}
            src={getRouteImageUrl(selectedRoute)}
            alt={selectedRoute}
            onClick={() => onExpandImage(selectedRoute)}
            style={{
              maxWidth: "100%",
              maxHeight: "100%",
              objectFit: "contain",
              cursor: "zoom-in",
            }}
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        ) : (
          <span style={{ color: "#333", fontSize: 11 }}>No route selected</span>
        )}
      </div>

      {/* Info toggle button + collapsible panel */}
      {selectedRoute && (
        <div style={{ flexShrink: 0, borderTop: "1px solid #1e1e1e" }}>
          <button
            onClick={async () => {
              const opening = !infoOpen;
              setInfoOpen(opening);
              if (opening && !routeInfo && !infoLoading) {
                setInfoLoading(true);
                try {
                  setRouteInfo(await getRouteInfo(selectedRoute));
                } catch {
                  // silently ignore — info stays null
                } finally {
                  setInfoLoading(false);
                }
              }
            }}
            style={{
              width: "100%",
              padding: "5px 14px",
              background: "transparent",
              border: "none",
              borderBottom: infoOpen ? "1px solid #1e1e1e" : "none",
              color: "#555",
              fontSize: 11,
              cursor: "pointer",
              textAlign: "left",
              display: "flex",
              alignItems: "center",
              gap: 5,
            }}
          >
            <span style={{ display: "inline-block", transition: "transform 0.15s", transform: infoOpen ? "rotate(90deg)" : "none" }}>▶</span>
            Route info
          </button>
          {infoOpen && (
            <div style={{ padding: "8px 14px 10px", fontSize: 12, color: "#ccc", lineHeight: 1.7 }}>
              {infoLoading && <span style={{ color: "#555" }}>Loading…</span>}
              {routeInfo && (
                <>
                  <div style={{ fontWeight: 600, marginBottom: 4, color: "#eee", fontSize: 12 }}>{routeInfo.name}</div>
                  <InfoRow label="Grade" value={routeInfo.grade_label} />
                  {routeInfo.angle !== null && <InfoRow label="Angle" value={`${routeInfo.angle}°`} />}
                  {routeInfo.quality !== null && <InfoRow label="Quality" value={`${routeInfo.quality.toFixed(2)} / 4.0`} />}
                  {routeInfo.ascents !== null && <InfoRow label="Ascents" value={String(routeInfo.ascents)} />}
                  <div style={{ marginTop: 6, marginBottom: 2, color: "#888", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                    Holds ({routeInfo.n_holds})
                  </div>
                  {(["start", "middle", "finish", "foot"] as const).map((r) =>
                    routeInfo.role_counts[r] ? (
                      <InfoRow key={r} label={r.charAt(0).toUpperCase() + r.slice(1)} value={String(routeInfo.role_counts[r])} indent />
                    ) : null
                  )}
                  {Object.keys(routeInfo.type_counts).length > 0 && (
                    <>
                      <div style={{ marginTop: 6, marginBottom: 2, color: "#888", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                        Hold types
                      </div>
                      {Object.entries(routeInfo.type_counts).map(([t, n]) => (
                        <InfoRow key={t} label={t} value={String(n)} indent />
                      ))}
                    </>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      )}

      {/* Vertical drag handle */}
      <div
        onMouseDown={handleResizeStart}
        style={{
          height: 6,
          flexShrink: 0,
          cursor: "row-resize",
          background: "#1a1a1a",
          borderTop: "1px solid #222",
          borderBottom: "1px solid #222",
          transition: "background 0.15s",
        }}
        onMouseEnter={(e) => ((e.currentTarget as HTMLDivElement).style.background = "#2a4a7a")}
        onMouseLeave={(e) => ((e.currentTarget as HTMLDivElement).style.background = "#1a1a1a")}
      />

      {/* Similar routes — fixed draggable height */}
      <div
        style={{
          height: similarHeight,
          flexShrink: 0,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "8px 14px 4px",
            fontSize: 10,
            fontWeight: 700,
            color: "#666",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            flexShrink: 0,
          }}
        >
          Similar routes
          {similarRoutes && (
            <span style={{ fontWeight: 400, color: "#444", marginLeft: 6 }}>
              ({similarRoutes.length})
            </span>
          )}
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: "0 0 8px" }}>
          {similarLoading && (
            <div style={{ padding: "12px 14px", color: "#444", fontSize: 12 }}>
              Loading…
            </div>
          )}
          {!similarLoading && !selectedRoute && (
            <div style={{ padding: "12px 14px", color: "#333", fontSize: 12 }}>
              Select a route to find similar climbs
            </div>
          )}
          {!similarLoading && !cacheKey && selectedRoute && (
            <div style={{ padding: "12px 14px", color: "#555", fontSize: 12 }}>
              Cluster first to enable similarity search
            </div>
          )}
          {!similarLoading &&
            similarRoutes &&
            similarRoutes.map((r) => (
              <div
                key={r.rank}
                onClick={() => onSelectRoute(r.name)}
                style={{
                  padding: "6px 14px",
                  cursor: "pointer",
                  borderBottom: "1px solid #1a1a1a",
                  display: "flex",
                  flexDirection: "column",
                  gap: 2,
                  transition: "background 0.1s",
                }}
                onMouseEnter={(e) =>
                  ((e.currentTarget as HTMLDivElement).style.background = "#1c1c1c")
                }
                onMouseLeave={(e) =>
                  ((e.currentTarget as HTMLDivElement).style.background = "transparent")
                }
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 6,
                  }}
                >
                  <span
                    style={{
                      fontSize: 12,
                      color: "#ddd",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      flex: 1,
                    }}
                  >
                    <span style={{ color: "#555", fontSize: 10, marginRight: 4 }}>
                      #{r.rank}
                    </span>
                    {r.name}
                  </span>
                  {clusterBadge(r.cluster)}
                </div>
                <div style={{ display: "flex", gap: 8, fontSize: 10, color: "#666" }}>
                  <span>{gradeLabel(r.grade)}</span>
                  {r.angle !== null && <span>{r.angle}°</span>}
                  <span style={{ marginLeft: "auto", color: "#3b5" }}>
                    d={r.z_distance.toFixed(3)}
                  </span>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}
