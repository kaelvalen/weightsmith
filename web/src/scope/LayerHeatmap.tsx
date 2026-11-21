// Layer × step heatmap on canvas (ported from trainscope). Rows are tensors (grouped by block
// when there are many), cells are the step-bucketed z-score of the metric; red = far from normal.
import { useEffect, useMemo, useRef, useState } from "react";
import { bandSpan, resolveColor } from "../ui/chartOptions";
import { firstCrossings, groupRows, heatmapSummary, sortRows, zColor } from "./layerMatrix";
import type { LayerMatrix } from "./types";

const ROW = 16;
const AXIS = 20;

function rgb(color: string, fallback: number[]): number[] {
  const m = /^#?([0-9a-f]{6})$/i.exec(color.trim());
  if (!m) return fallback;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function LayerHeatmap({
  matrix,
  domain,
  labelWidth = 132,
  warningStep = null,
  spikeWindow = null,
  highlight = null,
  onSelect,
}: {
  matrix: LayerMatrix | null;
  domain: [number, number];
  labelWidth?: number;
  warningStep?: number | null;
  spikeWindow?: { from: number; to: number } | null;
  highlight?: string | null;
  onSelect?: (layer: string, step: number) => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [byCrossing, setByCrossing] = useState(false);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);

  const starts = useMemo(() => matrix?.bucket_starts ?? [], [matrix]);
  const grouped = useMemo(() => (matrix ? groupRows(matrix, { expanded }) : []), [matrix, expanded]);
  const crossings = useMemo(() => firstCrossings(grouped, starts), [grouped, starts]);
  const rows = useMemo(() => sortRows(grouped, byCrossing ? "first" : "model", crossings), [grouped, byCrossing, crossings]);
  const [x0, x1] = domain;
  const plotW = Math.max(1, width - labelWidth - 10);
  const toPx = (step: number) => labelWidth + ((step - x0) / Math.max(1e-9, x1 - x0)) * plotW;
  const bucketEnd = (i: number) => (i + 1 < starts.length ? starts[i + 1] : x1 + 1);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.floor(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, [rows.length > 0]);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !width || !rows.length) return;
    const dpr = window.devicePixelRatio || 1;
    const height = rows.length * ROW + AXIS;
    c.width = width * dpr;
    c.height = height * dpr;
    c.style.width = `${width}px`;
    c.style.height = `${height}px`;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const surface = rgb(resolveColor("var(--bg)", "#f7f7f5"), [247, 247, 245]);
    const heat = rgb(resolveColor("var(--error)", "#d92d20"), [217, 45, 32]);
    const muted = resolveColor("var(--muted)");
    const ink = resolveColor("var(--ink)", "#111111");
    ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
    ctx.textBaseline = "middle";
    rows.forEach((row, r) => {
      const y = r * ROW;
      ctx.fillStyle = row.key === highlight ? ink : muted;
      ctx.textAlign = "right";
      const prefix = row.isGroup ? (expanded.has(row.key) ? "▾ " : "▸ ") : "";
      ctx.fillText(`${prefix}${row.label}`, labelWidth - 10, y + ROW / 2, labelWidth - 14);
      starts.forEach((start, b) => {
        const a = Math.max(labelWidth, toPx(start));
        const e = Math.min(labelWidth + plotW, toPx(bucketEnd(b)));
        if (e <= a) return;
        ctx.fillStyle = zColor(row.z[b], surface, heat);
        ctx.fillRect(a, y + 1, Math.max(1, e - a - (e - a > 3 ? 1 : 0)), ROW - 2);
      });
      if (row.key === highlight) {
        ctx.strokeStyle = ink;
        ctx.strokeRect(labelWidth + 0.5, y + 0.5, plotW - 1, ROW - 1);
      }
    });
    const bottom = rows.length * ROW;
    ctx.lineWidth = 1;
    if (warningStep != null) {
      const x = toPx(warningStep);
      ctx.strokeStyle = muted;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, bottom);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    if (spikeWindow) {
      const [sa, se] = bandSpan(toPx(spikeWindow.from), toPx(spikeWindow.to), 3);
      const a = Math.max(labelWidth, sa);
      const e = Math.min(labelWidth + plotW, se);
      if (e > a) {
        ctx.strokeStyle = resolveColor("var(--error)");
        ctx.strokeRect(a + 0.5, 0.5, e - a - 1, bottom - 1);
      }
    }
    ctx.fillStyle = muted;
    ctx.textAlign = "center";
    for (let i = 0; i < 5; i++) {
      const step = Math.round(x0 + ((x1 - x0) * i) / 4);
      ctx.fillText(String(step), toPx(step), bottom + AXIS / 2 + 2);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, width, expanded, starts, warningStep, spikeWindow, highlight, x0, x1]);

  if (!matrix) return <p className="m-0 text-small text-muted">Loading layer metrics…</p>;
  if (!matrix.layers.length) return <p className="m-0 text-small text-muted">No layer metrics recorded for this run.</p>;

  const hit = (e: React.MouseEvent) => {
    const rect = canvas.current!.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const r = Math.floor((e.clientY - rect.top) / ROW);
    if (r < 0 || r >= rows.length) return null;
    const row = rows[r];
    if (px < labelWidth) return { row, bucket: -1, px, y: r * ROW };
    const step = x0 + ((px - labelWidth) / plotW) * (x1 - x0);
    let b = -1;
    for (let i = 0; i < starts.length; i++) if (starts[i] <= step) b = i;
    return { row, bucket: b, px, y: r * ROW };
  };

  return (
    <div data-testid="layer-heatmap">
      <div className="flex justify-end pb-1 text-label">
        <button className="link" aria-pressed={byCrossing} onClick={() => setByCrossing((v) => !v)} style={{ color: byCrossing ? "var(--ink)" : undefined }}>
          sort by first crossing
        </button>
      </div>
      <div ref={wrap} className="relative">
        <canvas
          ref={canvas}
          role="img"
          aria-label={heatmapSummary(rows, crossings)}
          className="block cursor-pointer"
          onMouseLeave={() => setTip(null)}
          onMouseMove={(e) => {
            const h = hit(e);
            if (!h || h.bucket < 0) return setTip(null);
            const v = h.row.values[h.bucket];
            const z = h.row.z[h.bucket];
            setTip({ x: h.px, y: h.y + ROW, text: `${h.row.label} · step ${starts[h.bucket]} · ${v == null ? "n/a" : +v.toPrecision(4)} · z ${z == null ? "n/a" : z.toFixed(1)}` });
          }}
          onClick={(e) => {
            const h = hit(e);
            if (!h) return;
            if (h.row.isGroup)
              setExpanded((prev) => {
                const next = new Set(prev);
                if (next.has(h.row.key)) next.delete(h.row.key);
                else next.add(h.row.key);
                return next;
              });
            else onSelect?.(h.row.key, h.bucket >= 0 ? starts[h.bucket] : x0);
          }}
        />
        {tip && (
          <div className="num pointer-events-none absolute whitespace-nowrap border border-line bg-bg px-1 text-label" style={{ left: Math.min(tip.x + 8, Math.max(0, width - 320)), top: tip.y }}>
            {tip.text}
          </div>
        )}
      </div>
    </div>
  );
}
