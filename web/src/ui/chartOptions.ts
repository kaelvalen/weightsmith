// Pure uPlot options for the quiet swiss line chart (ported from trainscope's chartOptions).
// Series identity is a dash pattern, never a hue; the single accent marks the focused series.
// Bands, markers and the off-scale marker are drawn on the canvas from overlays().

import type uPlot from "uplot";

export interface Series {
  id: string;
  label: string;
  points: [number, number][];
  focus?: boolean;
}

export interface Marker {
  x: number;
  label?: string;
  kind?: "spike" | "note";
}

export interface Band {
  x0: number;
  x1: number;
  tone: "warn" | "spike";
}

export interface Overlays {
  markers?: Marker[];
  bands?: Band[];
  yMax?: number | null;
  offScale?: { at: number; text: string } | null;
}

export const DASHES: number[][] = [[], [6, 3], [2, 3], [8, 3, 2, 3], [1, 2]];
const FONT = '10px "JetBrains Mono", ui-monospace, monospace';

/** Canvas needs real colours: resolve `var(--token)` through the stylesheet. */
export function resolveColor(color: string, fallback = "#6b6b66"): string {
  if (!color.startsWith("var(")) return color;
  if (typeof document === "undefined") return fallback;
  return getComputedStyle(document.documentElement).getPropertyValue(color.slice(4, -1).trim()).trim() || fallback;
}

export const toFinite = (values: (number | null)[]): (number | null)[] => values.map((v) => (v != null && Number.isFinite(v) ? v : null));

/** One x axis for series sampled on different steps; missing points are gaps. */
export function alignSeries(series: Series[]): [number[], ...(number | null)[][]] {
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const at = new Map(xs.map((x, i) => [x, i]));
  const cols = series.map((s) => {
    const col: (number | null)[] = new Array(xs.length).fill(null);
    for (const [x, y] of s.points) col[at.get(x)!] = y;
    return toFinite(col);
  });
  return [xs, ...cols];
}

/** True when any series has a plotted point; reads live data, not the series at build time. */
export function hasData(data: (number | null)[][]): boolean {
  return data.slice(1).some((col) => col.some((v) => v != null));
}

/** 1.25x the largest finite loss before the first spike, so one explosion can't flatten the history. */
export function lossYMax(steps: number[], loss: (number | null)[], firstSpike: number | null): number | null {
  if (firstSpike == null) return null;
  let max = -Infinity;
  steps.forEach((step, i) => {
    const v = loss[i];
    if (step < firstSpike && v != null && Number.isFinite(v)) max = Math.max(max, v);
  });
  return Number.isFinite(max) ? max * 1.25 : null;
}

/** Where the line leaves the plot, labelled with its peak (or the first non-finite point; JSON sends NaN as null). */
export function offScaleMarker(steps: number[], values: (number | null)[], yMax: number | null): { at: number; text: string } | null {
  if (yMax == null) return null;
  const finite = (v: number | null): v is number => v != null && Number.isFinite(v);
  const i = values.findIndex((v) => (finite(v) ? v > yMax : true));
  if (i < 0) return null;
  if (!finite(values[i])) return { at: steps[i], text: `non-finite at step ${steps[i]}` };
  let peak = i;
  for (let j = i; j < values.length; j++) {
    const v = values[j];
    if (finite(v) && v > (values[peak] as number)) peak = j;
  }
  return { at: steps[i], text: `${(values[peak] as number).toFixed(1)} at step ${steps[peak]}, off scale` };
}

/** A band at least `min` pixels wide, so a single-step spike still shows. */
export function bandSpan(x0: number, x1: number, min: number): [number, number] {
  if (x1 - x0 >= min) return [x0, x1];
  const mid = (x0 + x1) / 2;
  return [mid - min / 2, mid + min / 2];
}

export const fmt = (v: number) => {
  if (!Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  return String(Number(v.toPrecision(3)));
};

const dpr = () => (typeof window === "undefined" ? 1 : window.devicePixelRatio || 1);

function drawBands(u: uPlot, bands: Band[]) {
  const { ctx, bbox } = u;
  ctx.save();
  for (const b of bands) {
    const [a, c] = bandSpan(u.valToPos(b.x0, "x", true), u.valToPos(b.x1, "x", true), 3 * dpr());
    const x0 = Math.max(bbox.left, a);
    const x1 = Math.min(bbox.left + bbox.width, c);
    if (x1 <= x0) continue;
    ctx.fillStyle = resolveColor(b.tone === "spike" ? "var(--band-spike)" : "var(--band-warn)");
    ctx.fillRect(x0, bbox.top, x1 - x0, bbox.height);
  }
  ctx.restore();
}

function drawOverlays(u: uPlot, o: Overlays, series: Series[]) {
  const { ctx, bbox } = u;
  const r = dpr();
  ctx.save();
  ctx.font = FONT.replace("10px", `${10 * r}px`);
  ctx.lineWidth = r;
  for (const m of o.markers ?? []) {
    const x = u.valToPos(m.x, "x", true);
    if (x < bbox.left || x > bbox.left + bbox.width) continue;
    ctx.strokeStyle = resolveColor(m.kind === "note" ? "var(--muted)" : "var(--error)");
    ctx.setLineDash(m.kind === "note" ? [3 * r, 3 * r] : []);
    ctx.beginPath();
    ctx.moveTo(x, bbox.top);
    ctx.lineTo(x, bbox.top + bbox.height);
    ctx.stroke();
  }
  if (o.offScale) {
    const x = u.valToPos(o.offScale.at, "x", true);
    ctx.fillStyle = resolveColor("var(--error)");
    ctx.textAlign = "right";
    ctx.fillText(o.offScale.text, x - 6 * r, bbox.top + 10 * r);
  }
  // direct labels at line ends, nudged apart so they never overlap
  if (series.length <= 4) {
    ctx.fillStyle = resolveColor("var(--muted)");
    ctx.textAlign = "left";
    const ends = series
      .map((s, i) => {
        const ys = (u.data[i + 1] ?? []) as (number | null)[];
        let k = ys.length - 1;
        while (k >= 0 && ys[k] == null) k--;
        return k < 0 ? null : { y: u.valToPos(ys[k] as number, "y", true), text: series.length > 1 ? s.label : fmt(ys[k] as number) };
      })
      .filter((e): e is { y: number; text: string } => e !== null)
      .sort((a, b) => a.y - b.y);
    for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 11 * r) ends[k].y = ends[k - 1].y + 11 * r;
    for (const e of ends) ctx.fillText(e.text.length > 10 ? e.text.slice(0, 10) + "…" : e.text, bbox.left + bbox.width + 6 * r, e.y + 3 * r);
  }
  if (!hasData(u.data as (number | null)[][])) {
    ctx.fillStyle = resolveColor("var(--muted)");
    ctx.textAlign = "left";
    ctx.fillText("no data yet", bbox.left + 4 * r, bbox.top + bbox.height / 2);
  }
  ctx.restore();
}

export interface LineOptionsProps {
  series: Series[];
  height: number;
  log?: boolean;
}

export function buildLineOptions(props: LineOptionsProps, { width }: { width: number }, overlays: () => Overlays = () => ({})): uPlot.Options {
  const muted = resolveColor("var(--muted)");
  const axis = { stroke: muted, font: FONT, ticks: { show: false }, grid: { stroke: resolveColor("var(--line)", "#e2e2dc"), width: 1 } };
  return {
    width,
    height: props.height,
    padding: [8, 72, 0, 0],
    legend: { show: false },
    scales: {
      x: { time: false },
      y: props.log
        ? { distr: 3 }
        : {
            range: (_u: uPlot, min: number, max: number): uPlot.Range.MinMax => {
              const yMax = overlays().yMax ?? null;
              if (min == null || max == null) return [0, yMax ?? 1];
              if (yMax != null && min < yMax) return [min, yMax]; // zoomed into a diverged tail: show the data
              const span = max - min || Math.abs(max) || 1;
              return [min - span * 0.05, max + span * 0.05];
            },
          },
    },
    axes: [
      { ...axis, grid: { show: false }, size: 20, values: (_u, ticks) => ticks.map(fmt) },
      { ...axis, size: 48, values: (_u, ticks) => ticks.map(fmt) },
    ],
    series: [
      { label: "step" },
      ...props.series.map((s, i) => ({
        label: s.label,
        stroke: resolveColor(s.focus ? "var(--accent)" : "var(--ink)", s.focus ? "#2b4eff" : "#111111"),
        width: 2,
        dash: DASHES[i % DASHES.length],
        spanGaps: true,
        points: { show: s.points.length === 1, size: 8 },
      })),
    ],
    hooks: {
      drawClear: [(u) => drawBands(u, overlays().bands ?? [])],
      draw: [(u) => drawOverlays(u, overlays(), props.series)],
    },
  };
}
