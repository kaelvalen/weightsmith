// Line chart on uPlot in the quiet swiss register: hairline grid, mono ticks, 2px lines,
// one y-axis. Series identity is carried by dash pattern + direct label + legend, never by
// hue; the single accent marks the focused series only. Options live in chartOptions.ts.

import { useEffect, useMemo, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { alignSeries, Band, buildLineOptions, DASHES, fmt, Marker, Overlays, Series } from "./chartOptions";

export type { Band, Marker, Series } from "./chartOptions";

/** Bumps when the colour scheme changes, so canvas colours are re-resolved. */
function useThemeKey(): number {
  const [key, setKey] = useState(0);
  useEffect(() => {
    const bump = () => setKey((k) => k + 1);
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    mq?.addEventListener("change", bump);
    const mo = new MutationObserver(bump);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      mq?.removeEventListener("change", bump);
      mo.disconnect();
    };
  }, []);
  return key;
}

export function LineChart({
  title,
  series,
  height = 160,
  log = false,
  markers,
  bands,
  yMax = null,
  syncKey,
}: {
  title: string;
  series: Series[];
  height?: number;
  log?: boolean;
  markers?: Marker[];
  bands?: Band[];
  yMax?: number | null;
  syncKey?: string;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const [width, setWidth] = useState(0);
  const [idx, setIdx] = useState<number | null>(null);
  const theme = useThemeKey();
  const data = useMemo(() => alignSeries(series), [series]);
  const overlays = useRef<Overlays>({});
  overlays.current = { markers, bands, yMax };

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(120, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // rebuild only when the chart's structure changes; data and overlays update in place
  const structure = series.map((s) => `${s.id}|${s.focus ? 1 : 0}|${s.points.length === 1 ? 1 : 0}`).join(",");
  useEffect(() => {
    if (!width || !wrap.current) return;
    const opts = buildLineOptions({ series, height, log }, { width }, () => overlays.current);
    opts.cursor = { sync: syncKey ? { key: syncKey } : undefined, drag: { x: false, y: false }, points: { size: 7 } };
    opts.hooks = { ...opts.hooks, setCursor: [(u) => setIdx(u.cursor.idx ?? null)] };
    const u = new uPlot(opts, data as uPlot.AlignedData, wrap.current);
    plot.current = u;
    return () => {
      u.destroy();
      plot.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height, log, structure, syncKey, theme]);

  useEffect(() => {
    plot.current?.setData(data as uPlot.AlignedData);
  }, [data, yMax]);

  useEffect(() => {
    plot.current?.redraw(false);
  }, [markers, bands]);

  const points = series.reduce((n, s) => n + s.points.length, 0);
  const x = idx != null ? data[0][idx] : null;

  return (
    <figure className="m-0 min-w-0" data-points={points}>
      <figcaption className="flex items-baseline justify-between text-label text-muted">
        <span>{title}</span>
        {series.length > 1 && (
          <span className="flex gap-3">
            {series.map((s, i) => (
              <span key={s.id} className="flex items-center gap-1">
                <svg width="16" height="6" aria-hidden>
                  <line x1="0" y1="3" x2="16" y2="3" stroke={s.focus ? "var(--accent)" : "var(--ink)"} strokeWidth="2" strokeDasharray={DASHES[i % DASHES.length].join(" ")} />
                </svg>
                {s.label}
              </span>
            ))}
          </span>
        )}
      </figcaption>
      <div ref={wrap} role="img" aria-label={title} style={{ height }} />
      {x != null && points > 0 && (
        <div className="num pointer-events-none text-label text-muted">
          step {fmt(x)}
          {series.map((s, i) => {
            const v = data[i + 1][idx!];
            return (
              <span key={s.id} className="ml-3 text-ink">
                {series.length > 1 ? `${s.label.slice(0, 8)} ` : ""}
                {v != null ? fmt(v) : "–"}
              </span>
            );
          })}
        </div>
      )}
    </figure>
  );
}
