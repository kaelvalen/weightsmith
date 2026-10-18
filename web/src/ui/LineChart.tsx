// Minimal SVG line chart in the quiet swiss register: hairline axes, mono ticks,
// 2px lines, one y-axis. Series identity is carried by dash pattern + direct label +
// legend, never by hue; the single accent marks the focused series only.

import { useEffect, useMemo, useRef, useState } from "react";

export interface Series {
  id: string;
  label: string;
  points: [number, number][];
  focus?: boolean;
}

const DASHES = ["", "6 3", "2 3", "8 3 2 3", "1 2"];

const fmt = (v: number) => {
  if (!Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  return String(Number(v.toPrecision(3)));
};

function ticks(min: number, max: number, n = 4): number[] {
  if (min === max) return [min];
  const step = Math.pow(10, Math.floor(Math.log10((max - min) / n)));
  const err = (max - min) / n / step;
  const s = step * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const out = [];
  for (let v = Math.ceil(min / s) * s; v <= max + s * 1e-9; v += s) out.push(+v.toPrecision(12));
  return out;
}

export function LineChart({ title, series, height = 160, log = false }: { title: string; series: Series[]; height?: number; log?: boolean }) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [width, setWidth] = useState(320);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(120, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const W = width;
  const H = height;
  const pad = { l: 48, r: 72, t: 8, b: 20 };

  const ty = (v: number) => (log ? Math.log10(Math.max(v, 1e-12)) : v);
  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    const xs = series.flatMap((s) => s.points.map((p) => p[0]));
    const ys = series.flatMap((s) => s.points.map((p) => ty(p[1]))).filter(Number.isFinite);
    const yMin0 = ys.length ? Math.min(...ys) : 0;
    const yMax0 = ys.length ? Math.max(...ys) : 1;
    const span = yMax0 - yMin0 || Math.abs(yMax0) || 1;
    return {
      xMin: xs.length ? Math.min(...xs) : 0,
      xMax: xs.length ? Math.max(...xs) : 1,
      yMin: yMin0 - span * 0.05,
      yMax: yMax0 + span * 0.05,
    };
  }, [series, log]);

  const sx = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * (W - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (ty(y) - yMin) / (yMax - yMin || 1)) * (H - pad.t - pad.b);
  const yt = ticks(yMin, yMax);
  const xt = ticks(xMin, xMax, 4).filter((v) => v >= xMin && v <= xMax);

  const onMove = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const x = xMin + ((e.clientX - r.left - pad.l) / (r.width - pad.l - pad.r)) * (xMax - xMin);
    setHover(Math.max(xMin, Math.min(xMax, x)));
  };

  const nearest = (s: Series, x: number) => {
    let best: [number, number] | null = null;
    for (const p of s.points) if (!best || Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
    return best;
  };

  const empty = !series.some((s) => s.points.length);

  return (
    <figure className="m-0 min-w-0">
      <figcaption className="flex items-baseline justify-between text-label text-muted">
        <span>{title}</span>
        {series.length > 1 && (
          <span className="flex gap-3">
            {series.map((s, i) => (
              <span key={s.id} className="flex items-center gap-1">
                <svg width="16" height="6" aria-hidden>
                  <line x1="0" y1="3" x2="16" y2="3" stroke={s.focus ? "var(--accent)" : "var(--ink)"} strokeWidth="2" strokeDasharray={DASHES[i % DASHES.length]} />
                </svg>
                {s.label}
              </span>
            ))}
          </span>
        )}
      </figcaption>
      <svg
        ref={ref}
        width="100%"
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        role="img"
        aria-label={title}
        className="block"
      >
        {yt.map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(log ? 10 ** v : v)} y2={sy(log ? 10 ** v : v)} stroke="var(--line)" strokeWidth="1" />
            <text x={pad.l - 6} y={sy(log ? 10 ** v : v) + 3} textAnchor="end" className="num" fontSize="10" fill="var(--muted)">
              {fmt(log ? 10 ** v : v)}
            </text>
          </g>
        ))}
        {xt.map((v) => (
          <text key={v} x={sx(v)} y={H - 6} textAnchor="middle" className="num" fontSize="10" fill="var(--muted)">
            {fmt(v)}
          </text>
        ))}
        {empty && (
          <text x={pad.l + 4} y={H / 2} fontSize="11" fill="var(--muted)">
            no data yet
          </text>
        )}
        {(() => {
          // direct labels at line ends, nudged apart so they never overlap
          if (series.length > 4) return null;
          const ends = series
            .map((s) => {
              const pts = s.points.filter((p) => Number.isFinite(p[1]));
              const last = pts[pts.length - 1];
              return last ? { id: s.id, y: sy(last[1]), text: series.length > 1 ? s.label : fmt(last[1]) } : null;
            })
            .filter(Boolean) as { id: string; y: number; text: string }[];
          ends.sort((a, b) => a.y - b.y);
          for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 11) ends[k].y = ends[k - 1].y + 11;
          return ends.map((e) => (
            <text key={e.id} x={W - pad.r + 6} y={e.y + 3} fontSize="10" className="num" fill="var(--muted)">
              {e.text.length > 10 ? e.text.slice(0, 10) + "…" : e.text}
            </text>
          ));
        })()}
        {series.map((s, i) => {
          const pts = s.points.filter((p) => Number.isFinite(p[1]));
          if (!pts.length) return null;
          const d = pts.map((p, j) => `${j ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
          const last = pts[pts.length - 1];
          const color = s.focus ? "var(--accent)" : "var(--ink)";
          return (
            <g key={s.id}>
              <path d={d} fill="none" stroke={color} strokeWidth="2" strokeDasharray={DASHES[i % DASHES.length]} strokeLinejoin="round" />
              {pts.length === 1 && <circle cx={sx(last[0])} cy={sy(last[1])} r="4" fill={color} stroke="var(--bg)" strokeWidth="2" />}
            </g>
          );
        })}
        {hover !== null && !empty && (
          <g pointerEvents="none">
            <line x1={sx(hover)} x2={sx(hover)} y1={pad.t} y2={H - pad.b} stroke="var(--muted)" strokeWidth="1" />
            {series.map((s) => {
              const p = nearest(s, hover);
              return p && Number.isFinite(p[1]) ? (
                <circle key={s.id} cx={sx(p[0])} cy={sy(p[1])} r="4" fill={s.focus ? "var(--accent)" : "var(--ink)"} stroke="var(--bg)" strokeWidth="2" />
              ) : null;
            })}
          </g>
        )}
      </svg>
      {hover !== null && !empty && (
        <div className="num pointer-events-none text-label text-muted">
          step {fmt(nearest(series[0], hover)?.[0] ?? hover)}
          {series.map((s) => {
            const p = nearest(s, hover);
            return (
              <span key={s.id} className="ml-3 text-ink">
                {series.length > 1 ? `${s.label.slice(0, 8)} ` : ""}
                {p ? fmt(p[1]) : "–"}
              </span>
            );
          })}
        </div>
      )}
    </figure>
  );
}
