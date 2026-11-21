// Mixture-of-experts / addressing: each block's routing shares over time. A share near 1 means
// one expert (or memory slot) is taking the tokens: the collapse trainscope warns about.
import { useMemo } from "react";
import { LineChart } from "../ui/LineChart";

export function Routing({ moe }: { moe: Record<string, any>[] }) {
  const blocks = useMemo(() => {
    const by = new Map<string, Record<string, any>[]>();
    for (const r of moe) (by.get(r.block) ?? by.set(r.block, []).get(r.block)!).push(r);
    return [...by.entries()];
  }, [moe]);
  if (!blocks.length) return <p className="m-0 p-4 text-small text-muted">This model has no router or addressor to show.</p>;
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(280px,1fr))] gap-x-6 gap-y-2 px-4 py-2">
      {blocks.map(([block, rows]) => {
        const n = Math.max(...rows.map((r) => (r.shares ?? []).length));
        const series = Array.from({ length: n }, (_, k) => ({
          id: `e${k}`,
          label: `#${k}`,
          points: rows.filter((r) => typeof r.shares?.[k] === "number").map((r) => [r.step, r.shares[k]] as [number, number]),
        }));
        const max = { id: "max", label: "max share", focus: true, points: rows.map((r) => [r.step, Math.max(...(r.shares ?? [0]))] as [number, number]) };
        return <LineChart key={block} title={`${block} · routing shares`} series={n <= 4 ? [...series, max] : [max]} height={130} />;
      })}
    </div>
  );
}
