// Renders core/weightsmith_core/diff.py output: what changed between two graphs.

const val = (v: unknown) => (v === undefined || v === null ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v));

function adaptSummary(list: any[]): string {
  if (!list?.length) return "none";
  return list.map((a) => `${a.kind}${a.kind === "lora" ? ` r${a.r} ${a.target}` : a.kind === "adapter" ? ` ${a.bottleneck}` : ""}${a.disabled ? " (off)" : ""}`).join(", ");
}

export function DiffList({ diff }: { diff: { same: boolean; changes: any[] } }) {
  if (diff.same) return <p className="m-0 text-small text-muted">Same graph.</p>;
  return (
    <ul className="num m-0 list-none p-0 text-label">
      {diff.changes.map((c, i) => {
        const sign = c.kind.endsWith("added") ? "+" : c.kind.endsWith("removed") ? "−" : "~";
        let text: string;
        switch (c.kind) {
          case "node_added":
          case "node_removed":
            text = `${c.node} ${c.type}`;
            break;
          case "param_changed":
            text = `${c.node}.${c.key} ${val(c.from)} → ${val(c.to)}`;
            break;
          case "adapt_changed":
            text = `${c.node} adapt ${adaptSummary(c.from)} → ${adaptSummary(c.to)}`;
            break;
          case "edge_added":
          case "edge_removed":
            text = `${c.edge[0]} → ${c.edge[1]}`;
            break;
          case "subgraph_added":
          case "subgraph_removed":
            text = `block ${c.subgraph}`;
            break;
          default:
            text = `${c.node} ${c.kind.replace("_changed", "")} ${val(c.from)} → ${val(c.to)}`;
        }
        return (
          <li key={i} className="flex gap-2">
            <span className="w-2 text-muted">{sign}</span>
            <span className="min-w-0 break-all">{text}</span>
          </li>
        );
      })}
    </ul>
  );
}
