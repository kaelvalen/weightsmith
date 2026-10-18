// The TS mirror must agree with the Python analyzer (the source of truth) on
// every fixture: validity, error locations, port shapes and parameter counts.
import { describe, expect, it } from "vitest";
import fixtures from "./fixtures.json";
import { analyze } from "./analyze";
import type { Graph } from "./types";

interface Fixture {
  name: string;
  graph: Graph;
  expected: { ok: boolean; errors: string[]; ports: Record<string, unknown>; params: Record<string, [number, number]> };
}

describe("TS analyzer mirrors Python", () => {
  for (const f of fixtures as unknown as Fixture[]) {
    it(f.name, () => {
      const a = analyze(f.graph);
      const errors = [...new Set(a.issues.filter((i) => i.severity === "error").map((i) => `${i.code}@${i.node ?? ""}.${i.port ?? ""}`))].sort();
      expect(errors).toEqual(f.expected.errors);
      expect(a.ok).toBe(f.expected.ok);
      expect(Object.fromEntries(Object.entries(a.ports).sort(([x], [y]) => (x < y ? -1 : 1)))).toEqual(f.expected.ports);
      expect(Object.fromEntries(Object.entries(a.params).map(([k, v]) => [k, [v.total, v.trainable]]))).toEqual(f.expected.params);
    });
  }
});
