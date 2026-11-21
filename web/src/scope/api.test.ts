import { afterEach, expect, test, vi } from "vitest";
import { scopeApi, scopeMulti } from "./api";

const calls: string[] = [];
function mockFetch(status: number, body: unknown) {
  globalThis.fetch = vi.fn(async (url: string | URL | Request) => {
    calls.push(String(url));
    return new Response(JSON.stringify(body), { status });
  }) as typeof fetch;
}
afterEach(() => {
  calls.length = 0;
});

test("per-run calls go to /runs/<run>/… under the base", async () => {
  mockFetch(200, [{ step: 0, loss: 1 }]);
  const rows = await scopeApi("/api/scope", "abc123").global();
  expect(rows).toEqual([{ step: 0, loss: 1 }]);
  expect(calls[0]).toBe("/api/scope/runs/abc123/global");
});

test("layer names are encoded and queries kept", async () => {
  mockFetch(200, []);
  const api = scopeApi("/api", "run a");
  await api.layer("blocks.0/attn");
  await api.layerMatrix({ metric: "grad_l2_norm", buckets: 300 });
  expect(calls).toEqual(["/api/runs/run%20a/layers/blocks.0%2Fattn", "/api/runs/run%20a/layers/matrix?metric=grad_l2_norm&buckets=300"]);
});

test("a run without a recording is null, not an error", async () => {
  mockFetch(404, { detail: "Run 'x' not found" });
  expect(await scopeApi("/api/scope", "x").global()).toBeNull();
});

test("other failures throw", async () => {
  mockFetch(500, { detail: "boom" });
  await expect(scopeApi("/api/scope", "x").global()).rejects.toThrow("boom");
});

test("multi-run calls take run lists", async () => {
  mockFetch(200, {});
  const m = scopeMulti("/api/scope");
  await m.compare(["a", "b"]);
  await m.clusters();
  await m.clusters(["a"]);
  await m.counterexample("a");
  expect(calls).toEqual(["/api/scope/compare?runs=a%2Cb", "/api/scope/cluster", "/api/scope/cluster?runs=a", "/api/scope/counterexample?run=a"]);
});
