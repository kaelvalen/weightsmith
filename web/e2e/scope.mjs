// A diverging run is diagnosed in the Diagnose tab; a run without a recording says so.
import { api, check, open } from "./lib.mjs";

const graph = {
  schema: 1, name: "e2e spiky",
  nodes: [
    { id: "x", type: "Input", params: { shape: ["B", 8] } },
    { id: "fc1", type: "Linear", params: { in: 8, out: 16 } },
    { id: "act", type: "GELU" },
    { id: "fc2", type: "Linear", params: { in: 16, out: 3 } },
    { id: "loss", type: "CrossEntropy" },
  ],
  edges: [["x.out", "fc1.in"], ["fc1.out", "act.in"], ["act.out", "fc2.in"], ["fc2.out", "loss.logits"]],
};
const data = { source: "synthetic", kind: "classification", n: 512, dim: 8, classes: 3 };
// sgd at lr 1.0 without clipping spikes repeatedly on this data but stays finite
const train = { epochs: 8, batch_size: 32, log_every: 2, device: "cpu", precision: "f32", optimizer: { name: "sgd", lr: 1.0 },
                scheduler: { name: "constant" }, grad_clip: null };
await api("/api/projects/e2e-spiky", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({
  name: "e2e spiky", active: "main", variants: { main: { graph, parent: null, created: Date.now() } }, job: { data, train } }) });

const { browser, page, errors } = await open("/?project=e2e-spiky");
await page.keyboard.press("Control+Enter");
for (let i = 0; i < 90 && !(await page.getByText(/^completed$/).count()); i++) await page.waitForTimeout(500);
check(await page.getByText(/^completed$/).count() > 0, "spiky run completes");

await page.locator("button", { hasText: /^Diagnose$/ }).click();
await page.waitForTimeout(1500);
const lossFig = page.locator("[data-testid=diagnose] figure", { hasText: "loss" }).first();
check(Number(await lossFig.getAttribute("data-points").catch(() => 0)) > 0, "Overview draws the loss");
const heat = await page.locator("[data-testid=layer-heatmap] canvas").boundingBox();
check(heat !== null, "Overview draws the layer heatmap");
check(heat !== null && heat.y < 900, "the heatmap is on screen without scrolling");
check(await page.locator("[data-signal]").count() > 0, "Overview lists the signals that crossed");

// the canvas link: a heatmap row selects its node; painting and the spike source show on the graph
const spiky = (await api("/api/runs")).find((r) => r.project === "e2e-spiky");
const matrix = await api(`/api/scope/runs/${spiky.id}/layers/matrix`);
const moduleMap = await api(`/api/runs/${spiky.id}/files/module_map.json`);
const firstNode = moduleMap[matrix.layers[0].split(".")[0]];
const hb = await page.locator("[data-testid=layer-heatmap] canvas").boundingBox();
await page.mouse.click(hb.x + 140, hb.y + 8);
await page.waitForTimeout(500);
check(await page.locator(`.react-flow__node[data-id="${firstNode}"].selected`).count() === 1, `a heatmap row selects its node (${firstNode})`);
check(await page.getByText("Layer metrics").count() > 0, "the Inspector shows the node's layer metrics");
await page.locator("button", { hasText: /^paint canvas$/ }).click();
await page.waitForTimeout(300);
check(await page.locator("[data-paint]").count() > 0, "painting marks nodes on the canvas");
check(await page.locator("[data-spike-source]").count() === 1, "exactly one node is marked as the spike source");

await page.locator("[data-testid=diagnose] button", { hasText: /^Spikes$/ }).click();
await page.waitForTimeout(800);
check(await page.locator("[data-spike]").count() > 0, "Spikes lists the spike events");

// a second run of the same graph with recording off: same graph, different seed
const plain = await api("/api/runs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({
  graph, data, train: { ...train, seed: 7, diagnostics: { trainscope: false } }, project: "e2e-spiky" }) });
for (let i = 0; i < 90; i++) {
  if ((await api(`/api/runs/${plain.id}`)).status === "completed") break;
  await new Promise((r) => setTimeout(r, 500));
}
await page.locator("button", { hasText: /^Runs$/ }).click();
await page.waitForTimeout(1500);
await page.locator("button", { hasText: plain.id.slice(0, 8) }).first().click();
await page.locator("button", { hasText: /^Diagnose$/ }).click();
await page.waitForTimeout(1000);
check(await page.getByText("No diagnostics were recorded for this run.").count() > 0, "a run without a recording shows the empty state");
check(errors.length === 0, `no page errors (${errors.join(" | ")})`);
await browser.close();
