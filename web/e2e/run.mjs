// A small project trains from the canvas; metrics stream in; the run lands in Runs.
import { api, check, open } from "./lib.mjs";

const graph = {
  schema: 1, name: "e2e tiny",
  nodes: [
    { id: "x", type: "Input", params: { shape: ["B", 8] } },
    { id: "fc1", type: "Linear", params: { in: 8, out: 16 } },
    { id: "act", type: "GELU" },
    { id: "fc2", type: "Linear", params: { in: 16, out: 3 } },
    { id: "loss", type: "CrossEntropy" },
  ],
  edges: [["x.out", "fc1.in"], ["fc1.out", "act.in"], ["act.out", "fc2.in"], ["fc2.out", "loss.logits"]],
};
await api("/api/projects/e2e-tiny", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({
  name: "e2e tiny", active: "main", variants: { main: { graph, parent: null, created: Date.now() } },
  job: { data: { source: "synthetic", kind: "classification", n: 512, dim: 8, classes: 3 },
         train: { epochs: 3, batch_size: 32, log_every: 2, device: "cpu", precision: "f32" } } }) });

const { browser, page, errors } = await open("/?project=e2e-tiny");
await page.keyboard.press("Control+Enter");
for (let i = 0; i < 60; i++) {
  if (await page.getByText(/^completed$/).count()) break;
  await page.waitForTimeout(500);
}
check(await page.getByText(/^completed$/).count() > 0, "run completes from the canvas");
const lossFig = page.locator("figure", { hasText: "loss" }).first();
check((await lossFig.locator("canvas").count()) > 0 && Number(await lossFig.getAttribute("data-points")) > 0, "loss curve is drawn");
await page.locator("button", { hasText: /^Runs$/ }).click();
await page.waitForTimeout(500);
check(await page.locator("tbody tr").count() >= 1, "run is listed in Runs");
check(errors.length === 0, `no page errors (${errors.join(" | ")})`);
await browser.close();
