// Building a model by clicking: empty state → chained adds (wired + sized) → splice → heal on
// delete → one-click loss fix → readiness checklist with a data preview → help page.
import { api, check, open, stage } from "./lib.mjs";

const { browser, page, errors } = await open("/");
const lib = (name) => page.locator("nav[aria-label='Node library']").getByText(name, { exact: true }).first();
const graph = async () => {
  await page.waitForTimeout(900); // saved shortly after each edit
  const slug = new URL(page.url()).searchParams.get("project");
  return (await api(`/api/projects/${slug}`)).variants.main.graph;
};

await page.getByRole("button", { name: "New project" }).click();
await page.waitForTimeout(800);
check(await page.getByText("Empty graph").count() === 1, "a blank project explains how to start");
await page.getByRole("button", { name: "Add an input" }).click();
await page.waitForTimeout(400);
check(await page.getByText("click a library item to add it after").count() === 1, "hint bar follows the selection");

await lib("Linear").click();
await page.waitForTimeout(300);
await lib("ReLU").click();
await page.waitForTimeout(300);
await lib("Linear").click();
let g = await graph();
const input = g.nodes.find((n) => n.type === "Input");
const linears = g.nodes.filter((n) => n.type === "Linear");
const width = input.params?.shape?.at(-1);
check(g.nodes.length === 4 && g.edges.length === 3, `clicking library items builds a connected chain (${g.nodes.length} nodes, ${g.edges.length} edges)`);
check(typeof width !== "number" || linears[0].params?.in === width, `the first Linear is sized from the input (in = ${linears[0].params?.in}, input ${width})`);
check(linears[1].params?.in === linears[0].params?.out, "the second Linear follows the first");

// remove the ReLU: the chain heals
const relu = g.nodes.find((n) => n.type === "ReLU");
await page.locator(`.react-flow__node[data-id='${relu.id}']`).click();
await page.keyboard.press("Delete");
g = await graph();
check(g.nodes.length === 3 && g.edges.length === 2, `deleting a middle node keeps the chain connected (${g.edges.length} edges)`);

// splice a GELU back into the edge between the two Linears
await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
const edge = page.locator(".react-flow__edge").nth(1);
await edge.click({ force: true });
await page.waitForTimeout(200);
await lib("GELU").click();
g = await graph();
check(g.nodes.some((n) => n.type === "GELU") && g.edges.length === 3, "a library item inserts into the selected connection");

// overview tab: the missing loss has a one-click fix
await page.keyboard.press("Escape");
await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
await page.waitForTimeout(400);
check(await page.getByText("Ready to train?").count() === 1, "with nothing selected the inspector says what is missing");
const fix = page.getByRole("button", { name: /^Add CrossEntropy after/ });
check(await fix.count() > 0, "the missing loss offers a fix");
await fix.first().click();
g = await graph();
check(g.nodes.some((n) => n.type === "CrossEntropy"), "the fix adds the loss after the output");

await stage(page, "Data");
await page.waitForTimeout(2500);
check(await page.getByText("What the model will see").count() === 1, "the data tab previews the dataset");
check(await page.locator("li", { hasText: "loss target" }).count() > 0, "the preview checks the loss target field");
const fit = page.getByRole("button", { name: /^Set in\d+ to \[32\]$/ });
check(await fit.count() === 1, "a data/input mismatch offers to fit the input");
await fit.click();
g = await graph();
await page.waitForTimeout(1500);
const first = g.nodes.find((n) => n.type === "Linear" && g.edges.some(([s, d]) => s.startsWith(input.id + ".") && d.startsWith(n.id + ".")));
check(first?.params?.in === 32, `fitting the input re-sizes what follows (in = ${first?.params?.in})`);
check(await page.locator("li", { hasText: "✕" }).count() === 0, "after the fix the data fits the graph");

await page.keyboard.press("3");
await page.waitForTimeout(400);
check(await page.getByText("Ready to train?").count() === 1 && await page.getByText("optimizer", { exact: true }).count() > 0, "key 3 opens Train: settings next to the checklist");
await stage(page, "Model");
await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
await page.keyboard.press("?");
await page.waitForTimeout(300);
check(await page.getByText("How Weightsmith works").count() > 0, "? opens the help page");
await page.keyboard.press("Escape");

check(errors.length === 0, `no page errors${errors.length ? ": " + errors.join(" | ") : ""}`);
await browser.close();
