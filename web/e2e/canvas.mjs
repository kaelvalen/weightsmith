// Projects → new project from a model → block insert → copy/paste → validation stays green.
import { api, check, command, open } from "./lib.mjs";

const { browser, page, errors } = await open("/");
check(await page.locator("h1", { hasText: "Projects" }).count() === 1, "projects tab is the start page");
await page.locator("button", { hasText: "GPT-2 small" }).first().click();
await page.waitForTimeout(1500);
// the canvas renders only what is on screen, so count through the API
const slug0 = new URL(page.url()).searchParams.get("project");
await page.waitForTimeout(600); // the project is saved shortly after it is created
const g0 = (await api(`/api/projects/${slug0}`)).variants.main.graph;
check(g0.nodes.length === 8 && g0.edges.length === 7, `GPT-2 opens with its 8 nodes and 7 edges (got ${g0.nodes.length}, ${g0.edges.length})`);
await page.keyboard.press("f");
await page.waitForTimeout(400);
const nodes = await page.locator(".react-flow__node").count();
check(await page.getByText("B×T×768").count() > 0, "port shapes are shown");
check(await page.getByText("124.44M").count() > 0, "parameter count matches GPT-2 small");

// regression: inserting a block used to write a 'schema' key into the subgraph and lose all ports
await page.locator("nav[aria-label='Node library'] >> text=Llama / Qwen layer").click();
await page.waitForTimeout(1200);
const slug = new URL(page.url()).searchParams.get("project");
const project = await api(`/api/projects/${slug}`);
const v = await api("/api/validate", { method: "POST", headers: { "content-type": "application/json" },
  body: JSON.stringify({ graph: project.variants.main.graph }) });
check(!v.issues.some((i) => i.code === "schema"), "graph with an inserted block passes the daemon's schema check");
check(await page.getByText("B×T×768").count() > 0, "ports still show shapes after inserting a block");

await page.keyboard.press("Control+c");
await page.keyboard.press("Control+v");
await page.waitForTimeout(400);
check(await page.locator(".react-flow__node").count() === nodes + 2, "copy + paste adds a node");

await page.keyboard.press("f");
await page.waitForTimeout(400);
await page.locator(".react-flow__node", { hasText: "12× block" }).getByRole("button", { name: "inside" }).click();
await page.waitForTimeout(300);
check(await page.locator(".react-flow__node", { hasText: "12× block" }).getByText("attn").count() > 0, "block peek lists the layers inside");

await command(page, "go to ln_f");
check(await page.locator(".react-flow__node.selected", { hasText: "ln_f" }).count() === 1, "go to node selects it");

// one language for dialogs, selects and tooltips
await page.keyboard.press("Control+k");
await page.waitForTimeout(250);
check(await page.locator("[role=dialog]").count() === 1, "the palette opens as a dialog");
await page.keyboard.press("Escape");
await page.waitForTimeout(200);
check(await page.locator("[role=dialog]").count() === 0, "Esc closes it");
await page.locator("nav[aria-label='Node library'] input").click();
await page.keyboard.press("Control+k");
await page.waitForTimeout(250);
check(await page.locator("[role=dialog]").count() === 1, "Ctrl+K works while typing in a field");
await page.keyboard.press("Escape");
await page.locator("[aria-label=variant]").focus();
await page.keyboard.press("ArrowDown");
await page.waitForTimeout(200);
check(await page.locator("[role=listbox] [role=option]").count() >= 2, "the variant select opens its own list from the keyboard");
await page.keyboard.press("Escape");
await page.locator("nav[aria-label='Node library'] >> text=ResNet-18").hover();
await page.waitForTimeout(700);
check(await page.locator("[role=tooltip].ws-tip").count() === 1, "tooltips use the interface's own box");
check(await page.locator("nav[aria-label='Node library'] [title]").count() === 0, "no native title boxes left in the library");

check(errors.length === 0, `no page errors (${errors.join(" | ")})`);
await browser.close();
