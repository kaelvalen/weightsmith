// Shared helpers for the browser scenarios (driven by scripts/e2e.sh).
import { chromium } from "playwright-core";
import { existsSync } from "node:fs";

export const BASE = process.env.WS_URL ?? "http://127.0.0.1:7899";

export function chromePath() {
  const candidates = [process.env.CHROME, "/etc/profiles/per-user/" + process.env.USER + "/bin/google-chrome", "/usr/bin/google-chrome",
    "/usr/bin/chromium", "/usr/bin/chromium-browser", "/run/current-system/sw/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"];
  return candidates.find((p) => p && existsSync(p));
}

export async function open(path = "/") {
  const browser = await chromium.launch({ executablePath: chromePath(), headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, permissions: ["clipboard-read", "clipboard-write"] });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));
  await page.goto(BASE + path);
  await page.waitForTimeout(1200);
  return { browser, page, errors };
}

export async function command(page, text) {
  await page.keyboard.press("Control+k");
  await page.waitForTimeout(150);
  await page.keyboard.type(text);
  await page.waitForTimeout(150);
  await page.keyboard.press("Enter");
  await page.waitForTimeout(700);
}

export async function api(path, init) {
  const r = await fetch(BASE + path, init);
  return r.json();
}

export function check(cond, msg) {
  if (!cond) {
    console.error("FAIL", msg);
    process.exitCode = 1;
  } else console.log("ok  ", msg);
}
