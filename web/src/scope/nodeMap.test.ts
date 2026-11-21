import { expect, test } from "vitest";
import { nodeFor } from "./nodeMap";

const moduleMap = { blocks: "blocks", fc_1: "fc-1", lm_head: "lm_head" };

test("a layer maps to the node whose module holds it", () => {
  expect(nodeFor("blocks.0.attn.q.weight", moduleMap)).toBe("blocks");
  expect(nodeFor("fc_1.lora_A", moduleMap)).toBe("fc-1");
});

test("unknown modules map to no node", () => {
  expect(nodeFor("unknown.w", moduleMap)).toBeNull();
  expect(nodeFor("", moduleMap)).toBeNull();
});
