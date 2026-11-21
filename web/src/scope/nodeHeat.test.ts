import { expect, test } from "vitest";
import { layersByNode, nodeHeat, spikeSourceNode } from "./nodeHeat";
import type { LayerMatrix } from "./types";

const moduleMap = { fc1: "fc1", fc2: "fc2", loss_: "loss" };
const matrix: LayerMatrix = {
  layers: ["fc1.base.weight", "fc1.lora_B", "fc2.weight", "other.w"],
  bucket_starts: [0, 10, 20, 30],
  // fc1.base is frozen: its grads are all 0, whatever its z says
  values: [
    [0, 0, 0, 0],
    [1, 1, 5, 9],
    [1, 2, 9, 9],
    [1, 1, 1, 1],
  ],
  z: [
    [0, 9, 9, 9],
    [0, 0, 1, 4],
    [0, 1, 5, 5],
    [0, 0, 0, 0],
  ],
};

test("a node's heat is the largest z of its trainable layers over the run", () => {
  expect(nodeHeat(matrix, moduleMap)).toEqual({ fc1: 4, fc2: 5 });
});

test("the spike source is the node whose trainable layer crossed 3σ first, up to the spike", () => {
  expect(spikeSourceNode(matrix, moduleMap, { from: 20, to: 25 })).toBe("fc2");
  expect(spikeSourceNode(matrix, moduleMap, { from: 5, to: 9 })).toBeNull();
  expect(spikeSourceNode(matrix, moduleMap, null)).toBeNull();
});

test("a frozen layer never makes its node the spike source", () => {
  const frozenOnly: LayerMatrix = { ...matrix, layers: ["fc1.base.weight"], values: [matrix.values[0]], z: [matrix.z[0]] };
  expect(spikeSourceNode(frozenOnly, moduleMap, { from: 0, to: 40 })).toBeNull();
});

test("layers are listed under the node that holds them", () => {
  expect(layersByNode(matrix.layers, moduleMap)).toEqual({ fc1: ["fc1.base.weight", "fc1.lora_B"], fc2: ["fc2.weight"] });
});
