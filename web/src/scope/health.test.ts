import { expect, test } from "vitest";
import { runHealth } from "./health";

test("health reads the recording summary", () => {
  expect(runHealth(undefined, "completed")).toBe("—");
  expect(runHealth({ name: "a", spike_count: 0, last_loss_finite: false }, "failed")).toBe("diverged");
  expect(runHealth({ name: "a", spike_count: 3, last_loss_finite: true }, "completed")).toBe("spiked");
  expect(runHealth({ name: "a", spike_count: 0, last_loss_finite: true }, "running")).toBe("training");
  expect(runHealth({ name: "a", spike_count: 0, last_loss_finite: true }, "completed")).toBe("stable");
});
