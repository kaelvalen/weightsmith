import { expect, test } from "vitest";
import { refetchDelay } from "./refetch";

test("the first fetch is immediate", () => {
  expect(refetchDelay(10_000, null, 5000)).toBe(0);
});

test("steps arriving every second still refetch every period, not never", () => {
  expect(refetchDelay(12_000, 10_000, 5000)).toBe(3000);
  expect(refetchDelay(16_000, 10_000, 5000)).toBe(0);
});
