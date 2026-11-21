import { expect, test } from "vitest";
import { alignSeries, bandSpan, buildLineOptions, lossYMax, offScaleMarker, toFinite } from "./chartOptions";

test("lossYMax is 1.25x the pre-spike max", () => {
  expect(lossYMax([0, 1, 2, 3], [4, 2, 3, 60], 3)).toBeCloseTo(5);
  expect(lossYMax([0, 1], [4, 2], null)).toBeNull();
  expect(lossYMax([5, 6], [4, 2], 3)).toBeNull();
});

test("offScaleMarker reports peak or non-finite", () => {
  expect(offScaleMarker([0, 1, 2], [1, 9, 63.6], 5)).toEqual({ at: 1, text: "63.6 at step 2, off scale" });
  expect(offScaleMarker([0, 1], [1, NaN], 5)!.text).toBe("non-finite at step 1");
  expect(offScaleMarker([0, 1], [1, 2], 5)).toBeNull();
  expect(offScaleMarker([0, 1], [1, 9], null)).toBeNull();
});

test("a null loss is what NaN looks like on the wire, so it counts as non-finite", () => {
  expect(offScaleMarker([0, 1], [1, null], 5)!.text).toBe("non-finite at step 1");
});

test("bandSpan keeps single-step bands visible", () => {
  expect(bandSpan(100, 100, 3)).toEqual([98.5, 101.5]);
  expect(bandSpan(10, 50, 3)).toEqual([10, 50]);
});

test("toFinite turns NaN and infinities into gaps", () => {
  expect(toFinite([1, NaN, Infinity, null, 2])).toEqual([1, null, null, null, 2]);
});

test("alignSeries joins series on different steps into one x axis", () => {
  const [x, a, b] = alignSeries([
    { id: "a", label: "a", points: [[0, 1], [2, 3]] },
    { id: "b", label: "b", points: [[1, 5], [2, 6]] },
  ]);
  expect(x).toEqual([0, 1, 2]);
  expect(a).toEqual([1, null, 3]);
  expect(b).toEqual([null, 5, 6]);
});

test("options: one y scale, 2px lines, quiet swiss identity by dash, accent for focus", () => {
  const o = buildLineOptions(
    { series: [{ id: "l", label: "loss", points: [[0, 1]] }, { id: "v", label: "val", points: [[0, 2]], focus: true }], height: 200 },
    { width: 600 },
  );
  expect(Object.keys(o.scales!)).toEqual(["x", "y"]);
  expect(o.width).toBe(600);
  expect(o.height).toBe(200);
  expect(o.series[1].width).toBe(2);
  expect(o.series[1].dash).toEqual([]);
  expect(o.series[2].dash).toEqual([6, 3]);
  expect(o.series[2].stroke).not.toBe(o.series[1].stroke);
  expect(o.legend!.show).toBe(false);
});

test("options: log scale uses uPlot's log distribution", () => {
  const o = buildLineOptions({ series: [{ id: "l", label: "loss", points: [[0, 1]] }], height: 100, log: true }, { width: 300 });
  expect(o.scales!.y.distr).toBe(3);
});

test("the y range follows the latest yMax without a rebuild", () => {
  let current: { yMax: number | null } = { yMax: null };
  const o = buildLineOptions({ series: [{ id: "l", label: "loss", points: [[0, 1], [1, 2]] }], height: 100 }, { width: 300 }, () => current);
  const range = o.scales!.y.range as (u: unknown, min: number, max: number) => [number, number];
  expect(range(null, 0, 10)[1]).toBeGreaterThan(10);
  current = { yMax: 5 };
  expect(range(null, 0, 10)).toEqual([0, 5]);
});

test("hasData looks at the plotted data, not the series the chart was built with", async () => {
  const { hasData } = await import("./chartOptions");
  expect(hasData([[], []])).toBe(false);
  expect(hasData([[0, 1], [null, null]])).toBe(false);
  expect(hasData([[0, 1], [null, 2]])).toBe(true);
});
