/** How long to wait before refetching, at most once per `period` ms; `last` null = never fetched. */
export function refetchDelay(now: number, last: number | null, period: number): number {
  return last == null ? 0 : Math.max(0, last + period - now);
}
