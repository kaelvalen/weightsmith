/** The graph node holding a trainscope layer: its first name segment is the module attribute (module_map.json). */
export function nodeFor(layer: string, moduleMap: Record<string, string>): string | null {
  const head = layer.split(".")[0];
  return head && Object.prototype.hasOwnProperty.call(moduleMap, head) ? moduleMap[head] : null;
}
