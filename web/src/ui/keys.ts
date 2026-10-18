/** For text fields: keep typing keys inside the field, but let the app-wide shortcuts
 * (Cmd/Ctrl+K for the palette, Cmd/Ctrl+Enter to run) reach the window. */
export function keepTyping(e: React.KeyboardEvent) {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && (e.key.toLowerCase() === "k" || e.key === "Enter")) return;
  e.stopPropagation();
}
