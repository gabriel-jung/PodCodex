// Tauri command issued by the Rust shell to relaunch the desktop app and
// its sidecar. No-op in pure web (dev preview, tests) so callers can fire
// without feature-detecting at every site.
export async function restartApp(): Promise<void> {
  const w = window as unknown as { __TAURI__?: unknown };
  if (!w.__TAURI__) return;
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("restart_app");
}
