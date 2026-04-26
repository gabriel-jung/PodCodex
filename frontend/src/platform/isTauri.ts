// Tauri 2.x always sets __TAURI_INTERNALS__; the legacy __TAURI__ only appears
// when `withGlobalTauri` is enabled. Check the internals symbol so detection
// works regardless of that config flag.
export const isTauri = (): boolean => {
  const w = window as unknown as {
    __TAURI_INTERNALS__?: unknown;
    __TAURI__?: unknown;
  };
  return !!(w.__TAURI_INTERNALS__ ?? w.__TAURI__);
};
