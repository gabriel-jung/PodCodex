export const isTauri = (): boolean =>
  !!(window as unknown as { __TAURI__?: unknown }).__TAURI__;
