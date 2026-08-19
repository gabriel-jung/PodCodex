/** Tauri (native desktop) platform implementation. */

import { open, save } from "@tauri-apps/plugin-dialog";
import { getCurrentWindow } from "@tauri-apps/api/window";
import type { Platform } from "./types";

export const tauriPlatform: Platform = {
  fs: {
    openFolderDialog: async () => {
      const result = await open({ directory: true });
      return result ?? null;
    },
    openFileDialog: async (extensions?: string[]) => {
      const result = await open({
        filters: extensions
          ? [{ name: "Files", extensions }]
          : undefined,
      });
      return typeof result === "string" ? result : null;
    },
    saveFileDialog: async (opts) => {
      const result = await save({
        defaultPath: opts?.defaultPath,
        filters: opts?.extensions
          ? [{ name: "Files", extensions: opts.extensions }]
          : undefined,
      });
      return result ?? null;
    },
  },
  window: {
    setTitle: (title) => {
      getCurrentWindow().setTitle(title);
    },
    minimize: () => {
      getCurrentWindow().minimize();
    },
    isNative: () => true,
  },
  lifecycle: {
    // DO NOT USE without first granting `core:window:allow-destroy` in
    // `src-tauri/capabilities/default.json`. Registering this at all makes the
    // app unclosable: Tauri's own `onCloseRequested` wrapper calls
    // `window.destroy()` whenever the handler does not `preventDefault`, and
    // `destroy` is not part of the `core:default` permission set — so the
    // close is denied and the window has no way out. Verified the hard way on
    // 2026-08-19. There is currently no consumer; keep it that way unless the
    // permission is added deliberately.
    onBeforeClose: (cb) => {
      const unlistenPromise = getCurrentWindow().onCloseRequested(() => {
        cb();
      });
      return () => {
        unlistenPromise.then((fn) => fn());
      };
    },
  },
  isTauri: true,
};
