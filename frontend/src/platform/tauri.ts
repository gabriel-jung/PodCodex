/** Tauri (native desktop) platform implementation. */

import { open } from "@tauri-apps/plugin-dialog";
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
    onBeforeClose: (cb) => {
      const unlistenPromise = getCurrentWindow().onCloseRequested(async () => {
        cb();
      });
      return () => {
        unlistenPromise.then((fn) => fn());
      };
    },
  },
  isTauri: true,
};
