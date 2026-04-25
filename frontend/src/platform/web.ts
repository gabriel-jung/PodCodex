/** Web (browser) platform implementation — noop/browser fallbacks. */

import type { Platform } from "./types";

export const webPlatform: Platform = {
  fs: {
    openFolderDialog: async () => null,
    openFileDialog: async () => null,
    saveFileDialog: async () => null,
  },
  window: {
    setTitle: (title) => { document.title = title; },
    minimize: () => {},
    isNative: () => false,
  },
  lifecycle: {
    onBeforeClose: (cb) => {
      window.addEventListener("beforeunload", cb);
      return () => window.removeEventListener("beforeunload", cb);
    },
  },
  isTauri: false,
};
