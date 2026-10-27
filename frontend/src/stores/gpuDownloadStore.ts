/** The GPU backend download in flight. Kept outside the Settings panel so
 *  leaving Settings mid-download keeps its progress, its error and the
 *  restart after an update; `useGpuDownloadWatcher` (mounted in RootLayout)
 *  finishes it. */

import { create } from "zustand";

interface GpuDownloadState {
  taskId: string | null;
  /** Whether the GPU backend was active when this download started (the
   *  update flow): then completion replaced a running binary and the app
   *  restarts. Kept here, not read from the status query at completion,
   *  whose cache entry is gone once Settings has been closed a while. */
  wasActivated: boolean;
  /** Error of the most recent attempt. Outlives the task, so a fast failure
   *  is still on screen when the user looks. */
  lastError: string | null;
  start: (taskId: string, wasActivated: boolean) => void;
  finish: (error: string | null) => void;
  setError: (error: string | null) => void;
}

export const useGpuDownloadStore = create<GpuDownloadState>()((set) => ({
  taskId: null,
  wasActivated: false,
  lastError: null,
  start: (taskId, wasActivated) => set({ taskId, wasActivated, lastError: null }),
  finish: (error) => set({ taskId: null, lastError: error }),
  setError: (lastError) => set({ lastError }),
}));
