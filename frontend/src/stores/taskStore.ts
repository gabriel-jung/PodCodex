/** Global task bar state — tracks active download / batch tasks across pages. */

import { create } from "zustand";

interface TaskBarState {
  /** Active download task (one at a time). */
  downloadTaskId: string | null;
  downloadFolder: string | null;
  setDownloadTask: (taskId: string | null, folder?: string | null) => void;

  /** Active batch pipeline task (one at a time). */
  batchTaskId: string | null;
  batchFolder: string | null;
  setBatchTask: (taskId: string | null, folder?: string | null) => void;
}

export const useTaskStore = create<TaskBarState>()((set) => ({
  downloadTaskId: null,
  downloadFolder: null,
  setDownloadTask: (taskId, folder = null) =>
    set({ downloadTaskId: taskId, downloadFolder: folder }),

  batchTaskId: null,
  batchFolder: null,
  setBatchTask: (taskId, folder = null) =>
    set({ batchTaskId: taskId, batchFolder: folder }),
}));
