/** Global task bar state — tracks active download / batch tasks across pages.
 *  Persisted so in-flight tasks can reconnect to the backend after the frontend reloads. */

import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface BatchEpisode {
  title: string;
  stem: string;
}

interface TaskBarState {
  /** Active download task (one at a time). */
  downloadTaskId: string | null;
  downloadFolder: string | null;
  setDownloadTask: (taskId: string | null, folder?: string | null) => void;

  /** Active batch pipeline task (one at a time). */
  batchTaskId: string | null;
  batchFolder: string | null;
  batchEpisodes: BatchEpisode[];
  batchStep: string | null;
  setBatchTask: (taskId: string | null, folder?: string | null, episodes?: BatchEpisode[], step?: string | null) => void;
}

export const useTaskStore = create<TaskBarState>()(
  persist(
    (set) => ({
      downloadTaskId: null,
      downloadFolder: null,
      setDownloadTask: (taskId, folder = null) =>
        set({ downloadTaskId: taskId, downloadFolder: folder }),

      batchTaskId: null,
      batchFolder: null,
      batchEpisodes: [],
      batchStep: null,
      setBatchTask: (taskId, folder = null, episodes = [], step = null) =>
        set({ batchTaskId: taskId, batchFolder: folder, batchEpisodes: episodes, batchStep: step }),
    }),
    { name: "podcodex-tasks" },
  ),
);
