/** Global task bar state — tracks active download / batch tasks across pages.
 *  Persisted so in-flight tasks can reconnect to the backend after the frontend reloads. */

import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface BatchEpisode {
  title: string;
  stem: string;
}

interface EpisodeTaskInfo {
  stem: string;
  folder: string;
  title: string;
  step: string;
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

  /** Active single-episode pipeline task (one at a time). Survives panel
   *  unmount so the global task bar keeps the progress visible after the
   *  user navigates elsewhere. */
  episodeTaskId: string | null;
  episodeStem: string | null;
  episodeFolder: string | null;
  episodeTitle: string | null;
  episodeStep: string | null;
  setEpisodeTask: (taskId: string | null, info?: EpisodeTaskInfo) => void;
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

      episodeTaskId: null,
      episodeStem: null,
      episodeFolder: null,
      episodeTitle: null,
      episodeStep: null,
      setEpisodeTask: (taskId, info) =>
        set({
          episodeTaskId: taskId,
          episodeStem: info?.stem ?? null,
          episodeFolder: info?.folder ?? null,
          episodeTitle: info?.title ?? null,
          episodeStep: info?.step ?? null,
        }),
    }),
    { name: "podcodex-tasks" },
  ),
);
