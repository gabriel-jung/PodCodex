/** Persisted user preferences — survives page reloads and app restarts. */

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ConfigState {
  /** Playback speed (0.5 - 3.0). */
  playbackSpeed: number;
  setPlaybackSpeed: (speed: number) => void;
  /** Hide episodes shorter than this (minutes). 0 = show all. */
  minDurationMinutes: number;
  setMinDurationMinutes: (min: number) => void;
}

export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      playbackSpeed: 1,
      setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),
      minDurationMinutes: 0,
      setMinDurationMinutes: (min) => set({ minDurationMinutes: min }),
    }),
    { name: "podcodex-config" },
  ),
);
