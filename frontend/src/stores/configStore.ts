/** Persisted user preferences — survives page reloads and app restarts. */

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ConfigState {
  /** Playback speed (0.5 - 3.0). */
  playbackSpeed: number;
  setPlaybackSpeed: (speed: number) => void;
  /** Hide episodes shorter than this (minutes). 0 = no minimum. */
  minDurationMinutes: number;
  setMinDurationMinutes: (min: number) => void;
  /** Hide episodes longer than this (minutes). 0 = no maximum. */
  maxDurationMinutes: number;
  setMaxDurationMinutes: (max: number) => void;
  /** Only show episodes whose title contains this text. Empty = no filter. */
  titleInclude: string;
  setTitleInclude: (v: string) => void;
  /** Hide episodes whose title contains this text. Empty = no filter. */
  titleExclude: string;
  setTitleExclude: (v: string) => void;
  /** Show view mode on the home page. */
  showViewMode: "list" | "card";
  setShowViewMode: (mode: "list" | "card") => void;
  /** Card size (columns) for the show grid. 1-5. */
  showCardSize: number;
  setShowCardSize: (size: number) => void;
}

export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      playbackSpeed: 1,
      setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),
      minDurationMinutes: 0,
      setMinDurationMinutes: (min) => set({ minDurationMinutes: min }),
      maxDurationMinutes: 0,
      setMaxDurationMinutes: (max) => set({ maxDurationMinutes: max }),
      titleInclude: "",
      setTitleInclude: (v) => set({ titleInclude: v }),
      titleExclude: "",
      setTitleExclude: (v) => set({ titleExclude: v }),
      showViewMode: "card",
      setShowViewMode: (mode) => set({ showViewMode: mode }),
      showCardSize: 3,
      setShowCardSize: (size) => set({ showCardSize: size }),
    }),
    { name: "podcodex-config" },
  ),
);
