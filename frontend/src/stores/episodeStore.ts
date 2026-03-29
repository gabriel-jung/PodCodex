/** Current episode context — set by EpisodePage, read by all pipeline panels. */

import { create } from "zustand";
import type { Episode, ShowMeta } from "@/api/types";

interface EpisodeState {
  episode: Episode | null;
  showMeta: ShowMeta | null;
  setEpisode: (episode: Episode | null) => void;
  setShowMeta: (meta: ShowMeta | null) => void;
}

export const useEpisodeStore = create<EpisodeState>((set) => ({
  episode: null,
  showMeta: null,
  setEpisode: (episode) => set({ episode }),
  setShowMeta: (meta) => set({ showMeta: meta }),
}));
