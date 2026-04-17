/**
 * Episode context + persisted episode-list filters.
 *
 * Runtime fields (episode, showMeta, folder) are ephemeral — set by
 * EpisodePage / ShowPage when navigating. Filter fields persist across
 * sessions via the `podcodex-episode-filters` storage key.
 */

import { create } from "zustand";
import { persist, type PersistOptions } from "zustand/middleware";
import type { Episode, ShowMeta } from "@/api/types";

interface EpisodeState {
  // ── Runtime context (not persisted) ──
  episode: Episode | null;
  showMeta: ShowMeta | null;
  folder: string | null;
  setEpisode: (episode: Episode | null, folder?: string | null) => void;
  setShowMeta: (meta: ShowMeta | null) => void;

  // ── Episode-list filters (persisted) ──
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
}

const persistOptions: PersistOptions<EpisodeState, Pick<EpisodeState,
  "minDurationMinutes" | "maxDurationMinutes" | "titleInclude" | "titleExclude"
>> = {
  name: "podcodex-episode-filters",
  partialize: (s) => ({
    minDurationMinutes: s.minDurationMinutes,
    maxDurationMinutes: s.maxDurationMinutes,
    titleInclude: s.titleInclude,
    titleExclude: s.titleExclude,
  }),
};

export const useEpisodeStore = create<EpisodeState>()(
  persist(
    (set) => ({
      episode: null,
      showMeta: null,
      folder: null,
      setEpisode: (episode, folder) => set({ episode, folder: folder ?? null }),
      setShowMeta: (meta) => set({ showMeta: meta }),

      minDurationMinutes: 0,
      setMinDurationMinutes: (min) => set({ minDurationMinutes: min }),
      maxDurationMinutes: 0,
      setMaxDurationMinutes: (max) => set({ maxDurationMinutes: max }),
      titleInclude: "",
      setTitleInclude: (v) => set({ titleInclude: v }),
      titleExclude: "",
      setTitleExclude: (v) => set({ titleExclude: v }),
    }),
    persistOptions,
  ),
);

/** Resolved episode path — real audio_path, or virtual path from folder+stem. */
export function useAudioPath(): string | null {
  const episode = useEpisodeStore((s) => s.episode);
  const folder = useEpisodeStore((s) => s.folder);
  if (!episode) return null;
  return episode.audio_path || (folder && episode.stem ? `${folder}/${episode.stem}.mp3` : null);
}
