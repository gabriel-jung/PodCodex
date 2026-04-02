/** Current episode context — set by EpisodePage, read by all pipeline panels. */

import { create } from "zustand";
import type { Episode, ShowMeta } from "@/api/types";

interface EpisodeState {
  episode: Episode | null;
  showMeta: ShowMeta | null;
  folder: string | null;
  setEpisode: (episode: Episode | null, folder?: string | null) => void;
  setShowMeta: (meta: ShowMeta | null) => void;
}

export const useEpisodeStore = create<EpisodeState>((set) => ({
  episode: null,
  showMeta: null,
  folder: null,
  setEpisode: (episode, folder) => set({ episode, folder: folder ?? null }),
  setShowMeta: (meta) => set({ showMeta: meta }),
}));

/** Resolved episode path — real audio_path, or virtual path from folder+stem. */
export function useAudioPath(): string | null {
  const episode = useEpisodeStore((s) => s.episode);
  const folder = useEpisodeStore((s) => s.folder);
  if (!episode) return null;
  return episode.audio_path || (folder && episode.stem ? `${folder}/${episode.stem}.mp3` : null);
}
