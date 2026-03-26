/** Global UI state — Zustand store. */

import { create } from "zustand";

interface AppState {
  // Audio player
  audioPath: string | null;
  audioTitle: string | null;
  audioArtwork: string | null;
  audioShowName: string | null;
  /** Pending seek target in seconds — consumed by AudioBar. */
  pendingSeek: number | null;
  /** Current playback position — updated by AudioBar. */
  currentTime: number;
  /** Whether audio is currently playing — updated by AudioBar. */
  isPlaying: boolean;
  playAudio: (path: string, title: string, artwork?: string, showName?: string) => void;
  seekTo: (path: string, time: number, title?: string, artwork?: string, showName?: string) => void;
  pauseAudio: () => void;
  consumeSeek: () => void;
  stopAudio: () => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  audioPath: null,
  audioTitle: null,
  audioArtwork: null,
  audioShowName: null,
  pendingSeek: null,
  currentTime: 0,
  isPlaying: false,
  /** pauseAudio sets pendingSeek to -1, which AudioBar interprets as "pause". */
  pauseAudio: () => set({ pendingSeek: -1 }),
  playAudio: (path, title, artwork, showName) =>
    set({ audioPath: path, audioTitle: title, audioArtwork: artwork || null, audioShowName: showName || null, pendingSeek: null }),
  seekTo: (path, time, title, artwork, showName) => {
    const state = get();
    if (state.audioPath === path) {
      // Same file — just seek
      set({ pendingSeek: time });
    } else {
      // Different file — load it first, then seek
      set({
        audioPath: path,
        audioTitle: title || path.split("/").pop()?.replace(/\.[^.]+$/, "") || "Audio",
        audioArtwork: artwork || null,
        audioShowName: showName || null,
        pendingSeek: time,
      });
    }
  },
  consumeSeek: () => set({ pendingSeek: null }),
  stopAudio: () =>
    set({ audioPath: null, audioTitle: null, audioArtwork: null, audioShowName: null, pendingSeek: null }),
}));
