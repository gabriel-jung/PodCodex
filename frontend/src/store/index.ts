/** Global UI state — Zustand store. */

import { create } from "zustand";

interface AppState {
  // Audio player
  audioPath: string | null;
  audioTitle: string | null;
  audioArtwork: string | null;
  audioShowName: string | null;
  playAudio: (path: string, title: string, artwork?: string, showName?: string) => void;
  stopAudio: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  audioPath: null,
  audioTitle: null,
  audioArtwork: null,
  audioShowName: null,
  playAudio: (path, title, artwork, showName) =>
    set({ audioPath: path, audioTitle: title, audioArtwork: artwork || null, audioShowName: showName || null }),
  stopAudio: () =>
    set({ audioPath: null, audioTitle: null, audioArtwork: null, audioShowName: null }),
}));
