/** Audio playback state — controls the global AudioBar. */

import { create } from "zustand";

interface AudioMeta {
  title: string;
  artwork?: string;
  showName?: string;
  /** Show folder — used to build episode link. */
  folder?: string;
  /** Episode stem — used to build episode link. */
  stem?: string;
}

export interface AudioSegment {
  start: number;
  end: number;
  speaker: string;
  text: string;
}

interface AudioState {
  audioPath: string | null;
  audioTitle: string | null;
  audioArtwork: string | null;
  audioShowName: string | null;
  /** Show folder for linking back to episode page. */
  audioFolder: string | null;
  /** Episode stem for linking back to episode page. */
  audioStem: string | null;
  /** Segments for the current audio — set by SegmentEditor, read by AudioBar. */
  audioSegments: AudioSegment[] | null;
  /** Pending seek target in seconds — consumed by AudioBar. */
  pendingSeek: number | null;
  /** Current playback position — updated by AudioBar. */
  currentTime: number;
  /** Whether audio is currently playing — updated by AudioBar. */
  isPlaying: boolean;
  /** Set metadata for the current audio (call once when episode is known). */
  setAudioMeta: (path: string, meta: AudioMeta) => void;
  /** Provide segments for the current audio so AudioBar can show active text. */
  setAudioSegments: (path: string, segments: AudioSegment[]) => void;
  /** Play/seek — loads the file if needed, seeks to time (0 = start). */
  seekTo: (path: string, time: number) => void;
  /** Set metadata then seek — atomic version of setAudioMeta + seekTo. */
  playEpisode: (path: string, time: number, meta: AudioMeta) => void;
  pauseAudio: () => void;
  consumeSeek: () => void;
  stopAudio: () => void;
}

export const useAudioStore = create<AudioState>((set, get) => ({
  audioPath: null,
  audioTitle: null,
  audioArtwork: null,
  audioShowName: null,
  audioFolder: null,
  audioStem: null,
  audioSegments: null,
  pendingSeek: null,
  currentTime: 0,
  isPlaying: false,
  pauseAudio: () => set({ pendingSeek: -1 }),
  setAudioMeta: (path, meta) => {
    const state = get();
    if (state.audioPath === path || !state.audioPath) {
      set({
        audioPath: path,
        audioTitle: meta.title,
        audioArtwork: meta.artwork || null,
        audioShowName: meta.showName || null,
        audioFolder: meta.folder || null,
        audioStem: meta.stem || null,
      });
    }
  },
  setAudioSegments: (path, segments) => {
    if (get().audioPath === path) {
      set({ audioSegments: segments });
    }
  },
  seekTo: (path, time) => {
    const state = get();
    if (state.audioPath === path) {
      set({ pendingSeek: time });
    } else {
      set({
        audioPath: path,
        audioTitle: state.audioPath ? null : state.audioTitle,
        audioSegments: null,
        pendingSeek: time,
      });
    }
  },
  playEpisode: (path, time, meta) => {
    set({
      audioPath: path,
      audioTitle: meta.title,
      audioArtwork: meta.artwork || null,
      audioShowName: meta.showName || null,
      audioFolder: meta.folder || null,
      audioStem: meta.stem || null,
      audioSegments: get().audioPath === path ? get().audioSegments : null,
      pendingSeek: time,
    });
  },
  consumeSeek: () => set({ pendingSeek: null }),
  stopAudio: () =>
    set({ audioPath: null, audioTitle: null, audioArtwork: null, audioShowName: null, audioFolder: null, audioStem: null, audioSegments: null, pendingSeek: null }),
}));
