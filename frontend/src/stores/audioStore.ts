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
  /**
   * Metadata registered per audio path by pages the user is viewing, WITHOUT
   * loading the track. Lets navigation stay separate from playback: seekTo/
   * playEpisode look meta up here so the bar shows the right title/artwork the
   * first time a registered track is actually played.
   */
  metaByPath: Record<string, AudioMeta>;
  /** Pending seek target in seconds — consumed by AudioBar. */
  pendingSeek: number | null;
  /** Current playback position — updated by AudioBar. */
  currentTime: number;
  /** Whether audio is currently playing — updated by AudioBar. */
  isPlaying: boolean;
  /** Set metadata for the current audio (call once when episode is known). */
  setAudioMeta: (path: string, meta: AudioMeta) => void;
  /**
   * Register metadata for a path the user is viewing, without loading it into
   * the player. Navigation calls this; the bar only appears once the track is
   * explicitly played (seekTo/playEpisode).
   */
  registerMeta: (path: string, meta: AudioMeta) => void;
  /** Provide segments for the current audio so AudioBar can show active text. */
  setAudioSegments: (path: string, segments: AudioSegment[]) => void;
  /** Play/seek — loads the file if needed, seeks to time (0 = start). */
  seekTo: (path: string, time: number) => void;
  /** Set metadata then seek — atomic version of setAudioMeta + seekTo. */
  playEpisode: (path: string, time: number, meta: AudioMeta) => void;
  pauseAudio: () => void;
  /** Resume current track from where it was paused. */
  resumeAudio: () => void;
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
  metaByPath: {},
  pendingSeek: null,
  currentTime: 0,
  isPlaying: false,
  pauseAudio: () => set({ pendingSeek: -1 }),
  resumeAudio: () => set({ pendingSeek: get().currentTime }),
  setAudioMeta: (path, meta) => {
    // Always write. An earlier guard ("only update if state.audioPath ===
    // path or unset") produced null titles when navigating between episodes:
    // the new EpisodePage's setAudioMeta was dropped because the store still
    // held the prior episode's path, and the next seekTo then wiped the
    // title. Trust the caller — pages invoke this for the audio they own.
    set((state) => ({
      audioPath: path,
      audioTitle: meta.title,
      audioArtwork: meta.artwork || null,
      audioShowName: meta.showName || null,
      audioFolder: meta.folder || null,
      audioStem: meta.stem || null,
      metaByPath: { ...state.metaByPath, [path]: meta },
    }));
  },
  registerMeta: (path, meta) =>
    set((state) => ({
      metaByPath: { ...state.metaByPath, [path]: meta },
      // If this is the track currently loaded in the bar, refresh its live meta
      // too (safe: same audioPath, so the <audio> src does not reload). Keeps
      // the bar's title/artwork current when the playing episode's own page
      // resolves its metadata. For any OTHER path, only the map is written, so
      // navigation never disturbs the active track.
      ...(state.audioPath === path
        ? {
            audioTitle: meta.title,
            audioArtwork: meta.artwork || null,
            audioShowName: meta.showName || null,
            audioFolder: meta.folder || null,
            audioStem: meta.stem || null,
          }
        : {}),
    })),
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
      // Switching tracks. Reuse meta registered by navigation/playEpisode/
      // setAudioMeta for this path; if the path was never seen (e.g. a raw
      // seekTo from the index inspector), the bar shows no title until a page
      // for it mounts. (audioTitle is only ever set alongside audioPath, so
      // there is no prior title worth preserving across a track switch.)
      const meta = state.metaByPath[path];
      set({
        audioPath: path,
        audioTitle: meta?.title ?? null,
        audioArtwork: meta?.artwork ?? null,
        audioShowName: meta?.showName ?? null,
        audioFolder: meta?.folder ?? null,
        audioStem: meta?.stem ?? null,
        audioSegments: null,
        pendingSeek: time,
      });
    }
  },
  playEpisode: (path, time, meta) => {
    set((state) => ({
      audioPath: path,
      audioTitle: meta.title,
      audioArtwork: meta.artwork || null,
      audioShowName: meta.showName || null,
      audioFolder: meta.folder || null,
      audioStem: meta.stem || null,
      audioSegments: state.audioPath === path ? state.audioSegments : null,
      pendingSeek: time,
      // Seed the map so a later seekTo to this path (e.g. from a segment row)
      // reuses this meta instead of blanking the bar.
      metaByPath: { ...state.metaByPath, [path]: meta },
    }));
  },
  consumeSeek: () => set({ pendingSeek: null }),
  stopAudio: () =>
    set({ audioPath: null, audioTitle: null, audioArtwork: null, audioShowName: null, audioFolder: null, audioStem: null, audioSegments: null, pendingSeek: null }),
}));
