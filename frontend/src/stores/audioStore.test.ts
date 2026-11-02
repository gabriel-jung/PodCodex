import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAudioStore } from "./audioStore";

/** The toggle the episode rows and the episode header share. A wrong branch
 *  here restarts or silently stalls playback from every play button. */

describe("toggleTrack", () => {
  beforeEach(() => {
    useAudioStore.setState({ audioPath: "/a.mp3", isPlaying: false, ended: false, currentTime: 42, pendingSeek: null });
  });

  it("starts another track through the caller", () => {
    const start = vi.fn();
    useAudioStore.getState().toggleTrack("/b.mp3", start);
    expect(start).toHaveBeenCalledOnce();
  });

  it("pauses the loaded track while it plays", () => {
    useAudioStore.setState({ isPlaying: true });
    useAudioStore.getState().toggleTrack("/a.mp3", () => {});
    expect(useAudioStore.getState().pendingSeek).toBe(-1);
  });

  it("resumes a paused track where it stopped", () => {
    useAudioStore.getState().toggleTrack("/a.mp3", () => {});
    expect(useAudioStore.getState().pendingSeek).toBe(42);
  });

  it("restarts a track that played to its end", () => {
    useAudioStore.setState({ ended: true });
    useAudioStore.getState().toggleTrack("/a.mp3", () => {});
    expect(useAudioStore.getState().pendingSeek).toBe(0);
  });
});
