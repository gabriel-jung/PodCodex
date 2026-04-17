import { useEffect } from "react";
import { useAudioStore } from "@/stores";

/** Global keyboard shortcuts registered once at the app root. */
export function useGlobalShortcuts() {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Don't intercept when typing in inputs
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement)?.isContentEditable) return;

      // Space — toggle play/pause
      if (e.key === " " && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const { audioPath, isPlaying, pauseAudio, currentTime } = useAudioStore.getState();
        if (!audioPath) return;
        e.preventDefault();
        if (isPlaying) {
          pauseAudio();
        } else {
          useAudioStore.setState({ pendingSeek: currentTime || 0 });
        }
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);
}
