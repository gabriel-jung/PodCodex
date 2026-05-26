import { useEffect } from "react";
import { useAudioStore } from "@/stores";

/** Global keyboard shortcuts registered once at the app root. */
export function useGlobalShortcuts() {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      const inEditable =
        tag === "INPUT" || tag === "TEXTAREA" || target?.isContentEditable;

      // Escape inside an editable — pause audio if playing and blur the
      // field. Lets the user free their hands from a long textarea without
      // having to mouse over to the audio bar. Some keyboard layouts (e.g.
      // French AZERTY on Firefox) strip the Shift modifier on Space, making
      // Shift+Space unreliable; Escape is the always-available fallback.
      if (e.key === "Escape" && inEditable && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const { audioPath, isPlaying, pauseAudio } = useAudioStore.getState();
        if (audioPath && isPlaying) {
          e.preventDefault();
          pauseAudio();
        }
        target?.blur();
        return;
      }

      // Shift+Space OR Ctrl+Space — toggle play/pause regardless of focus.
      // Plain Space still types in inputs/textareas via the inEditable
      // bail-out below. Two combos because some keyboard layouts (French
      // AZERTY on Firefox) consume Shift+Space natively for non-breaking
      // space, stripping the modifier before keydown — Ctrl+Space is the
      // reliable fallback. macOS users keep Cmd free for Spotlight.
      const shiftSpace =
        e.key === " " && e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey;
      const ctrlSpace =
        e.key === " " && e.ctrlKey && !e.shiftKey && !e.metaKey && !e.altKey;
      if (shiftSpace || ctrlSpace) {
        const { audioPath, isPlaying, pauseAudio, currentTime } = useAudioStore.getState();
        if (inEditable) e.preventDefault();
        if (!audioPath) return;
        if (!inEditable) e.preventDefault();
        if (isPlaying) pauseAudio();
        else useAudioStore.setState({ pendingSeek: currentTime || 0 });
        return;
      }

      if (inEditable) return;

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
