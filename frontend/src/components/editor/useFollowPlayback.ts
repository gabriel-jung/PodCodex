/**
 * Which row the player is on, and whether the list follows it.
 *
 * Extracted from `TranscriptViewer`, where sixty-five hook calls shared one
 * closure. This half depends only on the audio store, the list handle and
 * the editor's current rows; the cross-page jump helpers stay in the
 * component, since those are pagination concerns rather than playback ones.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { Segment } from "@/api/types";
import { useAudioStore } from "@/stores";

/** The imperative handle `SegmentList` exposes. */
interface ScrollableList {
  scrollToId: (id: number, behavior?: ScrollBehavior) => boolean;
}

/** The slice of `useSegments` this hook reads. */
interface SegmentRows {
  ids: number[];
  editedSegments: Segment[];
}

export interface FollowPlayback {
  /** Row currently under the playhead, or null. */
  activeId: number | null;
  /** True while the store is playing *this* episode's audio. */
  isPlayingThisFile: boolean;
  /** Whether the player is playing at all; drives the toolbar's indicator. */
  storeIsPlaying: boolean;
  followMode: boolean;
  /** Scroll the list to a row. Safe before the list mounts. */
  scrollToId: (id: number, behavior?: ScrollBehavior) => boolean;
  /** Re-engage follow and centre on the active row (explicit user gesture). */
  jumpToActive: () => void;
  /** Any wheel/touch scroll: disengages follow so the reader can browse. */
  handleUserScroll: () => void;
}

export function useFollowPlayback(
  audioPath: string | undefined,
  listRef: React.RefObject<ScrollableList | null>,
  editor: SegmentRows,
): FollowPlayback {
  const storeAudioPath = useAudioStore((s) => s.audioPath);
  const storeIsPlaying = useAudioStore((s) => s.isPlaying);
  const isPlayingThisFile = audioPath != null && storeAudioPath === audioPath;
  const [activeId, setActiveId] = useState<number | null>(null);

  const editedSegmentsRef = useRef(editor.editedSegments);
  // eslint-disable-next-line react-hooks/refs
  editedSegmentsRef.current = editor.editedSegments;
  const idsRef = useRef(editor.ids);
  // eslint-disable-next-line react-hooks/refs
  idsRef.current = editor.ids;

  // Drop activeId only when the player jumps to a different file. While the
  // current track is paused we keep the last activeId so the Now-playing
  // toolbar button still has somewhere to re-center to.
  useEffect(() => {
    if (audioPath == null || storeAudioPath !== audioPath) {
      setActiveId(null);
    }
  }, [audioPath, storeAudioPath]);

  useEffect(() => {
    if (!isPlayingThisFile) return;
    const interval = setInterval(() => {
      const t = useAudioStore.getState().currentTime;
      if (!useAudioStore.getState().isPlaying) return;
      const segs = editedSegmentsRef.current;
      const ids = idsRef.current;
      for (let e = segs.length - 1; e >= 0; e--) {
        const seg = segs[e];
        if (seg.start <= t && t < seg.end) {
          const id = ids[e];
          setActiveId((prev) => (prev === id ? prev : id));
          return;
        }
      }
      setActiveId((prev) => (prev == null ? prev : null));
    }, 250);
    return () => clearInterval(interval);
  }, [isPlayingThisFile]);

  const scrollToId = useCallback(
    (id: number, behavior: ScrollBehavior = "smooth") => {
      return listRef.current?.scrollToId(id, behavior) ?? false;
    },
    [listRef],
  );

  // Auto-follow: while ON, the list scrolls to the active segment as
  // playback advances. Any user-initiated scroll (wheel/touchmove) flips it
  // OFF so the reader can browse without being yanked back. "Now playing"
  // toolbar button flips it back ON and re-centers.
  const [followMode, setFollowMode] = useState(true);
  // Re-engage follow on track change so the new transcript opens aligned.
  // In an effect, not during render: the component this was lifted out of
  // did it inline, which the refs rule forbids in a hook body.
  useEffect(() => {
    setFollowMode(true);
  }, [audioPath]);
  // Set by jumpToActive so the follow effect's first run after the toggle
  // doesn't re-fire scrollToId on top of the explicit call.
  const suppressNextFollowScrollRef = useRef(false);
  useEffect(() => {
    if (!followMode || activeId == null) return;
    if (suppressNextFollowScrollRef.current) {
      suppressNextFollowScrollRef.current = false;
      return;
    }
    // Skip when the user is typing inside the list: an active textarea/input
    // means they're editing this row (or a nearby one) and the smooth-scroll
    // would yank the caret offscreen mid-keystroke.
    const focusTag = (document.activeElement as HTMLElement | null)?.tagName;
    const focusEditable =
      focusTag === "TEXTAREA" ||
      focusTag === "INPUT" ||
      (document.activeElement as HTMLElement | null)?.isContentEditable;
    if (focusEditable) return;
    // 'auto' avoids stacked smooth-scroll animations on the 250ms activeId
    // polling tick. 'smooth' is reserved for the explicit jumpToActive click.
    scrollToId(activeId, "auto");
  }, [followMode, activeId, scrollToId]);

  const handleUserScroll = useCallback(() => {
    setFollowMode((cur) => (cur ? false : cur));
  }, []);

  const jumpToActive = useCallback(() => {
    if (!followMode) {
      // Effect would scroll on its own once followMode commits — but we want
      // 'smooth' here (explicit user gesture), so do it manually and ask the
      // effect to skip its first post-toggle run.
      suppressNextFollowScrollRef.current = true;
      setFollowMode(true);
    }
    if (activeId != null) scrollToId(activeId, "smooth");
  }, [followMode, activeId, scrollToId]);

  return {
    activeId,
    isPlayingThisFile,
    storeIsPlaying,
    followMode,
    scrollToId,
    jumpToActive,
    handleUserScroll,
  };
}
