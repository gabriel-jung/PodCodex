/**
 * SegmentList — virtualized list of segment rows.
 *
 * Owns the scroll container and the virtualizer; exposes `scrollToId` via a
 * ref so the parent can drive jumps without holding the virtualizer itself.
 * Row rendering is delegated to SegmentRow.
 */

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import type { Segment } from "@/api/types";
import { useVirtualizer } from "@tanstack/react-virtual";
import SegmentRow from "./SegmentRow";
import { flagReason, type FilteredResult } from "@/hooks/useSegmentFiltering";
import { BREAK_SPEAKER } from "@/lib/speakers";

const GAP_SUBTLE_S = 10;
const GAP_PROMINENT_S = 60;

function formatGap(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return s === 0 ? `${m}m` : `${m}m ${s}s`;
}

// Precomputed bar heights for the prominent gap envelope. Static — avoids
// re-allocating 24 styled divs per virtualized row on every render.
const ENVELOPE_BARS = Array.from({ length: 24 }, (_, i) =>
  Math.max(2, 14 - Math.abs(i - 11.5) * 1.1),
);

function EnvelopeStrip({ opacity }: { opacity: number }) {
  return (
    <div className="flex-1 flex items-center gap-px h-6">
      {ENVELOPE_BARS.map((h, i) => (
        <div
          key={i}
          className="flex-1 bg-warning/40"
          style={{ height: `${h}px`, opacity }}
        />
      ))}
    </div>
  );
}

function GapDivider({ gap }: { gap: number }) {
  if (gap >= GAP_PROMINENT_S) {
    const opacity = 0.25 + (Math.min(gap, 600) / 600) * 0.55;
    return (
      <div className="py-1.5 px-2 flex items-center gap-2 select-none">
        <div className="w-12 shrink-0 flex justify-end">
          <span className="text-[10px] text-muted-foreground/50 tabular-nums">
            {Math.floor(gap / 60)}:{String(Math.floor(gap % 60)).padStart(2, "0")}
          </span>
        </div>
        <EnvelopeStrip opacity={opacity} />
        <span className="text-[11px] text-muted-foreground/80 tabular-nums tracking-tight px-1.5 py-0.5 rounded-sm bg-secondary/40 border border-border/60">
          {formatGap(gap)} pause
        </span>
        <EnvelopeStrip opacity={opacity} />
      </div>
    );
  }
  const opacity = Math.min(0.25 + gap / 120, 0.7);
  return (
    <div className="h-3 flex items-center gap-2 px-2 text-muted-foreground/40 select-none">
      <div className="w-12 shrink-0" />
      <div className="flex-1 border-t border-dashed border-border/60" style={{ opacity }} />
      <span className="text-[10px] tabular-nums tracking-tight">{gap.toFixed(0)}s</span>
      <div className="flex-1 border-t border-dashed border-border/60" style={{ opacity }} />
    </div>
  );
}

export type SegmentListItem = FilteredResult["pageSegments"][number];

export interface SegmentListHandle {
  /** Scroll the row matching `id` into the centre of the viewport. Returns
   *  true on success, false if the row isn't on the current page. */
  scrollToId: (id: number, behavior?: ScrollBehavior) => boolean;
}

export interface SegmentListProps {
  editorKey: string;
  pageSegments: SegmentListItem[];
  /** True when any filter (search/speaker/flagged/changed/removed) is active.
   *  Gap dividers are suppressed under filtering: filtered pageSegments can
   *  skip arbitrary stretches of source segments, so "Xm Ys pause" would
   *  mislabel the gap between two non-adjacent rows as a real silence. */
  filterActive?: boolean;
  // Derived row data
  speakers: string[];
  audioPath?: string;
  showFlags: boolean;
  showSpeaker: boolean;
  /** Fade the speaker label at rest (single default speaker). */
  speakerMuted?: boolean;
  showDelete: boolean;
  showDiff: boolean;
  // Filter context for flag reason
  densityThreshold: number;
  maxDensityThreshold: number;
  customPatterns: string[];
  // Per-row state
  activeId: number | null;
  storeIsPlaying: boolean;
  pendingRenames: Record<string, string>;
  pendingRemovals: Set<string>;
  dismissedFlags: Set<number>;
  deletedSet: ReadonlySet<number>;
  textEditedIds: ReadonlySet<number>;
  selectedIds: Set<number>;
  // Derivations supplied by parent (keep cross-row caches consistent)
  getRef: (id: number) => Segment | undefined;
  isChanged: (segment: Segment, id: number) => boolean;
  // Row callbacks (must be reference-stable from parent)
  onToggleSelect: (id: number) => void;
  onTextChange: (id: number, text: string) => void;
  onSpeakerChange: (id: number, speaker: string) => void;
  onTimestampChange: (id: number, field: "start" | "end", value: number) => void;
  onDelete: (id: number) => void;
  onRestore: (id: number) => void;
  onDismissFlag: (id: number) => void;
  onInsertBefore: (id: number, segment: Segment) => void;
  onInsertAfter: (id: number, segment: Segment) => void;
  onMergeNext: (id: number, speaker: string) => void;
  onSplit: (id: number, cursorPos: number, explicitTime?: number) => void;
  /** Fired on any genuine user-initiated scroll (wheel/touch). Programmatic
   *  scrolls from `scrollToId` do NOT fire this. Used by the parent to drop
   *  auto-follow mode when the reader takes manual control. */
  onUserScroll?: () => void;
}

const SegmentList = forwardRef<SegmentListHandle, SegmentListProps>(function SegmentList(
  {
    editorKey,
    pageSegments,
    filterActive,
    speakers,
    audioPath,
    showFlags,
    showSpeaker,
    speakerMuted,
    showDelete,
    showDiff,
    densityThreshold,
    maxDensityThreshold,
    customPatterns,
    activeId,
    storeIsPlaying,
    pendingRenames,
    pendingRemovals,
    dismissedFlags,
    deletedSet,
    textEditedIds,
    selectedIds,
    getRef,
    isChanged,
    onToggleSelect,
    onTextChange,
    onSpeakerChange,
    onTimestampChange,
    onDelete,
    onRestore,
    onDismissFlag,
    onInsertBefore,
    onInsertAfter,
    onMergeNext,
    onSplit,
    onUserScroll,
  },
  handleRef,
) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !onUserScroll) return;
    const handler = () => onUserScroll();
    // wheel + touchstart are user-only; the virtualizer's scrollToIndex
    // triggers `scroll` but not these.
    el.addEventListener("wheel", handler, { passive: true });
    // touchmove (not touchstart) — touchstart fires on every tap, including
    // taps on play buttons / speaker chips, which would falsely disengage
    // follow before the user has actually scrolled.
    el.addEventListener("touchmove", handler, { passive: true });
    return () => {
      el.removeEventListener("wheel", handler);
      el.removeEventListener("touchmove", handler);
    };
  }, [onUserScroll]);

  // Virtualizer: only rows visible in the scroll viewport are rendered. Row
  // heights are measured dynamically; the estimate covers first paint.
  // eslint-disable-next-line react-hooks/incompatible-library -- virtualizer is ref-based; compiler skip is expected
  const rowVirtualizer = useVirtualizer({
    count: pageSegments.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 72,
    overscan: 8,
    // Stable per-row id — survives insert/split so the virtualizer keeps its
    // height cache for rows below an insert.
    getItemKey: (i) => `${editorKey}-${pageSegments[i]?.id ?? i}`,
  });

  useImperativeHandle(handleRef, () => ({
    scrollToId: (id, behavior = "smooth") => {
      const idx = pageSegments.findIndex((p) => p.id === id);
      if (idx >= 0) rowVirtualizer.scrollToIndex(idx, { align: "center", behavior });
      return idx >= 0;
    },
  }), [pageSegments, rowVirtualizer]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto py-2">
      <div
        style={{ height: rowVirtualizer.getTotalSize(), position: "relative", width: "100%" }}
      >
        {rowVirtualizer.getVirtualItems().map((v) => {
          const item = pageSegments[v.index];
          if (!item) return null;
          const { segment, id, displayIndex } = item;
          const ref = getRef(id);
          const isBreak = segment.speaker === BREAK_SPEAKER;
          const reason = showFlags
            ? flagReason(segment, densityThreshold, maxDensityThreshold, customPatterns)
            : null;
          const flagged = reason !== null && !dismissedFlags.has(id);
          // Chip-rename lookup keyed on the *merged* speaker. By design, a
          // per-row SET_SPEAKER edit is an explicit override: it freezes the
          // row's speaker name and later chip renames won't reassign it.
          // Rows that never had a per-row edit follow the chip via this
          // pendingRenames map. Split-time uses the same resolution path
          // (see TranscriptViewer.handleRowSplit) so both halves agree.
          const renamedTo = pendingRenames[segment.speaker];
          const displaySegment =
            renamedTo && renamedTo !== segment.speaker
              ? { ...segment, speaker: renamedTo }
              : segment;
          // Gap divider against the immediately previous row in the current
          // page. Skip when either side is a BREAK marker (those already
          // encode their own pause visually) and when any filter is active
          // (the previous visible row may not be chronologically adjacent).
          const prevItem = v.index > 0 ? pageSegments[v.index - 1] : null;
          const prevIsBreak = prevItem?.segment.speaker === BREAK_SPEAKER;
          const rawGap = prevItem && !isBreak && !prevIsBreak && !filterActive
            ? segment.start - prevItem.segment.end
            : 0;
          const gap = Math.max(0, rawGap);
          const showGap = gap >= GAP_SUBTLE_S;
          return (
            <div
              key={`${editorKey}-${id}`}
              data-row-id={id}
              data-index={v.index}
              ref={rowVirtualizer.measureElement}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                transform: `translateY(${v.start}px)`,
              }}
            >
              {showGap && <GapDivider gap={gap} />}
              <SegmentRow
                segment={displaySegment}
                id={id}
                displayNumber={displayIndex + 1}
                isActive={activeId === id}
                isPlayingActive={activeId === id && storeIsPlaying}
                isFlagged={flagged}
                flagReasonText={flagged ? reason : null}
                isChanged={isChanged(segment, id)}
                isPendingRemoval={pendingRemovals.has(segment.speaker) || deletedSet.has(id)}
                isDeleted={deletedSet.has(id)}
                isTextEdited={textEditedIds.has(id)}
                selected={selectedIds.has(id)}
                onToggleSelect={onToggleSelect}
                audioPath={audioPath}
                speakers={speakers}
                showSpeaker={showSpeaker}
                speakerMuted={speakerMuted}
                showDelete={showDelete}
                onTextChange={onTextChange}
                onSpeakerChange={onSpeakerChange}
                onTimestampChange={onTimestampChange}
                onDelete={onDelete}
                onRestore={onRestore}
                onDismissFlag={showFlags ? onDismissFlag : undefined}
                onInsertBefore={onInsertBefore}
                onInsertAfter={onInsertAfter}
                onMergeNext={onMergeNext}
                onSplit={onSplit}
                referenceText={ref && !isBreak ? ref.text : undefined}
                showDiff={showDiff}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
});

export default SegmentList;
