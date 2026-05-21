/**
 * SegmentList — virtualized list of segment rows.
 *
 * Owns the scroll container and the virtualizer; exposes `scrollToId` via a
 * ref so the parent can drive jumps without holding the virtualizer itself.
 * Row rendering is delegated to SegmentRow.
 */

import { forwardRef, useImperativeHandle, useRef } from "react";
import type { Segment } from "@/api/types";
import { useVirtualizer } from "@tanstack/react-virtual";
import SegmentRow from "./SegmentRow";
import { flagReason, type FilteredResult } from "@/hooks/useSegmentFiltering";
import { BREAK_SPEAKER } from "@/lib/speakers";

export type SegmentListItem = FilteredResult["pageSegments"][number];

export interface SegmentListHandle {
  /** Scroll the row matching `id` into the centre of the viewport. Returns
   *  true on success, false if the row isn't on the current page. */
  scrollToId: (id: number, behavior?: ScrollBehavior) => boolean;
}

export interface SegmentListProps {
  editorKey: string;
  pageSegments: SegmentListItem[];
  // Derived row data
  speakers: string[];
  audioPath?: string;
  showFlags: boolean;
  showSpeaker: boolean;
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
}

const SegmentList = forwardRef<SegmentListHandle, SegmentListProps>(function SegmentList(
  {
    editorKey,
    pageSegments,
    speakers,
    audioPath,
    showFlags,
    showSpeaker,
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
  },
  handleRef,
) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Virtualizer: only rows visible in the scroll viewport are rendered. Row
  // heights are measured dynamically; the estimate covers first paint.
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
          const renamedTo = pendingRenames[segment.speaker];
          const displaySegment =
            renamedTo && renamedTo !== segment.speaker
              ? { ...segment, speaker: renamedTo }
              : segment;
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
                selected={selectedIds.has(id)}
                onToggleSelect={onToggleSelect}
                audioPath={audioPath}
                speakers={speakers}
                showSpeaker={showSpeaker}
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
