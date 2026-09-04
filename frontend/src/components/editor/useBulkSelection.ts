/**
 * Multi-row selection and the bulk actions over it.
 *
 * Extracted from `TranscriptViewer`, where sixty-five hook calls shared one
 * closure and each new editor feature had to reason about every other
 * feature's refs and effects. Nothing here touches playback, versions or
 * filtering: it needs the editor and a way to report what is selected.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Segment } from "@/api/types";
import { BREAK_SPEAKER } from "@/lib/speakers";

/** The slice of `useSegments` this hook drives. */
interface SegmentEditor {
  ids: number[];
  editedSegments: Segment[];
  deleteSegment: (id: number) => void;
  updateSpeaker: (id: number, speaker: string) => void;
  mergeWithNext: (id: number) => void;
}

export interface BulkSelection {
  selectedIds: Set<number>;
  /** Raw setter, for the toolbar's select-all/none and the reset on a
   *  version switch (row ids are reassigned there, so every id-keyed piece
   *  of UI state has to be dropped at once). */
  setSelectedIds: React.Dispatch<React.SetStateAction<Set<number>>>;
  toggleSelect: (id: number) => void;
  clearSelection: () => void;
  /** Row id → its segment and current position, for adjacency and lookup. */
  segmentById: Map<number, { segment: Segment; position: number }>;
  bulkDelete: () => void;
  bulkSpeaker: (speaker: string) => void;
  bulkMerge: () => void;
}

export function useBulkSelection(
  editor: SegmentEditor,
  {
    onSelectionChange,
    onBulkEdited,
  }: {
    /** Called with the selected segments (never the raw ids) whenever the
     *  selection's content changes. */
    onSelectionChange?: (segments: Segment[]) => void;
    /** Row ids a bulk action just rewrote, for the recently-edited highlight. */
    onBulkEdited?: (ids: Set<number>) => void;
  },
): BulkSelection {
  // Keyed by stable row id. Insert/split do not invalidate these entries.
  const [selectedIds, setSelectedIds] = useState<Set<number>>(() => new Set());

  const toggleSelect = useCallback((id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const clearSelection = useCallback(() => setSelectedIds(new Set()), []);

  // Per-id segment lookup. Built from the current edited list (non-deleted) so
  // bulkMerge can find positional adjacency and the selection-emit effect can
  // dereference each selected id.
  const editedSegments = editor.editedSegments;
  const editorIds = editor.ids;
  const segmentById = useMemo(() => {
    const map = new Map<number, { segment: Segment; position: number }>();
    for (let i = 0; i < editorIds.length; i++) {
      map.set(editorIds[i], { segment: editedSegments[i], position: i });
    }
    return map;
  }, [editedSegments, editorIds]);

  // Emit selection changes — downstream consumers (e.g. synthesis scope
  // filter) work on the edited segment payloads, not raw indices, so their
  // state survives version switches and pagination. Signature guard keeps
  // per-keystroke edits from re-emitting when the selection hasn't changed.
  const lastSelectionSigRef = useRef<string>("");
  useEffect(() => {
    if (!onSelectionChange) return;
    const out: Segment[] = [];
    const sigParts: string[] = [];
    for (const id of selectedIds) {
      const hit = segmentById.get(id);
      if (!hit || hit.segment.speaker === BREAK_SPEAKER) continue;
      out.push(hit.segment);
      sigParts.push(`${hit.segment.speaker}:${hit.segment.start}:${hit.segment.end}`);
    }
    const sig = sigParts.join("|");
    if (sig === lastSelectionSigRef.current) return;
    lastSelectionSigRef.current = sig;
    onSelectionChange(out);
  }, [selectedIds, segmentById, onSelectionChange]);

  const bulkDelete = useCallback(() => {
    // Order doesn't matter — ids stay valid across deletes.
    for (const id of selectedIds) editor.deleteSegment(id);
    clearSelection();
  }, [selectedIds, editor, clearSelection]);

  const bulkSpeaker = useCallback(
    (speaker: string) => {
      for (const id of selectedIds) editor.updateSpeaker(id, speaker);
      onBulkEdited?.(new Set(selectedIds));
      clearSelection();
    },
    [selectedIds, editor, clearSelection, onBulkEdited],
  );

  const bulkMerge = useCallback(() => {
    // Adjacency by current position, not by id (ids are not contiguous after
    // inserts). Sort selected ids by their position, then merge from the
    // bottom up so the merge index doesn't shift the rest.
    const positions = Array.from(selectedIds)
      .map((id) => ({ id, position: segmentById.get(id)?.position ?? -1 }))
      .filter((p) => p.position >= 0)
      .sort((a, b) => a.position - b.position);
    if (positions.length < 2) return;
    for (let i = positions.length - 1; i > 0; i--) {
      if (positions[i].position === positions[i - 1].position + 1) {
        editor.mergeWithNext(positions[i - 1].id);
      }
    }
    clearSelection();
  }, [selectedIds, segmentById, editor, clearSelection]);

  return {
    selectedIds,
    setSelectedIds,
    toggleSelect,
    clearSelection,
    segmentById,
    bulkDelete,
    bulkSpeaker,
    bulkMerge,
  };
}
