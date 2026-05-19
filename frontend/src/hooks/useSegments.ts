import { useCallback, useMemo, useReducer, useRef } from "react";
import type { Segment } from "@/api/types";

type Snapshot = {
  original: Segment[];
  edits: Map<number, Partial<Segment>>;
  deleted: Set<number>;
  // Index into the source array each segment came from, parallel to `original`.
  // null marks a segment inserted this session — it has no counterpart in the
  // diff reference, so the reference column must skip a row for it instead of
  // shifting every later row out of alignment.
  sourceIndices: (number | null)[];
};

type EditorState = Snapshot & {
  history: Snapshot[];
};

type EditorAction =
  | { type: "SET_TEXT"; index: number; text: string }
  | { type: "SET_SPEAKER"; index: number; speaker: string }
  | { type: "SET_TIMESTAMP"; index: number; field: "start" | "end"; value: number }
  | { type: "DELETE"; index: number }
  | { type: "DELETE_FLAGGED"; indices: number[] }
  | { type: "RESTORE"; index: number }
  | { type: "INSERT"; afterIndex: number; segment: Segment }
  | { type: "RESET"; segments: Segment[] }
  | { type: "UNDO" }
  | { type: "MERGE"; index: number; speaker?: string }
  | { type: "SPLIT"; index: number; cursorPos: number; explicitTime?: number };

const MAX_HISTORY = 50;

function snap(state: EditorState): Snapshot {
  return {
    original: state.original,
    edits: state.edits,
    deleted: state.deleted,
    sourceIndices: state.sourceIndices,
  };
}

function pushHistory(state: EditorState): Snapshot[] {
  const h = [...state.history, snap(state)];
  return h.length > MAX_HISTORY ? h.slice(-MAX_HISTORY) : h;
}

/** Insert a null at `at` in a parallel source-index array, marking a segment
 *  inserted this session (no diff-reference counterpart). */
function insertNullAt(arr: (number | null)[], at: number): (number | null)[] {
  return [...arr.slice(0, at), null, ...arr.slice(at)];
}

function reducer(state: EditorState, action: EditorAction): EditorState {
  switch (action.type) {
    case "SET_TEXT":
    case "SET_SPEAKER":
    case "SET_TIMESTAMP": {
      const edits = new Map(state.edits);
      const existing = edits.get(action.index) || {};
      if (action.type === "SET_TEXT") {
        edits.set(action.index, { ...existing, text: action.text });
      } else if (action.type === "SET_SPEAKER") {
        edits.set(action.index, { ...existing, speaker: action.speaker });
      } else {
        edits.set(action.index, { ...existing, [action.field]: action.value });
      }
      // Text edits are per-keystroke — don't push history (would flood).
      // Speaker and timestamp changes are discrete commits — push history so undo works.
      if (action.type === "SET_TEXT") {
        return { ...state, edits };
      }
      return { ...state, history: pushHistory(state), edits };
    }
    case "DELETE": {
      const history = pushHistory(state);
      const deleted = new Set(state.deleted);
      deleted.add(action.index);
      return { ...state, history, deleted };
    }
    case "DELETE_FLAGGED": {
      const history = pushHistory(state);
      const deleted = new Set(state.deleted);
      for (const idx of action.indices) deleted.add(idx);
      return { ...state, history, deleted };
    }
    case "RESTORE": {
      if (!state.deleted.has(action.index)) return state;
      const history = pushHistory(state);
      const deleted = new Set(state.deleted);
      deleted.delete(action.index);
      return { ...state, history, deleted };
    }
    case "INSERT": {
      const history = pushHistory(state);
      const insertAt = action.afterIndex + 1;
      const newOriginal = [
        ...state.original.slice(0, insertAt),
        action.segment,
        ...state.original.slice(insertAt),
      ];
      const newEdits = new Map<number, Partial<Segment>>();
      for (const [k, v] of state.edits) {
        newEdits.set(k >= insertAt ? k + 1 : k, v);
      }
      const newDeleted = new Set<number>();
      for (const d of state.deleted) {
        newDeleted.add(d >= insertAt ? d + 1 : d);
      }
      return {
        original: newOriginal,
        edits: newEdits,
        deleted: newDeleted,
        sourceIndices: insertNullAt(state.sourceIndices, insertAt),
        history,
      };
    }
    case "RESET": {
      const history = pushHistory(state);
      return {
        original: action.segments,
        edits: new Map(),
        deleted: new Set<number>(),
        sourceIndices: action.segments.map((_, i) => i),
        history,
      };
    }
    case "UNDO": {
      if (state.history.length === 0) return state;
      const history = [...state.history];
      const prev = history.pop()!;
      return { ...prev, history };
    }
    case "MERGE": {
      // Merge segment at `index` with the next non-deleted segment
      const seg = state.edits.has(action.index)
        ? { ...state.original[action.index], ...state.edits.get(action.index) }
        : state.original[action.index];
      let nextIdx = action.index + 1;
      while (nextIdx < state.original.length && state.deleted.has(nextIdx)) nextIdx++;
      if (nextIdx >= state.original.length) return state;
      const next = state.edits.has(nextIdx)
        ? { ...state.original[nextIdx], ...state.edits.get(nextIdx) }
        : state.original[nextIdx];
      const history = pushHistory(state);
      const edits = new Map(state.edits);
      edits.set(action.index, {
        text: seg.text + " " + next.text,
        end: next.end,
        ...(action.speaker ? { speaker: action.speaker } : {}),
      });
      const deleted = new Set(state.deleted);
      deleted.add(nextIdx);
      return { ...state, history, edits, deleted };
    }
    case "SPLIT": {
      // Split segment at cursor position into two segments
      const history = pushHistory(state);
      const seg = state.edits.has(action.index)
        ? { ...state.original[action.index], ...state.edits.get(action.index) }
        : state.original[action.index];
      const textBefore = seg.text.slice(0, action.cursorPos).trimEnd();
      const textAfter = seg.text.slice(action.cursorPos).trimStart();
      // Explicit timestamp (e.g. current playback position) overrides proportional estimate
      const splitTime = action.explicitTime != null
        ? Math.round(action.explicitTime * 10) / 10
        : (() => {
            const ratio = textBefore.length / Math.max(seg.text.length, 1);
            return Math.round((seg.start + (seg.end - seg.start) * ratio) * 10) / 10;
          })();
      // Update current segment
      const edits = new Map(state.edits);
      edits.set(action.index, { text: textBefore, end: splitTime });
      // Insert new segment after
      const insertAt = action.index + 1;
      const newSeg: Segment = {
        speaker: seg.speaker,
        text: textAfter,
        start: splitTime,
        end: seg.end,
        flagged: false,
      };
      const newOriginal = [
        ...state.original.slice(0, insertAt),
        newSeg,
        ...state.original.slice(insertAt),
      ];
      // Shift edits/deleted indices after insert point
      const newEdits = new Map<number, Partial<Segment>>();
      for (const [k, v] of edits) {
        newEdits.set(k >= insertAt ? k + 1 : k, v);
      }
      // The current segment's edit was at action.index (< insertAt), so it stays
      const newDeleted = new Set<number>();
      for (const d of state.deleted) {
        newDeleted.add(d >= insertAt ? d + 1 : d);
      }
      return {
        original: newOriginal,
        edits: newEdits,
        deleted: newDeleted,
        sourceIndices: insertNullAt(state.sourceIndices, insertAt),
        history,
      };
    }
  }
}

export interface UseSegmentsReturn {
  editedSegments: Segment[];
  /** Maps each editedSegments index to its original index in the source array. */
  originalIndices: number[];
  /** Source-array index per editedSegments entry; null for segments inserted
   *  this session. Used to align the diff reference column so an insert does
   *  not shift every later row. */
  sourceIndices: (number | null)[];
  /** Every segment, including ones pending removal. Save-side still uses editedSegments. */
  allEditedSegments: Segment[];
  /** Original index per entry in allEditedSegments. */
  allOriginalIndices: number[];
  /** Source index per allEditedSegments entry; null for inserted segments. */
  allSourceIndices: (number | null)[];
  /** Direct read access to the pending-delete set — for predicates that key by originalIndex. */
  deletedSet: ReadonlySet<number>;
  isDirty: boolean;
  deletedCount: number;
  canUndo: boolean;
  flaggedIndices: number[];
  updateText: (index: number, text: string) => void;
  updateSpeaker: (index: number, speaker: string) => void;
  updateTimestamp: (index: number, field: "start" | "end", value: number) => void;
  deleteSegment: (index: number) => void;
  deleteFlagged: () => void;
  restoreSegment: (index: number) => void;
  insertAfter: (index: number, segment: Segment) => void;
  mergeWithNext: (index: number, speaker?: string) => void;
  /** Returns the next non-deleted segment's data (for merge dialog). */
  getNextSegment: (index: number) => Segment | null;
  splitAt: (index: number, cursorPos: number, explicitTime?: number) => void;
  reset: (segments: Segment[]) => void;
  undo: () => void;
}

export function useSegments(
  initialSegments: Segment[],
): UseSegmentsReturn {
  const [state, dispatch] = useReducer(reducer, {
    original: initialSegments,
    edits: new Map(),
    deleted: new Set<number>(),
    sourceIndices: initialSegments.map((_, i) => i),
    history: [],
  });

  const {
    editedSegments,
    originalIndices,
    sourceIndices,
    allEditedSegments,
    allOriginalIndices,
    allSourceIndices,
  } = useMemo(() => {
    const segs: Segment[] = [];
    const indices: number[] = [];
    const srcIdx: (number | null)[] = [];
    const allSegs: Segment[] = [];
    const allIndices: number[] = [];
    const allSrcIdx: (number | null)[] = [];
    for (let i = 0; i < state.original.length; i++) {
      const seg = state.original[i];
      const edit = state.edits.get(i);
      const merged = edit ? { ...seg, ...edit } : seg;
      allSegs.push(merged);
      allIndices.push(i);
      allSrcIdx.push(state.sourceIndices[i] ?? null);
      if (state.deleted.has(i)) continue;
      segs.push(merged);
      indices.push(i);
      srcIdx.push(state.sourceIndices[i] ?? null);
    }
    return {
      editedSegments: segs,
      originalIndices: indices,
      sourceIndices: srcIdx,
      allEditedSegments: allSegs,
      allOriginalIndices: allIndices,
      allSourceIndices: allSrcIdx,
    };
  }, [state.original, state.edits, state.deleted, state.sourceIndices]);

  const isDirty = state.edits.size > 0 || state.deleted.size > 0
    || state.original.length !== initialSegments.length;

  const flaggedIndices = useMemo(() => {
    const indices: number[] = [];
    for (let i = 0; i < state.original.length; i++) {
      if (state.deleted.has(i)) continue;
      const seg = state.edits.has(i)
        ? { ...state.original[i], ...state.edits.get(i) }
        : state.original[i];
      if (seg.flagged) indices.push(i);
    }
    return indices;
  }, [state.original, state.edits, state.deleted]);

  // Keep fresh state accessible from stable callbacks (deleteFlagged needs
  // the current flagged list, getNextSegment scans live state).
  const stateRef = useRef(state);
  const flaggedRef = useRef(flaggedIndices);
  // eslint-disable-next-line react-hooks/refs
  stateRef.current = state;
  // eslint-disable-next-line react-hooks/refs
  flaggedRef.current = flaggedIndices;

  const updateText = useCallback((index: number, text: string) => {
    dispatch({ type: "SET_TEXT", index, text });
  }, []);
  const updateSpeaker = useCallback((index: number, speaker: string) => {
    dispatch({ type: "SET_SPEAKER", index, speaker });
  }, []);
  const updateTimestamp = useCallback(
    (index: number, field: "start" | "end", value: number) => {
      dispatch({ type: "SET_TIMESTAMP", index, field, value });
    },
    [],
  );
  const deleteSegment = useCallback((index: number) => {
    dispatch({ type: "DELETE", index });
  }, []);
  const deleteFlagged = useCallback(() => {
    dispatch({ type: "DELETE_FLAGGED", indices: flaggedRef.current });
  }, []);
  const restoreSegment = useCallback((index: number) => {
    dispatch({ type: "RESTORE", index });
  }, []);
  const insertAfter = useCallback((index: number, segment: Segment) => {
    dispatch({ type: "INSERT", afterIndex: index, segment });
  }, []);
  const mergeWithNext = useCallback((index: number, speaker?: string) => {
    dispatch({ type: "MERGE", index, speaker });
  }, []);
  const getNextSegment = useCallback((index: number): Segment | null => {
    const s = stateRef.current;
    let nextIdx = index + 1;
    while (nextIdx < s.original.length && s.deleted.has(nextIdx)) nextIdx++;
    if (nextIdx >= s.original.length) return null;
    const seg = s.original[nextIdx];
    const edit = s.edits.get(nextIdx);
    return edit ? { ...seg, ...edit } : seg;
  }, []);
  const splitAt = useCallback(
    (index: number, cursorPos: number, explicitTime?: number) => {
      dispatch({ type: "SPLIT", index, cursorPos, explicitTime });
    },
    [],
  );
  const reset = useCallback((segments: Segment[]) => {
    dispatch({ type: "RESET", segments });
  }, []);
  const undo = useCallback(() => {
    dispatch({ type: "UNDO" });
  }, []);

  return {
    editedSegments,
    originalIndices,
    sourceIndices,
    allEditedSegments,
    allOriginalIndices,
    allSourceIndices,
    deletedSet: state.deleted,
    isDirty,
    deletedCount: state.deleted.size,
    canUndo: state.history.length > 0,
    flaggedIndices,
    updateText,
    updateSpeaker,
    updateTimestamp,
    deleteSegment,
    deleteFlagged,
    restoreSegment,
    insertAfter,
    mergeWithNext,
    getNextSegment,
    splitAt,
    reset,
    undo,
  };
}
