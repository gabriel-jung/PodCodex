import { useMemo, useReducer } from "react";
import type { Segment } from "@/api/types";

type Snapshot = {
  original: Segment[];
  edits: Map<number, Partial<Segment>>;
  deleted: Set<number>;
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
  | { type: "INSERT"; afterIndex: number; segment: Segment }
  | { type: "RESET"; segments: Segment[] }
  | { type: "UNDO" }
  | { type: "MERGE"; index: number; speaker?: string }
  | { type: "SPLIT"; index: number; cursorPos: number; explicitTime?: number };

const MAX_HISTORY = 50;

function snap(state: EditorState): Snapshot {
  return { original: state.original, edits: state.edits, deleted: state.deleted };
}

function pushHistory(state: EditorState): Snapshot[] {
  const h = [...state.history, snap(state)];
  return h.length > MAX_HISTORY ? h.slice(-MAX_HISTORY) : h;
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
      return { original: newOriginal, edits: newEdits, deleted: newDeleted, history };
    }
    case "RESET": {
      const history = pushHistory(state);
      return {
        original: action.segments,
        edits: new Map(),
        deleted: new Set<number>(),
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
      return { original: newOriginal, edits: newEdits, deleted: newDeleted, history };
    }
  }
}

export interface UseSegmentsReturn {
  editedSegments: Segment[];
  /** Maps each editedSegments index to its original index in the source array. */
  originalIndices: number[];
  isDirty: boolean;
  deletedCount: number;
  canUndo: boolean;
  flaggedIndices: number[];
  updateText: (index: number, text: string) => void;
  updateSpeaker: (index: number, speaker: string) => void;
  updateTimestamp: (index: number, field: "start" | "end", value: number) => void;
  deleteSegment: (index: number) => void;
  deleteFlagged: () => void;
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
    history: [],
  });

  const { editedSegments, originalIndices } = useMemo(() => {
    const segs: Segment[] = [];
    const indices: number[] = [];
    for (let i = 0; i < state.original.length; i++) {
      if (state.deleted.has(i)) continue;
      const seg = state.original[i];
      const edit = state.edits.get(i);
      segs.push(edit ? { ...seg, ...edit } : seg);
      indices.push(i);
    }
    return { editedSegments: segs, originalIndices: indices };
  }, [state.original, state.edits, state.deleted]);

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

  return {
    editedSegments,
    originalIndices,
    isDirty,
    deletedCount: state.deleted.size,
    canUndo: state.history.length > 0,
    flaggedIndices,
    updateText: (index, text) => dispatch({ type: "SET_TEXT", index, text }),
    updateSpeaker: (index, speaker) => dispatch({ type: "SET_SPEAKER", index, speaker }),
    updateTimestamp: (index, field, value) =>
      dispatch({ type: "SET_TIMESTAMP", index, field, value }),
    deleteSegment: (index) => dispatch({ type: "DELETE", index }),
    deleteFlagged: () => dispatch({ type: "DELETE_FLAGGED", indices: flaggedIndices }),
    insertAfter: (index, segment) => dispatch({ type: "INSERT", afterIndex: index, segment }),
    mergeWithNext: (index, speaker?) => dispatch({ type: "MERGE", index, speaker }),
    getNextSegment: (index) => {
      let nextIdx = index + 1;
      while (nextIdx < state.original.length && state.deleted.has(nextIdx)) nextIdx++;
      if (nextIdx >= state.original.length) return null;
      const seg = state.original[nextIdx];
      const edit = state.edits.get(nextIdx);
      return edit ? { ...seg, ...edit } : seg;
    },
    splitAt: (index, cursorPos, explicitTime) =>
      dispatch({ type: "SPLIT", index, cursorPos, explicitTime }),
    reset: (segments) => dispatch({ type: "RESET", segments }),
    undo: () => dispatch({ type: "UNDO" }),
  };
}
