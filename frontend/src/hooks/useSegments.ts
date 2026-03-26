import { useMemo, useReducer } from "react";
import type { Segment } from "@/api/types";

type EditorState = {
  original: Segment[];
  edits: Map<number, Partial<Segment>>;
  deleted: Set<number>;
};

type EditorAction =
  | { type: "SET_TEXT"; index: number; text: string }
  | { type: "SET_SPEAKER"; index: number; speaker: string }
  | { type: "SET_TIMESTAMP"; index: number; field: "start" | "end"; value: number }
  | { type: "DELETE"; index: number }
  | { type: "DELETE_FLAGGED"; indices: number[] }
  | { type: "INSERT"; afterIndex: number; segment: Segment }
  | { type: "RESET"; segments: Segment[] };

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
      return { ...state, edits };
    }
    case "DELETE": {
      const deleted = new Set(state.deleted);
      deleted.add(action.index);
      return { ...state, deleted };
    }
    case "DELETE_FLAGGED": {
      const deleted = new Set(state.deleted);
      for (const idx of action.indices) deleted.add(idx);
      return { ...state, deleted };
    }
    case "INSERT": {
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
      return { original: newOriginal, edits: newEdits, deleted: newDeleted };
    }
    case "RESET":
      return {
        original: action.segments,
        edits: new Map(),
        deleted: new Set<number>(),
      };
  }
}

export interface UseSegmentsReturn {
  editedSegments: Segment[];
  /** Maps each editedSegments index to its original index in the source array. */
  originalIndices: number[];
  isDirty: boolean;
  deletedCount: number;
  flaggedIndices: number[];
  updateText: (index: number, text: string) => void;
  updateSpeaker: (index: number, speaker: string) => void;
  updateTimestamp: (index: number, field: "start" | "end", value: number) => void;
  deleteSegment: (index: number) => void;
  deleteFlagged: () => void;
  insertAfter: (index: number, segment: Segment) => void;
  reset: (segments: Segment[]) => void;
}

export function useSegments(
  initialSegments: Segment[],
): UseSegmentsReturn {
  const [state, dispatch] = useReducer(reducer, {
    original: initialSegments,
    edits: new Map(),
    deleted: new Set<number>(),
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
    flaggedIndices,
    updateText: (index, text) => dispatch({ type: "SET_TEXT", index, text }),
    updateSpeaker: (index, speaker) => dispatch({ type: "SET_SPEAKER", index, speaker }),
    updateTimestamp: (index, field, value) =>
      dispatch({ type: "SET_TIMESTAMP", index, field, value }),
    deleteSegment: (index) => dispatch({ type: "DELETE", index }),
    deleteFlagged: () => dispatch({ type: "DELETE_FLAGGED", indices: flaggedIndices }),
    insertAfter: (index, segment) => dispatch({ type: "INSERT", afterIndex: index, segment }),
    reset: (segments) => dispatch({ type: "RESET", segments }),
  };
}
