import { useCallback, useMemo, useReducer, useRef } from "react";
import type { Segment } from "@/api/types";

/** A row carries its own stable id. Position in `rows` is the current display
 *  order; the id never changes for the row's lifetime. Inserts/splits add new
 *  rows with fresh ids, so edits/deleted Sets keyed by id stay valid across
 *  structural changes (no shifting needed). */
export interface Row {
  id: number;
  base: Segment;
  /** Index of the source segment this row came from. null for rows inserted
   *  this session — they have no counterpart in the diff reference. */
  sourceIndex: number | null;
}

type Snapshot = {
  rows: Row[];
  edits: Map<number, Partial<Segment>>;
  deleted: Set<number>;
};

type EditorState = Snapshot & {
  history: Snapshot[];
  nextId: number;
};

type EditorAction =
  | { type: "SET_TEXT"; id: number; text: string }
  | { type: "SET_SPEAKER"; id: number; speaker: string }
  | { type: "SET_TIMESTAMP"; id: number; field: "start" | "end"; value: number }
  | { type: "DELETE"; id: number }
  | { type: "DELETE_FLAGGED"; ids: number[] }
  | { type: "RESTORE"; id: number }
  | { type: "INSERT"; afterId: number; segment: Segment }
  | { type: "RESET"; segments: Segment[] }
  | { type: "UNDO" }
  | { type: "MERGE"; id: number; speaker?: string }
  | {
      type: "SPLIT";
      id: number;
      cursorPos: number;
      explicitTime?: number;
      /** Speaker to assign to BOTH halves. Lets the caller pre-resolve any
       *  staged rename (e.g. from a SpeakerStrip pendingRename) so splits
       *  inherit the displayed name, not the raw base name. */
      resolvedSpeaker?: string;
    };

const MAX_HISTORY = 50;

function snap(state: EditorState): Snapshot {
  return {
    rows: state.rows,
    edits: state.edits,
    deleted: state.deleted,
  };
}

function pushHistory(state: EditorState): Snapshot[] {
  const h = [...state.history, snap(state)];
  return h.length > MAX_HISTORY ? h.slice(-MAX_HISTORY) : h;
}

function findPos(rows: Row[], id: number): number {
  return rows.findIndex((r) => r.id === id);
}

function mergedSeg(row: Row, edits: Map<number, Partial<Segment>>): Segment {
  const e = edits.get(row.id);
  return e ? { ...row.base, ...e } : row.base;
}

function reducer(state: EditorState, action: EditorAction): EditorState {
  switch (action.type) {
    case "SET_TEXT":
    case "SET_SPEAKER":
    case "SET_TIMESTAMP": {
      const edits = new Map(state.edits);
      const existing = edits.get(action.id) || {};
      if (action.type === "SET_TEXT") {
        edits.set(action.id, { ...existing, text: action.text });
      } else if (action.type === "SET_SPEAKER") {
        edits.set(action.id, { ...existing, speaker: action.speaker });
      } else {
        edits.set(action.id, { ...existing, [action.field]: action.value });
      }
      // Per-keystroke text edits don't push history (would flood). Discrete
      // commits (speaker / timestamp) do.
      if (action.type === "SET_TEXT") return { ...state, edits };
      return { ...state, history: pushHistory(state), edits };
    }
    case "DELETE": {
      const history = pushHistory(state);
      const deleted = new Set(state.deleted);
      deleted.add(action.id);
      return { ...state, history, deleted };
    }
    case "DELETE_FLAGGED": {
      const history = pushHistory(state);
      const deleted = new Set(state.deleted);
      for (const id of action.ids) deleted.add(id);
      return { ...state, history, deleted };
    }
    case "RESTORE": {
      if (!state.deleted.has(action.id)) return state;
      const history = pushHistory(state);
      const deleted = new Set(state.deleted);
      deleted.delete(action.id);
      return { ...state, history, deleted };
    }
    case "INSERT": {
      const pos = findPos(state.rows, action.afterId);
      if (pos < 0) return state;
      const history = pushHistory(state);
      const newRow: Row = { id: state.nextId, base: action.segment, sourceIndex: null };
      const rows = [...state.rows.slice(0, pos + 1), newRow, ...state.rows.slice(pos + 1)];
      return { ...state, history, rows, nextId: state.nextId + 1 };
    }
    case "RESET": {
      // Monotonic ids across resets — never reuse an id that a stale
      // React-side Set (selection, dismissed flags, etc.) might still hold.
      const startId = state.nextId;
      const rows: Row[] = action.segments.map((seg, i) => ({
        id: startId + i,
        base: seg,
        sourceIndex: i,
      }));
      return {
        rows,
        edits: new Map(),
        deleted: new Set<number>(),
        history: [],
        nextId: startId + action.segments.length,
      };
    }
    case "UNDO": {
      if (state.history.length === 0) return state;
      const history = [...state.history];
      const prev = history.pop()!;
      return { ...state, ...prev, history };
    }
    case "MERGE": {
      const pos = findPos(state.rows, action.id);
      if (pos < 0) return state;
      let nextPos = pos + 1;
      while (nextPos < state.rows.length && state.deleted.has(state.rows[nextPos].id)) nextPos++;
      if (nextPos >= state.rows.length) return state;
      const row = state.rows[pos];
      const nextRow = state.rows[nextPos];
      const seg = mergedSeg(row, state.edits);
      const next = mergedSeg(nextRow, state.edits);
      const history = pushHistory(state);
      const edits = new Map(state.edits);
      // Preserve any prior edits on row.id (notably a speaker edit stamped by
      // a previous SPLIT) — only override text/end and, if requested, speaker.
      const prior = edits.get(row.id) ?? {};
      const speakerEdit = action.speaker ? { speaker: action.speaker } : {};
      edits.set(row.id, {
        ...prior,
        ...speakerEdit,
        text: seg.text + " " + next.text,
        end: next.end,
      });
      const deleted = new Set(state.deleted);
      deleted.add(nextRow.id);
      return { ...state, history, edits, deleted };
    }
    case "SPLIT": {
      const pos = findPos(state.rows, action.id);
      if (pos < 0) return state;
      const row = state.rows[pos];
      const seg = mergedSeg(row, state.edits);
      const textBefore = seg.text.slice(0, action.cursorPos).trimEnd();
      const textAfter = seg.text.slice(action.cursorPos).trimStart();
      // Explicit timestamp (e.g. current playback position) overrides
      // proportional estimate.
      const splitTime = action.explicitTime != null
        ? Math.round(action.explicitTime * 10) / 10
        : (() => {
            const ratio = textBefore.length / Math.max(seg.text.length, 1);
            return Math.round((seg.start + (seg.end - seg.start) * ratio) * 10) / 10;
          })();
      const newSpeaker = action.resolvedSpeaker ?? seg.speaker;
      const history = pushHistory(state);
      const edits = new Map(state.edits);
      // Stamp the resolved speaker on the original row too so a later
      // pendingRename change to the base speaker doesn't desync the two halves.
      const prior = edits.get(row.id) ?? {};
      const speakerEdit = newSpeaker !== row.base.speaker ? { speaker: newSpeaker } : {};
      edits.set(row.id, { ...prior, ...speakerEdit, text: textBefore, end: splitTime });
      const newRow: Row = {
        id: state.nextId,
        base: {
          speaker: newSpeaker,
          text: textAfter,
          start: splitTime,
          end: seg.end,
          flagged: false,
        },
        sourceIndex: null,
      };
      const rows = [...state.rows.slice(0, pos + 1), newRow, ...state.rows.slice(pos + 1)];
      return { ...state, history, rows, edits, nextId: state.nextId + 1 };
    }
  }
}

export interface UseSegmentsReturn {
  /** Edited segments, non-deleted, in display order. */
  editedSegments: Segment[];
  /** Stable row id per `editedSegments` entry. Use as React/virtualizer key. */
  ids: number[];
  /** All rows, including pending-deleted ones (rendered with strike-through). */
  allEditedSegments: Segment[];
  allIds: number[];
  /** Diff-reference index per row in `allEditedSegments`; null for rows
   *  inserted this session. */
  allSourceIndices: (number | null)[];
  /** Ids of rows pending deletion. */
  deletedSet: ReadonlySet<number>;
  isDirty: boolean;
  deletedCount: number;
  canUndo: boolean;
  /** Ids of currently-flagged segments. */
  flaggedIds: number[];
  /** Ids of rows whose text differs from the loaded base (per-row text
   *  edits + freshly inserted/split rows). Used to highlight the textarea
   *  visually so unsaved edits are easy to spot. */
  textEditedIds: ReadonlySet<number>;
  updateText: (id: number, text: string) => void;
  updateSpeaker: (id: number, speaker: string) => void;
  updateTimestamp: (id: number, field: "start" | "end", value: number) => void;
  deleteSegment: (id: number) => void;
  deleteFlagged: () => void;
  restoreSegment: (id: number) => void;
  insertAfter: (id: number, segment: Segment) => void;
  mergeWithNext: (id: number, speaker?: string) => void;
  getNextSegment: (id: number) => Segment | null;
  splitAt: (
    id: number,
    cursorPos: number,
    explicitTime?: number,
    resolvedSpeaker?: string,
  ) => void;
  reset: (segments: Segment[]) => void;
  undo: () => void;
}

export function useSegments(initialSegments: Segment[]): UseSegmentsReturn {
  const [state, dispatch] = useReducer(reducer, undefined, () => ({
    rows: initialSegments.map((seg, i) => ({ id: i, base: seg, sourceIndex: i })),
    edits: new Map<number, Partial<Segment>>(),
    deleted: new Set<number>(),
    history: [],
    nextId: initialSegments.length,
  }));

  const derived = useMemo(() => {
    const editedSegments: Segment[] = [];
    const ids: number[] = [];
    const allEditedSegments: Segment[] = [];
    const allIds: number[] = [];
    const allSourceIndices: (number | null)[] = [];
    for (const row of state.rows) {
      const merged = mergedSeg(row, state.edits);
      allEditedSegments.push(merged);
      allIds.push(row.id);
      allSourceIndices.push(row.sourceIndex);
      if (state.deleted.has(row.id)) continue;
      editedSegments.push(merged);
      ids.push(row.id);
    }
    return { editedSegments, ids, allEditedSegments, allIds, allSourceIndices };
  }, [state.rows, state.edits, state.deleted]);

  const isDirty = state.edits.size > 0 || state.deleted.size > 0
    || state.rows.length !== initialSegments.length;

  const flaggedIds = useMemo(() => {
    const out: number[] = [];
    for (const row of state.rows) {
      if (state.deleted.has(row.id)) continue;
      if (mergedSeg(row, state.edits).flagged) out.push(row.id);
    }
    return out;
  }, [state.rows, state.edits, state.deleted]);

  const textEditedIds = useMemo(() => {
    const s = new Set<number>();
    for (const [id, edit] of state.edits) {
      if (edit.text !== undefined) s.add(id);
    }
    // Inserted / split rows have no source counterpart — their text is
    // user-authored too, so highlight them as edited.
    for (const row of state.rows) {
      if (row.sourceIndex == null && !state.deleted.has(row.id)) s.add(row.id);
    }
    return s;
  }, [state.rows, state.edits, state.deleted]);

  // Stable callbacks need fresh state for predicates that scan live rows
  // (`getNextSegment`) or pull a snapshot of flagged ids (`deleteFlagged`).
  const stateRef = useRef(state);
  const flaggedRef = useRef(flaggedIds);
  // eslint-disable-next-line react-hooks/refs
  stateRef.current = state;
  // eslint-disable-next-line react-hooks/refs
  flaggedRef.current = flaggedIds;

  const updateText = useCallback((id: number, text: string) => {
    dispatch({ type: "SET_TEXT", id, text });
  }, []);
  const updateSpeaker = useCallback((id: number, speaker: string) => {
    dispatch({ type: "SET_SPEAKER", id, speaker });
  }, []);
  const updateTimestamp = useCallback(
    (id: number, field: "start" | "end", value: number) => {
      dispatch({ type: "SET_TIMESTAMP", id, field, value });
    },
    [],
  );
  const deleteSegment = useCallback((id: number) => {
    dispatch({ type: "DELETE", id });
  }, []);
  const deleteFlagged = useCallback(() => {
    dispatch({ type: "DELETE_FLAGGED", ids: flaggedRef.current });
  }, []);
  const restoreSegment = useCallback((id: number) => {
    dispatch({ type: "RESTORE", id });
  }, []);
  const insertAfter = useCallback((id: number, segment: Segment) => {
    dispatch({ type: "INSERT", afterId: id, segment });
  }, []);
  const mergeWithNext = useCallback((id: number, speaker?: string) => {
    dispatch({ type: "MERGE", id, speaker });
  }, []);
  const getNextSegment = useCallback((id: number): Segment | null => {
    const s = stateRef.current;
    const pos = s.rows.findIndex((r) => r.id === id);
    if (pos < 0) return null;
    let nextPos = pos + 1;
    while (nextPos < s.rows.length && s.deleted.has(s.rows[nextPos].id)) nextPos++;
    if (nextPos >= s.rows.length) return null;
    return mergedSeg(s.rows[nextPos], s.edits);
  }, []);
  const splitAt = useCallback(
    (id: number, cursorPos: number, explicitTime?: number, resolvedSpeaker?: string) => {
      dispatch({ type: "SPLIT", id, cursorPos, explicitTime, resolvedSpeaker });
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
    ...derived,
    deletedSet: state.deleted,
    isDirty,
    deletedCount: state.deleted.size,
    canUndo: state.history.length > 0,
    flaggedIds,
    textEditedIds,
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
