import { describe, expect, it } from "vitest";
import type { Segment } from "@/api/types";
import { initEditorState, segmentsReducer, type EditorAction, type EditorState } from "./useSegments";

/** The transcript editor's state machine: edits, structure and undo. A
 *  regression here loses user edits, so the history rules are pinned. */

const seg = (text: string, start: number, end: number, speaker = "A"): Segment =>
  ({ speaker, text, start, end }) as Segment;

const run = (state: EditorState, ...actions: EditorAction[]) =>
  actions.reduce(segmentsReducer, state);

const texts = (s: EditorState) =>
  s.rows.filter((r) => !s.deleted.has(r.id)).map((r) => ({ ...r.base, ...s.edits.get(r.id) }).text);

const start = () => initEditorState([seg("hello world", 0, 2), seg("second", 2, 4)]);

describe("segmentsReducer", () => {
  it("splits at the cursor and merges back", () => {
    const split = run(start(), { type: "SPLIT", id: 0, cursorPos: 5 });
    expect(texts(split)).toEqual(["hello", "world", "second"]);
    const merged = run(split, { type: "MERGE", id: 0 });
    expect(texts(merged)).toEqual(["hello world", "second"]);
  });

  it("undoes a burst of typing as one step", () => {
    const s = run(
      start(),
      { type: "SET_TEXT", id: 0, text: "h" },
      { type: "SET_TEXT", id: 0, text: "hi" },
      { type: "SET_TEXT", id: 0, text: "hi there" },
    );
    expect(s.history).toHaveLength(1);
    expect(texts(run(s, { type: "UNDO" }))).toEqual(["hello world", "second"]);
  });

  it("keeps a structural edit when undoing the typing that followed it", () => {
    const s = run(
      start(),
      { type: "DELETE", id: 1 },
      { type: "SET_TEXT", id: 0, text: "typed after the delete" },
    );
    const once = run(s, { type: "UNDO" });
    expect(texts(once)).toEqual(["hello world"]);
    const twice = run(once, { type: "UNDO" });
    expect(texts(twice)).toEqual(["hello world", "second"]);
  });

  it("keeps every timestamp change its own undo step", () => {
    // A typed time and a "set to playhead" click dispatch the same action.
    const s = run(
      start(),
      { type: "SET_TIMESTAMP", id: 0, field: "start", value: 0.5 },
      { type: "SET_TIMESTAMP", id: 0, field: "start", value: 0.7 },
    );
    expect(s.history).toHaveLength(2);
  });

  it("starts a new undo step when typing moves to another row", () => {
    const s = run(
      start(),
      { type: "SET_TEXT", id: 0, text: "a" },
      { type: "SET_TEXT", id: 1, text: "b" },
    );
    expect(s.history).toHaveLength(2);
  });

  it("merges onto the empty-string speaker when it is picked", () => {
    const s = run(
      initEditorState([seg("one", 0, 1, "A"), seg("two", 1, 2, "")]),
      { type: "MERGE", id: 0, speaker: "" },
    );
    expect(s.edits.get(0)?.speaker).toBe("");
  });

  it("drops edits and history on reset, with fresh row ids", () => {
    const s = run(start(), { type: "SET_TEXT", id: 0, text: "x" }, { type: "RESET", segments: [seg("new", 0, 1)] });
    expect(s.history).toEqual([]);
    expect(s.edits.size).toBe(0);
    expect(s.rows[0].id).toBe(2);
  });
});
