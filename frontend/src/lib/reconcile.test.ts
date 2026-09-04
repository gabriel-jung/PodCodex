import { describe, expect, it } from "vitest";
import {
  entryText,
  parseResponseObjects,
  promptToInputEntries,
  reconcileBatch,
} from "./reconcile";

/**
 * These guard the manual paste flow (ManualModePanel, BatchReconcileModal).
 * A count or index mismatch they fail to catch writes corrected or translated
 * text onto the wrong segments, and the drift signals behind the error message
 * are heuristics that only a fixture table keeps honest.
 */

const INPUT = [
  { index: 0, text: "the first line" },
  { index: 1, text: "the second line" },
  { index: 2, text: "the third line" },
];

const asJson = (xs: unknown[]) => JSON.stringify(xs);

describe("parseResponseObjects", () => {
  it("accepts an array of objects", () => {
    expect(parseResponseObjects('[{"text":"a"},{"text":"b"}]')).toEqual([
      { text: "a" },
      { text: "b" },
    ]);
  });

  it("wraps a bare object in an array", () => {
    expect(parseResponseObjects('{"text":"a"}')).toEqual([{ text: "a" }]);
  });

  it("promotes bare strings to {text}", () => {
    expect(parseResponseObjects('["a","b"]')).toEqual([{ text: "a" }, { text: "b" }]);
  });

  it("preserves fields other than text", () => {
    expect(parseResponseObjects('[{"index":4,"text":"a"}]')).toEqual([
      { index: 4, text: "a" },
    ]);
  });

  it("returns null on malformed JSON rather than throwing", () => {
    expect(parseResponseObjects("not json")).toBeNull();
    expect(parseResponseObjects("")).toBeNull();
  });
});

describe("entryText", () => {
  it("stringifies a missing text field to empty", () => {
    expect(entryText({})).toBe("");
    expect(entryText({ text: "hi" })).toBe("hi");
  });
});

describe("promptToInputEntries", () => {
  it("reads the [N] lines and ignores the instruction block", () => {
    const prompt = [
      "Correct the following:",
      "[0] the first line",
      "[1] the second line",
      "",
      "Return JSON only.",
    ].join("\n");
    expect(promptToInputEntries(prompt)).toEqual([
      { index: 0, text: "the first line" },
      { index: 1, text: "the second line" },
    ]);
  });

  it("sorts by index, so a shuffled prompt still lines up", () => {
    const prompt = "[7] seven\n[3] three";
    expect(promptToInputEntries(prompt).map((e) => e.index)).toEqual([3, 7]);
  });

  it("returns nothing for a prompt with no markers", () => {
    expect(promptToInputEntries("just prose")).toEqual([]);
  });
});

describe("reconcileBatch", () => {
  it("accepts a response with matching counts", () => {
    const out = reconcileBatch(asJson(INPUT.map((e) => ({ text: e.text }))), INPUT);
    expect(out).toHaveProperty("objs");
  });

  it("rejects unparseable text", () => {
    const out = reconcileBatch("oops", INPUT);
    expect(out).toEqual({ error: expect.stringContaining("Invalid JSON") });
  });

  it("rejects an off-by-one response and says both counts", () => {
    const out = reconcileBatch(asJson([{ text: "a" }, { text: "b" }]), INPUT);
    expect("error" in out && out.error).toContain("Expected 3 entries, got 2");
  });

  it("names the segment where an echoed index first drifts", () => {
    // The LLM dropped [1] and renumbered, so entry 1 claims index 2.
    const out = reconcileBatch(
      asJson([
        { index: 0, text: "the first line" },
        { index: 2, text: "the third line" },
      ]),
      INPUT,
    );
    expect("error" in out && out.error).toContain("[1]");
    expect("error" in out && out.error).toContain("index=2");
  });

  it("names the segment where the text stops resembling the input", () => {
    const out = reconcileBatch(
      asJson([{ text: "the first line" }, { text: "completely unrelated content" }]),
      INPUT,
    );
    expect("error" in out && out.error).toContain("[1]");
    expect("error" in out && out.error).toContain("the second line");
  });

  it("tolerates a corrected line that is still recognisably the same", () => {
    // Punctuation and diacritics must not read as drift.
    const out = reconcileBatch(
      asJson([
        { text: "The first line!" },
        { text: "the second line…" },
        { text: "The third line." },
      ]),
      INPUT,
    );
    expect(out).toHaveProperty("objs");
  });

  it("falls back to head and tail when nothing pinpoints the drift", () => {
    const out = reconcileBatch(
      asJson([
        { text: "the first line" },
        { text: "the second line" },
        { text: "the third line" },
        { text: "the third line" },
      ]),
      INPUT,
    );
    expect("error" in out && out.error).toContain("Head:");
    expect("error" in out && out.error).toContain("Tail:");
  });

  it("handles a bare-string response of the right length", () => {
    const out = reconcileBatch(asJson(INPUT.map((e) => e.text)), INPUT);
    expect(out).toHaveProperty("objs");
  });

  it("reports drift in a later batch at that batch's own indices", () => {
    const offset = [
      { index: 10, text: "the tenth line" },
      { index: 11, text: "the eleventh line" },
      { index: 12, text: "the twelfth line" },
    ];
    const out = reconcileBatch(
      asJson([
        { index: 10, text: "the tenth line" },
        { index: 12, text: "the twelfth line" },
      ]),
      offset,
    );
    expect("error" in out && out.error).toContain("[11]");
  });

  it("passes a matching count even when the echoed index is wrong", () => {
    // Documented, not accidental: both this and the backend map by position,
    // so the `index` field is only ever a drift *signal*, consulted when the
    // counts already disagree.
    const out = reconcileBatch(
      asJson(INPUT.map((e, i) => ({ index: i + 99, text: e.text }))),
      INPUT,
    );
    expect(out).toHaveProperty("objs");
  });
});
