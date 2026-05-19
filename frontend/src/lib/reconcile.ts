/** Helpers for reconciling a pasted/auto LLM batch response against the
 *  input segments it was meant to correct or translate. Shared by the manual
 *  paste flow and the batch-fix modal. */

/** One input segment a batch was built from. `index` is the absolute
 *  (non-[BREAK]) segment index in the source transcript. */
export interface InputEntry {
  index: number;
  text: string;
}

/** Parse a pasted JSON response into entry objects, or null when it does not
 *  parse. Bare strings become `{text}`; other fields are preserved. */
export function parseResponseObjects(text: string): Record<string, unknown>[] | null {
  try {
    const parsed = JSON.parse(text);
    const arr = Array.isArray(parsed) ? parsed : [parsed];
    return arr.map((x) =>
      typeof x === "string" ? { text: x } : ((x as Record<string, unknown>) ?? {}),
    );
  } catch {
    return null;
  }
}

export function entryText(obj: Record<string, unknown>): string {
  return String(obj.text ?? "");
}

/** Pull expected `[N] text` lines out of a manual-mode prompt. Ignores the
 *  trailing instruction block. */
export function promptToInputEntries(prompt: string): InputEntry[] {
  const entries: InputEntry[] = [];
  const re = /^\[(\d+)\]\s+(.+)$/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(prompt)) !== null) {
    entries.push({ index: Number(m[1]), text: m[2] });
  }
  return entries.sort((a, b) => a.index - b.index);
}

/** Validate a pasted response against the expected input entries. Returns the
 *  parsed objects on success, or a human-readable error string. */
export function reconcileBatch(
  text: string,
  inputEntries: InputEntry[],
):
  | { objs: Record<string, unknown>[] }
  | { error: string } {
  const objs = parseResponseObjects(text);
  if (objs == null) {
    return { error: "Invalid JSON — could not parse the response." };
  }
  const expected = inputEntries.length;
  if (objs.length !== expected) {
    const drift = describeDrift(objs, inputEntries);
    return {
      error:
        `Expected ${expected} entries, got ${objs.length}. ` +
        (drift ? drift + " " : "") +
        "Regenerate or trim the response so the counts match.",
    };
  }
  return { objs };
}

/** Locate the first point where the response diverges from the input.
 *  Three signals, in order of reliability:
 *    1. `index` field on entries (the LLM often echoes the [N] markers).
 *    2. Text similarity to the corresponding input entry.
 *    3. Fallback: surface the first and last few entries so the user can
 *       eyeball where the list got too long or short. */
function describeDrift(
  arr: unknown[],
  inputEntries: InputEntry[],
): string | null {
  const expected = new Map(inputEntries.map((e) => [e.index, e.text]));
  const startIndex = inputEntries[0]?.index ?? 0;
  const preview = (s: unknown, n = 50) =>
    String(s ?? "").replace(/\s+/g, " ").slice(0, n);

  // Signal 1: explicit index field drift
  for (let i = 0; i < arr.length; i++) {
    const item = arr[i] as { index?: unknown; text?: unknown } | null;
    if (item && typeof item.index === "number") {
      const want = i + startIndex;
      if (item.index !== want) {
        const expText = expected.get(want) ?? "(input entry missing)";
        return (
          `Drift begins around segment [${want}]: expected ` +
          `"${preview(expText)}…", got index=${item.index} ` +
          `("${preview(item.text)}…").`
        );
      }
    }
  }

  // Signal 2: text similarity (Jaccard on first ~40 chars)
  if (expected.size > 0) {
    for (let i = 0; i < arr.length; i++) {
      const expText = expected.get(i + startIndex);
      if (!expText) continue;
      const gotText = (arr[i] as { text?: unknown })?.text;
      if (typeof gotText !== "string") continue;
      if (!looksRelated(expText, gotText)) {
        return (
          `Drift begins around segment [${i + startIndex}]: input had ` +
          `"${preview(expText)}…", response has "${preview(gotText)}…".`
        );
      }
    }
  }

  // Signal 3: count-only difference — show head/tail so user can scan
  const diff = arr.length - expected.size;
  if (diff !== 0 && arr.length > 0) {
    const head = arr
      .slice(0, 2)
      .map((x, i) => `[${startIndex + i}]: "${preview((x as { text?: unknown })?.text)}…"`);
    const tail = arr
      .slice(-2)
      .map(
        (x, i) =>
          `[${startIndex + arr.length - 2 + i}]: ` +
          `"${preview((x as { text?: unknown })?.text)}…"`,
      );
    return `No index field to pinpoint drift. Head: ${head.join(", ")}. Tail: ${tail.join(", ")}.`;
  }
  return null;
}

/** Cheap "same-ish line" check. Normalises punctuation and diacritics before
 *  tokenising so LLM corrections like "moi" → "muy" or "Si" → "Sí" still count
 *  as related. Low threshold (0.2) keeps short segments from producing false
 *  positives — the goal is only to catch clearly different lines. */
function looksRelated(a: string, b: string): boolean {
  const toks = (s: string) =>
    new Set(
      s
        .normalize("NFD")
        .replace(/[̀-ͯ]/g, "") // strip combining diacritics
        .toLowerCase()
        .replace(/[^\p{L}\p{N}\s]/gu, " ") // strip punctuation
        .split(/\s+/)
        .slice(0, 10)
        .filter((w) => w.length > 1),
    );
  const A = toks(a);
  const B = toks(b);
  if (A.size === 0 || B.size === 0) return true;
  let inter = 0;
  for (const w of A) if (B.has(w)) inter++;
  return inter / Math.min(A.size, B.size) >= 0.2;
}
