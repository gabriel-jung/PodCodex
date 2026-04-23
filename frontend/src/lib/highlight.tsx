import type { ReactNode } from "react";

const HYPHENS_RE = /[-‐-―−]/;
const APOSTROPHES_RE = /['‘’‚‛′ʼ]/;
const COMBINING_MARKS_RE = /[̀-ͯ]/g;

/** Fold a single original character to 0+ folded chars (mirrors Python fold_text). */
function foldChar(c: string): string {
  if (HYPHENS_RE.test(c) || APOSTROPHES_RE.test(c)) return " ";
  if (c === "œ" || c === "Œ") return "oe";
  if (c === "æ" || c === "Æ") return "ae";
  return c.normalize("NFD").replace(COMBINING_MARKS_RE, "").toLowerCase();
}

function foldAccents(s: string): string {
  // Iterate by code point so non-BMP chars (4-byte Unicode) aren't split
  // into surrogate halves and fed to foldChar as invalid input.
  const parts: string[] = [];
  for (const c of s) parts.push(foldChar(c));
  return parts.join("");
}

/** Fold a string while keeping a per-folded-char index back to the original. */
function foldWithMap(s: string): { folded: string; origOffsets: number[] } {
  const parts: string[] = [];
  const origOffsets: number[] = [];
  let i = 0;
  for (const c of s) {
    const out = foldChar(c);
    for (let k = 0; k < out.length; k++) {
      parts.push(out[k]);
      origOffsets.push(i);
    }
    i += c.length;
  }
  return { folded: parts.join(""), origOffsets };
}

/**
 * Split text into alternating plain / highlighted spans using accent- and
 * case-insensitive matching. Ligatures (œ, æ) expand during folding but
 * indices map back to the original string, so a query "oeuvre" highlights
 * "œuvre" and other matches in the same paragraph are not dropped.
 */
export function highlightText(text: string, query: string): ReactNode {
  if (!query.trim()) return text;

  const { folded: foldedText, origOffsets } = foldWithMap(text);
  const foldedQuery = foldAccents(query);
  if (!foldedQuery || !foldedText.includes(foldedQuery)) return text;

  const nodes: ReactNode[] = [];
  let last = 0;
  const qLen = foldedQuery.length;
  let idx = foldedText.indexOf(foldedQuery);

  while (idx !== -1) {
    const origStart = origOffsets[idx];
    const lastChar = origOffsets[idx + qLen - 1];
    const origEnd = lastChar != null ? lastChar + 1 : text.length;
    if (origStart > last) nodes.push(text.slice(last, origStart));
    nodes.push(
      <mark
        key={idx}
        className="bg-yellow-200/80 dark:bg-yellow-700/50 text-inherit rounded-sm px-px"
      >
        {text.slice(origStart, origEnd)}
      </mark>,
    );
    last = origEnd;
    idx = foldedText.indexOf(foldedQuery, idx + qLen);
  }

  if (last < text.length) nodes.push(text.slice(last));
  return <>{nodes}</>;
}
