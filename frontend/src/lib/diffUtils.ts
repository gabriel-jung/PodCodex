/** Word-level diff with character-level refinement.
 *
 * A plain word diff flags a whole word when only its punctuation changed
 * ("truc" vs "truc," become a full removed+added pair). After the word diff
 * we re-diff each adjacent removed/added run at character level when the two
 * strings are clearly related, so a punctuation-scale edit highlights just
 * the punctuation, not the surrounding word.
 *
 * Tokens carry their own spacing (whitespace runs collapse to single-space
 * tokens), so the renderer concatenates `part.text` directly — it must not
 * insert spaces between parts.
 */

export type DiffPart = { type: "same" | "removed" | "added"; text: string };

// Product of token-array lengths above which the full LCS table is skipped
// for a cheaper prefix/suffix trim. Token arrays run ~2x word count.
const LCS_LIMIT = 200000;
// Minimum share of the shorter string that must survive as common chars for
// a removed/added pair to be refined at char level. Below this the two words
// are unrelated and a char diff would be noise — keep them as whole blocks.
const REFINE_MIN_RATIO = 0.5;
// Char-product ceiling above which a removed/added pair is left as whole
// blocks rather than char-diffed. Refinement targets word/punctuation-scale
// edits; the long-segment fast path can hand it transcript-sized strings,
// where a char LCS table would be needlessly huge.
const REFINE_CHAR_LIMIT = 10000;

/** Split into alternating word / single-space tokens. */
function tokenize(s: string): string[] {
  const words = s.split(/\s+/).filter(Boolean);
  const out: string[] = [];
  words.forEach((w, i) => {
    if (i > 0) out.push(" ");
    out.push(w);
  });
  return out;
}

/** LCS diff over an arbitrary token array (one DiffPart per token). */
function lcsDiff(a: string[], b: string[]): DiffPart[] {
  const m = a.length, n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  const stack: DiffPart[] = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
      stack.push({ type: "same", text: a[i - 1] });
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      stack.push({ type: "added", text: b[j - 1] });
      j--;
    } else {
      stack.push({ type: "removed", text: a[i - 1] });
      i--;
    }
  }
  stack.reverse();
  return stack;
}

/** Merge consecutive same-type parts into one (text concatenated, no spacer). */
function mergeAdjacent(parts: DiffPart[]): DiffPart[] {
  const out: DiffPart[] = [];
  for (const part of parts) {
    const last = out[out.length - 1];
    if (last && last.type === part.type) last.text += part.text;
    else out.push({ ...part });
  }
  return out;
}

/** Re-diff a removed/added pair at character level when the two strings are
 *  clearly related; otherwise keep them as whole removed + added blocks. */
function refinePair(removed: string, added: string): DiffPart[] {
  if (removed.length * added.length > REFINE_CHAR_LIMIT) {
    return [{ type: "removed", text: removed }, { type: "added", text: added }];
  }
  const chars = lcsDiff([...removed], [...added]);
  const common = chars.reduce((s, c) => (c.type === "same" ? s + c.text.length : s), 0);
  const minLen = Math.min(removed.length, added.length);
  if (minLen === 0 || common / minLen < REFINE_MIN_RATIO) {
    return [{ type: "removed", text: removed }, { type: "added", text: added }];
  }
  return mergeAdjacent(chars);
}

export function computeWordDiff(original: string, current: string): DiffPart[] {
  const a = tokenize(original);
  const b = tokenize(current);
  const m = a.length, n = b.length;

  // For very long segments, skip the full LCS table: diff only the middle
  // between a shared prefix and suffix.
  if (m * n > LCS_LIMIT) {
    let prefix = 0;
    while (prefix < m && prefix < n && a[prefix] === b[prefix]) prefix++;
    let suffix = 0;
    while (suffix < m - prefix && suffix < n - prefix && a[m - 1 - suffix] === b[n - 1 - suffix]) suffix++;
    const parts: DiffPart[] = [];
    if (prefix > 0) parts.push({ type: "same", text: a.slice(0, prefix).join("") });
    const removedMiddle = a.slice(prefix, m - suffix).join("");
    const addedMiddle = b.slice(prefix, n - suffix).join("");
    if (removedMiddle && addedMiddle) parts.push(...refinePair(removedMiddle, addedMiddle));
    else if (removedMiddle) parts.push({ type: "removed", text: removedMiddle });
    else if (addedMiddle) parts.push({ type: "added", text: addedMiddle });
    if (suffix > 0) parts.push({ type: "same", text: a.slice(m - suffix).join("") });
    return mergeAdjacent(parts);
  }

  const parts = mergeAdjacent(lcsDiff(a, b));

  // Refine each adjacent removed → added run at character level.
  const refined: DiffPart[] = [];
  for (let k = 0; k < parts.length; k++) {
    const part = parts[k];
    const next = parts[k + 1];
    if (part.type === "removed" && next && next.type === "added") {
      refined.push(...refinePair(part.text, next.text));
      k++;
    } else {
      refined.push(part);
    }
  }
  return mergeAdjacent(refined);
}
