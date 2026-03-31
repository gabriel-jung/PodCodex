/** Word-level diff utilities — highlights removed and added words using LCS. */

export type DiffPart = { type: "same" | "removed" | "added"; text: string };

export function computeWordDiff(original: string, current: string): DiffPart[] {
  const a = original.split(/\s+/).filter(Boolean);
  const b = current.split(/\s+/).filter(Boolean);

  const m = a.length, n = b.length;
  // For very long segments, use a sliding-window approach instead of full LCS
  if (m * n > 50000) {
    // Find common prefix and suffix, diff only the middle
    let prefix = 0;
    while (prefix < m && prefix < n && a[prefix] === b[prefix]) prefix++;
    let suffix = 0;
    while (suffix < m - prefix && suffix < n - prefix && a[m - 1 - suffix] === b[n - 1 - suffix]) suffix++;
    const parts: DiffPart[] = [];
    if (prefix > 0) parts.push({ type: "same", text: a.slice(0, prefix).join(" ") });
    const removedMiddle = a.slice(prefix, m - suffix);
    const addedMiddle = b.slice(prefix, n - suffix);
    if (removedMiddle.length > 0) parts.push({ type: "removed", text: removedMiddle.join(" ") });
    if (addedMiddle.length > 0) parts.push({ type: "added", text: addedMiddle.join(" ") });
    if (suffix > 0) parts.push({ type: "same", text: a.slice(m - suffix).join(" ") });
    return parts;
  }

  // LCS table
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  // Backtrack
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

  // Merge consecutive same-type parts, joining with spaces
  const parts: DiffPart[] = [];
  for (const part of stack) {
    if (parts.length > 0 && parts[parts.length - 1].type === part.type) {
      parts[parts.length - 1].text += " " + part.text;
    } else {
      parts.push({ ...part });
    }
  }
  return parts;
}
