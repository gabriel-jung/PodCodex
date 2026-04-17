import type { ReactNode } from "react";

// Mirrors Python's fold_text(): apostrophes/hyphens → space, ligatures expanded,
// diacritics stripped, lowercase.
function foldAccents(s: string): string {
  return s
    .replace(/[\u002D\u2010-\u2015\u2212]/g, " ")                          // hyphens → space
    .replace(/[\u0027\u2018\u2019\u201A\u201B\u2032\u02BC]/g, " ")         // apostrophes → space
    .replace(/[œŒ]/g, "oe").replace(/[æÆ]/g, "ae")                         // ligatures
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

/**
 * Split text into alternating plain / highlighted spans using accent- and
 * case-insensitive matching.  Returns an array of ReactNode so callers can
 * embed it inside any element.
 *
 * Example:
 *   highlightText("L'éducation nationale", "education")
 *   → ["L'", <mark>éducation</mark>, " nationale"]
 */
export function highlightText(text: string, query: string): ReactNode {
  if (!query.trim()) return text;

  const foldedText = foldAccents(text);
  const foldedQuery = foldAccents(query);
  if (!foldedText.includes(foldedQuery)) return text;

  // When ligatures expand, folded length differs from original — positions no
  // longer align, so highlighting would cut wrong spans. Return plain text.
  if (foldedText.length !== text.length) return text;

  const nodes: ReactNode[] = [];
  let last = 0;
  let idx = foldedText.indexOf(foldedQuery);
  const matchLen = foldedQuery.length; // use folded length, not query.length

  while (idx !== -1) {
    if (idx > last) nodes.push(text.slice(last, idx));
    nodes.push(
      <mark
        key={idx}
        className="bg-yellow-200/80 dark:bg-yellow-700/50 text-inherit rounded-sm px-px"
      >
        {text.slice(idx, idx + matchLen)}
      </mark>,
    );
    last = idx + matchLen;
    idx = foldedText.indexOf(foldedQuery, last);
  }

  if (last < text.length) nodes.push(text.slice(last));
  return <>{nodes}</>;
}
