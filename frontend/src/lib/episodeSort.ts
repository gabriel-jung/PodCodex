import type { Episode } from "@/api/types";

/** Publication time in ms, or null when the episode carries no parseable date. */
const pubTime = (e: Episode): number | null => {
  const t = e.pub_date ? new Date(e.pub_date).getTime() : NaN;
  return isNaN(t) ? null : t;
};

/** Compare two episodes by publication date, falling back to feed_order
 *  (source-feed position, 0 = newest) when a date is missing, important for
 *  YouTube where flat extraction often omits upload dates. Undated episodes
 *  sort last regardless of direction. `dir` is 1 for ascending, -1 for
 *  descending. */
export const dateCmp = (a: Episode, b: Episode, dir: 1 | -1): number => {
  const ta = pubTime(a),
    tb = pubTime(b);
  if (ta != null && tb != null) {
    if (ta !== tb) return (ta - tb) * dir;
  } else if (ta != null) {
    return -1; // undated sorts last regardless of direction
  } else if (tb != null) {
    return 1;
  }
  const oa = a.feed_order ?? Number.POSITIVE_INFINITY;
  const ob = b.feed_order ?? Number.POSITIVE_INFINITY;
  return dir === -1 ? oa - ob : ob - oa;
};

/** Default episode order (newest first). Used for prev/next navigation so the
 *  arrows walk episodes in the same order the show list shows by default. */
export const byDefaultOrder = (a: Episode, b: Episode): number => dateCmp(a, b, -1);
