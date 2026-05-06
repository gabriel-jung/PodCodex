import type { ShowSummary } from "@/api/types";

/**
 * Format episode counts for a show card / row.
 *   - Has a feed: "248 episodes · 12 downloaded" (drops "· 0 downloaded")
 *   - Local only: "12 episodes"
 *   - Empty:      null
 */
export function showEpisodeCountLabel(show: ShowSummary): string | null {
  const downloaded = show.episode_count ?? 0;
  const total = show.feed_episode_count ?? null;

  if (total != null) {
    const head = `${total} episode${total !== 1 ? "s" : ""}`;
    return downloaded > 0 ? `${head} · ${downloaded} downloaded` : head;
  }
  if (downloaded > 0) {
    return `${downloaded} episode${downloaded !== 1 ? "s" : ""}`;
  }
  return null;
}
