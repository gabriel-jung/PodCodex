import type { ShowSummary } from "@/api/types";

/** Noun agreeing with `n`: "episode" / "episodes". */
export function pluralize(n: number, noun: string): string {
  return n === 1 ? noun : `${noun}s`;
}

/** "3 episodes", "1 model". The one place count labels are formatted. */
export function countLabel(n: number, noun: string): string {
  return `${n} ${pluralize(n, noun)}`;
}

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
    const head = countLabel(total, "episode");
    return downloaded > 0 ? `${head} · ${downloaded} downloaded` : head;
  }
  if (downloaded > 0) {
    return countLabel(downloaded, "episode");
  }
  return null;
}
