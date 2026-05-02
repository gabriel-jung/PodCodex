import type { TaskProgress } from "@/hooks/useProgress";
import type { BatchEpisode } from "@/stores/taskStore";

export type EpStatus = "pending" | "running" | "done" | "failed";

interface EpisodeStatus {
  title: string;
  stem: string;
  status: EpStatus;
  error?: string;
}

/**
 * Parse the backend's "[i/total] ..." progress prefix. Both fields are null
 * when the message is empty or the prefix is missing.
 */
export function parseProgressCount(
  message: string | null | undefined,
): { current: number | null; total: number | null } {
  const match = (message ?? "").match(/^\[(\d+)\/(\d+)\]/);
  if (!match) return { current: null, total: null };
  return { current: Number(match[1]), total: Number(match[2]) };
}

/** Single-pass count of episodes by status. */
export function countByStatus(
  statuses: EpisodeStatus[],
): Record<EpStatus, number> {
  const out: Record<EpStatus, number> = { pending: 0, running: 0, done: 0, failed: 0 };
  for (const s of statuses) out[s.status]++;
  return out;
}

/**
 * Derive per-episode status from the batch progress message format `[X/N] ...`.
 */
export function deriveEpisodeStatuses(
  episodes: BatchEpisode[],
  progress: TaskProgress | null,
): EpisodeStatus[] {
  if (!episodes.length) return [];
  if (!progress) return episodes.map((ep) => ({ ...ep, status: "pending" as const }));

  const { current } = parseProgressCount(progress.message);
  const currentIdx = current != null ? current - 1 : 0;
  const isFinished = ["completed", "failed", "cancelled"].includes(progress.status);

  // Check result for detailed per-episode info
  const result = progress.result;
  const errors: { episode: string; error: string }[] =
    result && typeof result === "object" && Array.isArray((result as Record<string, unknown>).errors)
      ? (result as { errors: { episode: string; error: string }[] }).errors
      : [];
  const errorByEpisode = new Map(errors.map((e) => [e.episode, e.error]));

  return episodes.map((ep, i) => {
    const error = errorByEpisode.get(ep.title) ?? errorByEpisode.get(ep.stem);
    if (isFinished) {
      if (error) return { ...ep, status: "failed", error };
      if (i <= currentIdx || progress.status === "completed") return { ...ep, status: "done" };
      return { ...ep, status: "pending" };
    }
    if (i < currentIdx) return { ...ep, status: "done" };
    if (i === currentIdx) return { ...ep, status: "running" };
    return { ...ep, status: "pending" };
  });
}
