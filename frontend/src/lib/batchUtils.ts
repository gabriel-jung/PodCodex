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
 * Derive per-episode status from the batch progress message format `[X/N] ...`.
 */
export function deriveEpisodeStatuses(
  episodes: BatchEpisode[],
  progress: TaskProgress | null,
): EpisodeStatus[] {
  if (!episodes.length) return [];
  if (!progress) return episodes.map((ep) => ({ ...ep, status: "pending" as const }));

  const msg = progress.message || "";
  // Parse "[3/17] Correcting..." → current episode index is 3 (1-based)
  const match = msg.match(/^\[(\d+)\/(\d+)\]/);
  const currentIdx = match ? parseInt(match[1], 10) - 1 : 0;
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
