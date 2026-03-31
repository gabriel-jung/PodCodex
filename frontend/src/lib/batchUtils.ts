import type { TaskProgress } from "@/hooks/useProgress";

export type EpStatus = "pending" | "running" | "done" | "failed";

/**
 * Derive per-episode status from the batch progress message format `[X/N] ...`.
 */
export function deriveEpisodeStatuses(
  names: string[],
  progress: TaskProgress | null,
): { name: string; status: EpStatus }[] {
  if (!names.length) return [];
  if (!progress) return names.map((name) => ({ name, status: "pending" as const }));

  const msg = progress.message || "";
  // Parse "[3/17] Polishing..." → current episode index is 3 (1-based)
  const match = msg.match(/^\[(\d+)\/(\d+)\]/);
  const currentIdx = match ? parseInt(match[1], 10) - 1 : 0;
  const isFinished = ["completed", "failed", "cancelled"].includes(progress.status);

  // Check result for detailed per-episode info
  const result = progress.result as { errors?: { episode: string }[] } | undefined;
  const failedEpisodes = new Set((result?.errors ?? []).map((e) => e.episode));

  return names.map((name, i) => {
    if (isFinished) {
      if (failedEpisodes.has(name)) return { name, status: "failed" };
      if (i <= currentIdx || progress.status === "completed") return { name, status: "done" };
      return { name, status: "pending" };
    }
    if (i < currentIdx) return { name, status: "done" };
    if (i === currentIdx) return { name, status: "running" };
    return { name, status: "pending" };
  });
}
