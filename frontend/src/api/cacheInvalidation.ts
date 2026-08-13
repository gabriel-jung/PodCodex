import type { QueryClient } from "@tanstack/react-query";
import { isUnderPath } from "@/lib/utils";

/**
 * Remove every cached query whose key references `path` either exactly or
 * as a parent of a slash-separated value (e.g. a show folder and the audio
 * paths inside it). Tighter than substring `includes`, which would also
 * wipe unrelated keys whenever `path` is a short or common token.
 */
/**
 * Invalidate every query that renders episode speakers: the show-wide roster
 * (Speakers tab + episode-list column) and the per-episode airtime line
 * (Overview tab). Call after anything that changes the canonical transcript
 * or its labels: transcript/correct runs, editor saves (speaker renames),
 * version deletes, and verified-pointer changes. Broad sweeps by namespace;
 * both queries are cheap to refetch relative to the staleness they cause.
 */
export function invalidateSpeakerViews(qc: QueryClient): void {
  qc.invalidateQueries({ queryKey: ["speakerRoster"] });
  qc.invalidateQueries({ queryKey: ["episodeSpeakers"] });
}

export function removeQueriesUnderPath(qc: QueryClient, path: string): void {
  if (!path) return;
  qc.removeQueries({
    predicate: (q) =>
      q.queryKey.some((k) => typeof k === "string" && isUnderPath(k, path)),
  });
}
