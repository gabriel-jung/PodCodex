/**
 * Poll live pipeline state while a download or batch is running.
 *
 * `/unified` is the heavy endpoint (feed cache parse + per-episode metadata
 * reads), so polling it every 5s to watch a flag flip is wasteful. `/status`
 * returns only the mutable half of an episode, keyed by stem, and this hook
 * merges it into the cached `/unified` list in place. Because
 * `UnifiedEpisodeOut` extends `EpisodeStatusOut` on the Python side, the merge
 * is a plain spread and stays correct when a status field is added.
 *
 * Episodes whose status is unchanged keep their object identity, so a poll
 * tick only re-renders the rows that actually moved.
 */
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { getEpisodeStatus } from "@/api/shows";
import { queryKeys } from "@/api/queryKeys";
import type { Episode, EpisodeStatus, PipelineDefaults } from "@/api/types";

const POLL_INTERVAL_MS = 5000;

/** True when `ep` already reflects every field of `status`. */
function sameStatus(ep: Episode, status: EpisodeStatus): boolean {
  for (const key of Object.keys(status) as (keyof EpisodeStatus)[]) {
    const before: unknown = ep[key];
    const after: unknown = status[key];
    if (before === after) continue;
    if (Array.isArray(before) && Array.isArray(after)) {
      if (before.length === after.length && before.every((v, i) => v === after[i])) continue;
      return false;
    }
    // `provenance` and `verified` are the only nested values; both are small.
    if (before && after && typeof before === "object" && typeof after === "object") {
      if (JSON.stringify(before) === JSON.stringify(after)) continue;
      return false;
    }
    return false;
  }
  return true;
}

export function useEpisodeStatusPoll(
  folder: string | undefined,
  pipelineDefaults: PipelineDefaults,
  enabled: boolean,
  /** `dataUpdatedAt` of the caller's episodes query. Only a dependency: the
   *  merge bails when the full list has not arrived yet, and the cheap poll
   *  routinely resolves first, so without this the first tick after opening a
   *  page during a run is dropped and live flags lag a whole interval. */
  episodesUpdatedAt = 0,
) {
  const queryClient = useQueryClient();
  const { data: statuses, dataUpdatedAt } = useQuery({
    queryKey: queryKeys.episodeStatus(folder ?? "", pipelineDefaults),
    queryFn: () => getEpisodeStatus(folder!, pipelineDefaults),
    enabled: !!folder && enabled,
    refetchInterval: POLL_INTERVAL_MS,
  });

  // A stem the cached list has never seen means the run created an episode
  // (first download on a local-only show, a subtitle import). Only a full
  // fetch can supply its title and feed metadata, so fall back to one.
  const refetchedForRef = useRef("");

  useEffect(() => {
    if (!folder || !statuses) return;
    const key = queryKeys.episodes(folder, pipelineDefaults);
    const episodes = queryClient.getQueryData<Episode[]>(key);
    if (!episodes) return;
    // Polling stops and restarts across runs, so this query can hand back a
    // cached result older than the full list (which TaskBar refetches on
    // completion). Merging that would roll flags backwards for one tick.
    // Read fresh from the cache rather than using the prop: the prop is a
    // render-stale snapshot and exists only to re-run this effect.
    const cachedEpisodesUpdatedAt = queryClient.getQueryState(key)?.dataUpdatedAt ?? 0;
    if (dataUpdatedAt < cachedEpisodesUpdatedAt) return;

    const byStem = new Map<string, EpisodeStatus>();
    for (const s of statuses) if (s.stem) byStem.set(s.stem, s);

    const known = new Set<string>();
    let changed = false;
    const merged = episodes.map((ep) => {
      if (ep.stem) known.add(ep.stem);
      const status = ep.stem ? byStem.get(ep.stem) : undefined;
      if (!status || sameStatus(ep, status)) return ep;
      changed = true;
      return { ...ep, ...status };
    });
    if (changed) queryClient.setQueryData(key, merged);

    const stems = [...byStem.keys()];
    const signature = [...stems].sort().join(" ");
    if (stems.some((stem) => !known.has(stem)) && refetchedForRef.current !== signature) {
      refetchedForRef.current = signature;
      queryClient.invalidateQueries({ queryKey: key });
    }
  }, [statuses, dataUpdatedAt, episodesUpdatedAt, folder, pipelineDefaults, queryClient]);
}
