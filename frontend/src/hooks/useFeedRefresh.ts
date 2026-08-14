/**
 * Single facility for feed refreshes (RSS + YouTube).
 *
 * Both hooks share `mutationKeys.feedRefresh()` so `useIsMutating` reports
 * activity across pages: a refresh-all started on the home page keeps the
 * show page's button spinning, and vice versa. Cache invalidation lives
 * inside `mutationFn`, not `onSuccess`: component-level callbacks are
 * skipped once the owning page unmounts, and users routinely navigate
 * while feeds refresh.
 */
import { useState } from "react";
import { useIsMutating, useMutation, useQueryClient } from "@tanstack/react-query";
import { refreshRSS, refreshYouTube } from "@/api/client";
import { mutationKeys, queryKeys } from "@/api/queryKeys";
import type { ShowSummary } from "@/api/types";

/** True while any feed refresh (single-show or refresh-all) is in flight. */
export function useFeedRefreshing(): boolean {
  return useIsMutating({ mutationKey: mutationKeys.feedRefresh() }) > 0;
}

/** Refresh a single show's feed (RSS or YouTube). */
export function useFeedRefresh(folder: string, isYouTube: boolean) {
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationKey: mutationKeys.feedRefresh(),
    mutationFn: async () => {
      if (isYouTube) await refreshYouTube(folder);
      else await refreshRSS(folder);
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesForFolder(folder) });
      queryClient.invalidateQueries({ queryKey: queryKeys.showMeta(folder) });
      // Home page reads last_rss_update per show from the shows list.
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
    },
  });
  return { mutation };
}

/** Refresh every feed show (home page "update feeds"), tracking progress. */
export function useFeedRefreshAll(rssShows: ShowSummary[], ytShows: ShowSummary[]) {
  const queryClient = useQueryClient();
  // done/total snapshotted together at mutate() time: the live show arrays
  // can grow or shrink mid-run (shows() invalidation), and a live total
  // would make the count read 7/9 forever, or overshoot.
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const mutation = useMutation({
    mutationKey: mutationKeys.feedRefresh(),
    mutationFn: async () => {
      const targets: Array<() => Promise<unknown>> = [
        ...rssShows.map((s) => () => refreshRSS(s.path)),
        ...ytShows.map((s) => () => refreshYouTube(s.path)),
      ];
      setProgress({ done: 0, total: targets.length });
      const tick = <T,>(p: Promise<T>) =>
        p.finally(() => setProgress((prev) => prev && { ...prev, done: prev.done + 1 }));
      await Promise.allSettled(targets.map((run) => tick(run())));
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
      queryClient.invalidateQueries({ queryKey: queryKeys.showMetaAll() });
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
    },
  });
  // Progress label only while OUR mutation runs; a refresh started elsewhere
  // has no count to show, so consumers fall back to the button's default.
  const refreshingLabel = mutation.isPending && progress
    ? `Refreshing ${progress.done}/${progress.total}`
    : undefined;
  return { mutation, refreshingLabel };
}
