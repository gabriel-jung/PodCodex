/**
 * Single facility for feed refreshes (RSS + YouTube).
 *
 * Both hooks share `mutationKeys.feedRefresh()` so `useIsMutating` reports
 * activity across pages: a refresh-all started on the home page keeps the
 * show page's button spinning, and vice versa. Cache invalidation is
 * declared via `meta.invalidates` (handled by the MutationCache default in
 * main.tsx), so it still runs when the user navigates away mid-refresh.
 */
import { useState } from "react";
import { useIsMutating, useMutation } from "@tanstack/react-query";
import { refreshRSS, refreshYouTube } from "@/api/client";
import { mutationKeys, queryKeys } from "@/api/queryKeys";
import type { ShowSummary } from "@/api/types";
import { useMutationErrorStore } from "@/stores/mutationErrorStore";
import { errorMessage } from "@/lib/utils";

/** True while any feed refresh (single-show or refresh-all) is in flight. */
export function useFeedRefreshing(): boolean {
  return useIsMutating({ mutationKey: mutationKeys.feedRefresh() }) > 0;
}

/** Refresh a single show's feed (RSS or YouTube). */
export function useFeedRefresh(folder: string, isYouTube: boolean) {
  const mutation = useMutation({
    mutationKey: mutationKeys.feedRefresh(),
    // shows() included: the home page reads last_rss_update from the shows list.
    meta: {
      invalidates: [
        queryKeys.episodesForFolder(folder),
        queryKeys.showMeta(folder),
        queryKeys.shows(),
      ],
    },
    mutationFn: async () => {
      if (isYouTube) await refreshYouTube(folder);
      else await refreshRSS(folder);
    },
  });
  return { mutation };
}

/** Refresh every feed show (home page "update feeds"), tracking progress. */
export function useFeedRefreshAll(rssShows: ShowSummary[], ytShows: ShowSummary[]) {
  // done/total snapshotted together at mutate() time: the live show arrays
  // can grow or shrink mid-run (shows() invalidation), and a live total
  // would make the count read 7/9 forever, or overshoot.
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const mutation = useMutation({
    mutationKey: mutationKeys.feedRefresh(),
    meta: {
      invalidates: [queryKeys.shows(), queryKeys.showMetaAll(), queryKeys.episodesAll()],
    },
    mutationFn: async () => {
      const targets: Array<{ name: string; run: () => Promise<unknown> }> = [
        ...rssShows.map((s) => ({ name: s.name || s.path, run: () => refreshRSS(s.path) })),
        ...ytShows.map((s) => ({ name: s.name || s.path, run: () => refreshYouTube(s.path) })),
      ];
      setProgress({ done: 0, total: targets.length });
      const tick = <T,>(p: Promise<T>) =>
        p.finally(() => setProgress((prev) => prev && { ...prev, done: prev.done + 1 }));
      const outcomes = await Promise.allSettled(targets.map((t) => tick(t.run())));
      // Resolve anyway, so the shows that did refresh are invalidated (meta
      // invalidates run on success only), and report the ones that did not.
      const failed = outcomes.flatMap((o, i) =>
        o.status === "rejected" ? [`${targets[i].name} (${errorMessage(o.reason)})`] : [],
      );
      if (failed.length > 0) {
        useMutationErrorStore.getState().report(
          new Error(`Could not refresh ${failed.length} of ${targets.length} feeds: ${failed.join("; ")}`),
        );
      }
    },
  });
  // Progress label only while OUR mutation runs; a refresh started elsewhere
  // has no count to show, so consumers fall back to the button's default.
  const refreshingLabel = mutation.isPending && progress
    ? `Refreshing ${progress.done}/${progress.total}`
    : undefined;
  return { mutation, refreshingLabel };
}
