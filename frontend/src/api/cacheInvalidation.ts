import type { QueryClient } from "@tanstack/react-query";
import { queryKeys } from "@/api/queryKeys";
import { isUnderPath } from "@/lib/utils";

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

/**
 * Remove every cached query whose key references `path` either exactly or
 * as a parent of a slash-separated value (e.g. a show folder and the audio
 * paths inside it). Tighter than substring `includes`, which would also
 * wipe unrelated keys whenever `path` is a short or common token.
 */
export function removeQueriesUnderPath(qc: QueryClient, path: string): void {
  if (!path) return;
  qc.removeQueries({
    predicate: (q) =>
      q.queryKey.some((k) => typeof k === "string" && isUnderPath(k, path)),
  });
}

/**
 * Remove every cached query keyed by `showName`.
 *
 * For a *rename*, dropping is the only correct move: several namespaces
 * (`search`, `index`, `bot-access`) key on the show's display name, so after
 * a rename those entries reference a name the backend no longer knows.
 * Invalidating them refetches, and every refetch 404s. Removing them lets the
 * components mount fresh queries under the new name instead.
 *
 * Exact segment equality, not the substring/prefix match `removeQueriesUnderPath`
 * uses: show names are free text, so a prefix rule would let one show's rename
 * wipe another's cache.
 */
export function removeQueriesForShowName(qc: QueryClient, showName: string): void {
  if (!showName) return;
  qc.removeQueries({
    predicate: (q) => q.queryKey.some((k) => k === showName),
  });
}

/**
 * Top-level query namespaces a finished pipeline step can change.
 *
 * Deliberately per-step: a transcribe run does not touch the index, and a
 * translate run does not change the canonical transcript the speaker views
 * resolve, so sweeping every namespace after every step (what the batch
 * handler used to do) triggers refetches that can only return what the cache
 * already held.
 *
 * `speakerViews` is separate because only steps that rewrite the *canonical*
 * transcript (transcript / corrected) shift the roster; translations and
 * synthesized audio never do.
 */
const STEP_INVALIDATIONS: Record<
  string,
  { namespaces: readonly string[]; translations?: boolean; speakerViews?: boolean }
> = {
  transcribe: {
    namespaces: ["transcribe", "versions", "speaker-map", "best-source-segments"],
    speakerViews: true,
  },
  correct: {
    namespaces: ["correct", "versions", "best-source-segments"],
    speakerViews: true,
  },
  translate: {
    namespaces: ["versions", "best-source-segments"],
    translations: true,
  },
  synthesize: { namespaces: ["synthesize", "versions"] },
  index: { namespaces: ["index", "search"] },
};

/** Every namespace above, for an unrecognized step. */
const ALL_STEP_NAMESPACES = [
  ...new Set(Object.values(STEP_INVALIDATIONS).flatMap((s) => s.namespaces)),
];

/**
 * Namespaces whose entries are per-episode, so a single-episode run can
 * invalidate one key instead of sweeping every episode's.
 */
const PER_EPISODE_KEY: Record<string, (audioPath: string) => readonly unknown[]> = {
  versions: (p) => queryKeys.allVersions(p),
  "best-source-segments": (p) => queryKeys.bestSourceSegments(p),
  "speaker-map": (p) => queryKeys.speakerMap(p),
};

/**
 * Invalidate what a finished pipeline step actually changed.
 *
 * Shared by the batch task bar (folder-wide) and the per-episode pipeline
 * task hook so the two can't drift on which namespaces a step touches.
 * An unknown step falls back to sweeping everything, so adding a pipeline
 * step without updating the map is stale-free by default, just wasteful.
 *
 * @param step Pipeline step key ("transcribe", "correct", ...).
 * @param folder Show folder, when the caller knows the run was scoped to one.
 * @param audioPath Episode, for a single-episode run. Narrows the per-episode
 *                  namespaces to that episode's keys.
 */
export function invalidateAfterStep(
  qc: QueryClient,
  step: string | null | undefined,
  { folder, audioPath }: { folder?: string | null; audioPath?: string | null } = {},
): void {
  qc.invalidateQueries({
    queryKey: folder ? queryKeys.episodesForFolder(folder) : queryKeys.episodesAll(),
  });

  const plan = step ? STEP_INVALIDATIONS[step] : undefined;
  const namespaces = plan?.namespaces ?? ALL_STEP_NAMESPACES;
  for (const namespace of namespaces) {
    const scoped = audioPath ? PER_EPISODE_KEY[namespace] : undefined;
    qc.invalidateQueries({ queryKey: scoped ? scoped(audioPath!) : [namespace] });
  }

  // Translation editors are keyed per language ("translate-pl"), so they can
  // only be reached by prefix.
  if (!plan || plan.translations) {
    qc.invalidateQueries({
      predicate: (q) =>
        typeof q.queryKey[0] === "string" && q.queryKey[0].startsWith("translate-"),
    });
  }
  if (!plan || plan.speakerViews) invalidateSpeakerViews(qc);
}

/**
 * Cache sweep after an episode is deleted outright.
 *
 * `remove`, not invalidate, for the per-episode queries: the episode is gone,
 * so refetching them would only 404. Everything else goes through
 * `invalidateAfterStep` with no step, which is the documented
 * sweep-every-namespace fallback, and the right semantics here: a deleted
 * episode touches every namespace at once, so enumerating them at the call
 * site would just be a copy of the map that can drift from it.
 *
 * @param folder Show folder the episode belonged to.
 * @param sourceRef The episode's per-episode query key (`getEpisodeSourceRef`).
 */
export function invalidateAfterEpisodeDelete(
  qc: QueryClient,
  { folder, sourceRef }: { folder?: string | null; sourceRef?: string | null } = {},
): void {
  if (sourceRef) removeQueriesUnderPath(qc, sourceRef);
  invalidateAfterStep(qc, null, { folder });
  // invalidateAfterStep narrows to ["episodes", folder] when a folder is
  // given, and React Query matches by prefix, so the global ["episodes"] list
  // is NOT a match. A deleted episode has to leave that one too.
  if (folder) qc.invalidateQueries({ queryKey: queryKeys.episodesAll() });
}
