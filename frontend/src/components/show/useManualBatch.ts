/** State and actions of the batch dialog's manual LLM flow: for each
 *  episode, fetch prompts, collect the pasted replies batch by batch, and
 *  apply them once every batch of the episode is validated. */

import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { Episode, ShowMeta } from "@/api/types";
import { queryKeys } from "@/api/queryKeys";
import { invalidateAfterStep } from "@/api/cacheInvalidation";
import { getCorrectManualPrompts, applyCorrectManual } from "@/api/correct";
import { getTranslateManualPrompts, applyTranslateManual } from "@/api/translate";
import { buildDefaultContext, errorMessage } from "@/lib/utils";
import { promptToInputEntries, reconcileBatch } from "@/lib/reconcile";

export interface PromptBatch {
  batch_index: number;
  prompt: string;
  segment_count: number;
}

/** Stable key for an episode across the batch dialog. */
export function epKey(ep: Episode): string {
  return ep.audio_path || ep.id;
}

/** Validated replies of one episode, for the batch count they were
 *  validated against (a new count means new prompts, so they are dropped). */
interface EpisodeReplies {
  count: number;
  batches: Record<number, Record<string, unknown>[]>;
}

function isComplete(r: EpisodeReplies | undefined): boolean {
  if (!r || r.count === 0) return false;
  for (let i = 0; i < r.count; i++) if (r.batches[i] == null) return false;
  return true;
}

export function useManualBatch({
  step,
  episodes,
  folder,
  showMeta,
  sourceLang,
  targetLang,
  sourceVersions,
}: {
  step: "correct" | "translate";
  /** Episodes the flow walks through, in order. */
  episodes: Episode[];
  folder: string;
  showMeta: ShowMeta | null | undefined;
  sourceLang: string;
  targetLang: string;
  /** Pinned source version per `epKey`; prompts and apply must agree on it. */
  sourceVersions: Record<string, string>;
}) {
  const queryClient = useQueryClient();
  const [active, setActive] = useState(false);
  const [epIndex, setEpIndex] = useState(0);
  const [batchIndex, setBatchIndex] = useState(0);
  const [batchCounts, setBatchCounts] = useState<Record<string, number>>({});
  const [replies, setReplies] = useState<Record<string, EpisodeReplies>>({});
  const [pasted, setPastedRaw] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [applyingEp, setApplyingEp] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);
  const [applied, setApplied] = useState<Set<string>>(() => new Set());
  const [error, setError] = useState<string | null>(null);

  const ep = active ? episodes[epIndex] : undefined;
  const ek = ep ? epKey(ep) : "";
  const batchCount = batchCounts[ek] ?? 1;

  // The batch slider moves through values quickly; fetch for where it stops.
  const [fetchCount, setFetchCount] = useState(batchCount);
  useEffect(() => {
    const t = setTimeout(() => setFetchCount(batchCount), 300);
    return () => clearTimeout(t);
  }, [batchCount]);

  const versionId = sourceVersions[ek] ?? null;
  // Per episode, like the episode panels: the working LLM context holds
  // whichever episode was opened last.
  const context = ep ? buildDefaultContext(ep, showMeta) : "";
  const hasSource = !!ep && (!!ep.audio_path || !!ep.output_dir);

  const promptsQuery = useQuery({
    queryKey: queryKeys.batchManualPrompts(
      step, ek, fetchCount, versionId, sourceLang, step === "translate" ? targetLang : null, context,
    ),
    queryFn: async (): Promise<PromptBatch[]> => {
      const params = {
        audio_path: ep!.audio_path || undefined,
        output_dir: ep!.output_dir || undefined,
        source_lang: sourceLang,
        // Exactly the slider's count, whatever the episode's duration
        // (unknown durations used to guess an hour and split by minutes).
        batch_count: fetchCount,
        context: context || undefined,
        source_version_id: versionId ?? undefined,
      };
      return step === "translate"
        ? getTranslateManualPrompts({ ...params, target_lang: targetLang })
        : getCorrectManualPrompts(params);
    },
    enabled: active && hasSource && fetchCount === batchCount,
    staleTime: Infinity,
    gcTime: 0,
    retry: false,
  });
  const prompts = promptsQuery.data ?? [];
  const epReplies = replies[ek];
  // Replies validated against another batch count belong to other prompts.
  const currentBatches = epReplies && epReplies.count === prompts.length ? epReplies.batches : {};

  const resetPaste = () => {
    setPastedRaw("");
    setParseError(null);
    setCopied(false);
  };

  const applyEpisode = async (target: Episode, r: EpisodeReplies) => {
    const corrections: Record<string, unknown>[] = [];
    for (let i = 0; i < r.count; i++) corrections.push(...(r.batches[i] ?? []));
    const params = {
      audio_path: target.audio_path || undefined,
      output_dir: target.output_dir || undefined,
      corrections,
      // The same version the prompts were built from: the route validates the
      // replies against it and rejects a count mismatch.
      source_version_id: sourceVersions[epKey(target)],
    };
    if (step === "translate") await applyTranslateManual({ ...params, lang: targetLang });
    else await applyCorrectManual(params);
    // The same sweep as an auto run: versions, editors, best source,
    // speaker views and the show-wide versions list.
    invalidateAfterStep(queryClient, step, { folder });
    setApplied((prev) => new Set(prev).add(epKey(target)));
  };

  // Complete but not applied, and not being applied right now: offering a
  // retry for the in-flight episode would save its replies twice.
  const pendingApply = episodes.filter(
    (e) => isComplete(replies[epKey(e)]) && !applied.has(epKey(e)) && epKey(e) !== applyingEp,
  );

  return {
    active,
    start: () => {
      setActive(true);
      setEpIndex(0);
      setBatchIndex(0);
    },
    /** Leave the flow, dropping prompts and unapplied replies. */
    back: () => {
      setActive(false);
      setReplies({});
      setEpIndex(0);
      setBatchIndex(0);
      resetPaste();
    },
    /** A pasted reply, or validated batches not yet applied. */
    dirty:
      active &&
      (pasted.trim() !== "" ||
        Object.entries(replies).some(
          ([k, r]) => !applied.has(k) && Object.keys(r.batches).length > 0,
        )),
    applied,
    allApplied: episodes.length > 0 && episodes.every((e) => applied.has(epKey(e))),

    ep,
    ek,
    epIndex,
    episodeCount: episodes.length,
    goToEpisode: (i: number) => {
      setEpIndex(Math.max(0, Math.min(episodes.length - 1, i)));
      setBatchIndex(0);
      resetPaste();
    },
    batchCount,
    setBatchCount: (n: number) => {
      setBatchCounts((prev) => ({ ...prev, [ek]: n }));
      setBatchIndex(0);
    },

    prompts,
    promptsLoading: promptsQuery.isFetching || (hasSource && fetchCount !== batchCount),
    promptsError: promptsQuery.isError ? errorMessage(promptsQuery.error) : null,
    hasSource,
    batchIndex,
    goToBatch: (i: number) => {
      setBatchIndex(i);
      resetPaste();
    },
    batches: currentBatches,

    pasted,
    setPasted: (v: string) => {
      setPastedRaw(v);
      setParseError(null);
    },
    parseError,
    copied,
    copy: async (text: string) => {
      try {
        await navigator.clipboard.writeText(text);
      } catch (e) {
        // Denied or unfocused: say so, or the user pastes an old clipboard.
        setError(`Could not copy the prompt: ${errorMessage(e)}`);
        return;
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    },

    /** Parse the pasted reply into the current batch; once the episode's
     *  last batch is in, apply it and move to the next unfinished episode. */
    validate: async () => {
      if (!ep) return;
      setParseError(null);
      // Same check as the episode panels: parse (bare strings become
      // `{text}`, the shape the apply route takes) and match the batch's
      // entries, so a short reply is caught here, not as an apply that
      // fails on every retry.
      const batch = prompts[batchIndex];
      if (!batch) return;
      const checked = reconcileBatch(pasted, promptToInputEntries(batch.prompt));
      if ("error" in checked) {
        setParseError(checked.error);
        return;
      }
      const arr = checked.objs;
      const next: EpisodeReplies = {
        count: prompts.length,
        batches: { ...currentBatches, [batchIndex]: arr },
      };
      setReplies((prev) => ({ ...prev, [ek]: next }));
      setPastedRaw("");
      if (!isComplete(next)) {
        const nextBatch = prompts.findIndex((_, i) => i > batchIndex && next.batches[i] == null);
        if (nextBatch >= 0) setBatchIndex(nextBatch);
        return;
      }
      setApplyingEp(ek);
      setError(null);
      try {
        await applyEpisode(ep, next);
        const nextEp = episodes.findIndex(
          (e, i) => i > epIndex && !applied.has(epKey(e)) && !isComplete(replies[epKey(e)]),
        );
        if (nextEp >= 0) {
          setEpIndex(nextEp);
          setBatchIndex(0);
          setCopied(false);
        }
      } catch (e) {
        setError(errorMessage(e));
      } finally {
        setApplyingEp(null);
      }
    },
    /** Drop the current batch's validated reply so it can be pasted again. */
    redo: () => {
      const batches = { ...currentBatches };
      delete batches[batchIndex];
      setReplies((prev) => ({ ...prev, [ek]: { count: prompts.length, batches } }));
    },
    applyingEp,
    error,

    /** Episodes whose replies are complete but whose apply failed. */
    pendingApply,
    retrying,
    retry: async () => {
      setRetrying(true);
      setError(null);
      try {
        for (const e of pendingApply) await applyEpisode(e, replies[epKey(e)]);
      } catch (e) {
        setError(errorMessage(e));
      } finally {
        setRetrying(false);
      }
    },
  };
}

export type ManualBatch = ReturnType<typeof useManualBatch>;
