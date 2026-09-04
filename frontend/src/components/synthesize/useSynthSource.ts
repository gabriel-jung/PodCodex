/**
 * The synthesis source: which version the cloned voices read, and its
 * segments.
 *
 * Lifted out of `SourceSegmentPicker`, which used to own both queries and
 * copy their results up into `SynthesizePanel` state through two effects.
 * The panel then rendered one frame behind the query on every version
 * switch, and the state had no owner that could be reasoned about locally —
 * which is what made the surrounding fingerprint and stamp bookkeeping
 * necessary. The panel owns the data now and passes it down.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Episode, Segment, VersionEntry } from "@/api/types";
import {
  getAllVersions,
  getCorrectSegments,
  getSegments,
  getTranslateSegments,
  loadCorrectVersion,
  loadTranscribeVersion,
  loadTranslateVersion,
} from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { sourceRefFor } from "@/lib/episodeRef";
import { isEdited } from "@/lib/utils";

/** Module-level stable reference, so a consumer's memo dependency does not
 *  see a fresh `[]` literal on every render while the query is loading. */
const EMPTY_SEGMENTS: Segment[] = [];

/** Resolved source info: which step the selected version belongs to, plus
 *  the payload fields the generate endpoint needs. */
export interface ResolvedSource {
  step: "transcript" | "corrected" | "translate";
  lang: string;
  sourceLang: string | undefined;
  sourceVersionId: string | null;
}

export interface SynthSource {
  /** Versions offering usable source text, in pick order. */
  inputVersions: VersionEntry[];
  selectedVersion: VersionEntry | null;
  resolved: ResolvedSource;
  segments: Segment[];
  /** Per-step editor key, shared with the Transcribe/Correct/Translate
   *  panels so React Query dedupes their segment fetches with this one. */
  editorKey: string;
  versions: {
    isError: boolean;
    error: unknown;
    refetch: () => void;
  };
  segmentsQuery: {
    isError: boolean;
    error: unknown;
    /** True until the segments query has produced a result. `segments` is
     *  `[]` in both states, so a version that genuinely has none would
     *  otherwise read as "still loading" forever. */
    isPending: boolean;
    refetch: () => void;
  };
}

export function useSynthSource(
  /** Nullable: the panel calls this above its own `if (!episode) return null`
   *  guard, and the store's `episode` is runtime-only — it is null on the
   *  render before `EpisodePage`'s effect sets it, and again whenever the
   *  requested stem is not in the show. */
  episode: Episode | null | undefined,
  audioPath: string | null,
  outputDir: string | null | undefined,
  sourceVersionId: string | null,
): SynthSource {
  const ref = sourceRefFor(audioPath, outputDir);
  const od = outputDir ?? undefined;

  const versionsQuery = useQuery({
    queryKey: queryKeys.allVersions(ref),
    queryFn: () => getAllVersions(audioPath, outputDir),
    enabled: !!ref,
  });
  const allVersions = versionsQuery.data;

  const translationSet = useMemo(
    () => new Set(episode?.translations ?? []),
    [episode?.translations],
  );

  const inputVersions = useMemo<VersionEntry[]>(() => {
    if (!allVersions) return [];
    // Ordering: translation > corrected > transcript; edited before raw
    // within each step; newest first within each (step, edited) bucket.
    const stepRank = (step: string | undefined): number => {
      if (!step) return 3;
      if (step === "transcript") return 2;
      if (step === "corrected") return 1;
      return 0;
    };
    return allVersions
      .filter((v) => {
        if (!v.step) return false;
        if (v.step === "transcript" || v.step === "corrected") return true;
        return translationSet.has(v.step);
      })
      .sort((a, b) => {
        const sr = stepRank(a.step) - stepRank(b.step);
        if (sr !== 0) return sr;
        const er = (isEdited(a) ? 0 : 1) - (isEdited(b) ? 0 : 1);
        if (er !== 0) return er;
        return b.timestamp.localeCompare(a.timestamp);
      });
  }, [allVersions, translationSet]);

  const selectedVersion = useMemo<VersionEntry | null>(() => {
    if (!inputVersions.length) return null;
    if (sourceVersionId) {
      return inputVersions.find((v) => v.id === sourceVersionId) ?? inputVersions[0];
    }
    return inputVersions[0];
  }, [inputVersions, sourceVersionId]);

  const resolved = useMemo<ResolvedSource>(() => {
    if (!selectedVersion) {
      return { step: "transcript", lang: "", sourceLang: undefined, sourceVersionId: null };
    }
    const step = selectedVersion.step ?? "transcript";
    if (step === "transcript" || step === "corrected") {
      return { step, lang: "", sourceLang: undefined, sourceVersionId };
    }
    // Translation versions store the language as the step name.
    return { step: "translate", lang: step, sourceLang: step, sourceVersionId };
  }, [selectedVersion, sourceVersionId]);

  const editorKey =
    resolved.step === "transcript"
      ? "transcribe"
      : resolved.step === "corrected"
        ? "correct"
        : `translate-${resolved.lang}`;

  // Segment cache is shared with the editors: stepVersionSegments for a
  // pinned version, stepSegments for "latest".
  const segmentsQuery = useQuery({
    queryKey: sourceVersionId
      ? queryKeys.stepVersionSegments(editorKey, ref, sourceVersionId)
      : queryKeys.stepSegments(editorKey, ref),
    queryFn: async (): Promise<Segment[]> => {
      // Load the pinned version when provided; otherwise the step's latest.
      if (sourceVersionId) {
        if (resolved.step === "transcript")
          return loadTranscribeVersion(audioPath, sourceVersionId, od);
        if (resolved.step === "corrected")
          return loadCorrectVersion(audioPath, sourceVersionId, od);
        return loadTranslateVersion(audioPath, resolved.lang, sourceVersionId, od);
      }
      if (resolved.step === "transcript") return getSegments(audioPath, od);
      if (resolved.step === "corrected") return getCorrectSegments(audioPath, od);
      return getTranslateSegments(audioPath, resolved.lang, od);
    },
    enabled: !!selectedVersion,
  });

  return {
    inputVersions,
    selectedVersion,
    resolved,
    segments: segmentsQuery.data ?? EMPTY_SEGMENTS,
    editorKey,
    versions: {
      isError: versionsQuery.isError,
      error: versionsQuery.error,
      refetch: () => void versionsQuery.refetch(),
    },
    segmentsQuery: {
      isError: segmentsQuery.isError,
      error: segmentsQuery.error,
      isPending: !segmentsQuery.isSuccess && !segmentsQuery.isError,
      refetch: () => void segmentsQuery.refetch(),
    },
  };
}
