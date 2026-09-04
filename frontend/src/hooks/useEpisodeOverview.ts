/**
 * Everything the episode Overview hub reads, in one place.
 *
 * Five queries used to sit inside `OverviewTab` between its mutations and
 * its layout, so a 568-line component body mixed data wiring with markup and
 * every layout change scrolled past the fetch rules. The hub now reads as
 * layout over this hook's result; the pure grouping it applies to
 * `allVersions` lives in `lib/versionGroups.ts`.
 */

import { useCallback } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { Episode, EpisodeSpeakersResponse, Segment, VersionEntry } from "@/api/types";
import { getEpisodeSpeakers } from "@/api/client";
import { getAllVersions, getEpisodeCollections } from "@/api/search";
import { getSegmentsPreview as getTranscribePreview, getSpeakerMap } from "@/api/transcribe";
import { getCorrectSegmentsPreview as getCorrectPreview } from "@/api/correct";
import { invalidateAfterStep } from "@/api/cacheInvalidation";
import { queryKeys } from "@/api/queryKeys";
import { getEpisodeSourceRef } from "@/lib/episodeRef";

/** Rows shown in the transcript preview card. */
const PREVIEW_LIMIT = 5;

export interface EpisodeOverviewData {
  speakerMap: Record<string, string> | undefined;
  episodeSpeakers: EpisodeSpeakersResponse | undefined;
  previewSegments: Segment[] | undefined;
  /** Which step the preview came from, so the card can label it. */
  previewStep: "transcribe" | "correct";
  allVersions: VersionEntry[] | undefined;
  indexEntries: Awaited<ReturnType<typeof getEpisodeCollections>> | undefined;
  /** Sweep every namespace for this episode. A version or collection delete
   *  can touch any step's output, so this is the documented fallback,
   *  narrowed to one episode. */
  invalidateAll: () => void;
}

export function useEpisodeOverview(
  episode: Episode,
  folder: string | undefined,
  showName: string,
): EpisodeOverviewData {
  const queryClient = useQueryClient();
  const { audioPath, outputDir, sourceRef, hasSourceRef } = getEpisodeSourceRef(episode);
  const hasTranscript = !!episode.transcribed;

  const { data: speakerMap } = useQuery({
    queryKey: queryKeys.speakerMap(sourceRef),
    queryFn: () => getSpeakerMap(audioPath, outputDir ?? undefined),
    enabled: hasSourceRef && hasTranscript,
  });

  // Speakers of the canonical transcript with per-speaker airtime share.
  const { data: episodeSpeakers } = useQuery({
    queryKey: queryKeys.episodeSpeakers(folder ?? "", episode.stem ?? ""),
    queryFn: () => getEpisodeSpeakers(folder!, episode.stem!),
    enabled: !!folder && !!episode.stem && hasTranscript,
  });

  const previewStep = episode.corrected ? "correct" : "transcribe";
  const { data: previewSegments } = useQuery({
    queryKey: [...queryKeys.stepSegments(previewStep, sourceRef), "preview"],
    queryFn: () =>
      previewStep === "correct"
        ? getCorrectPreview(audioPath, PREVIEW_LIMIT, outputDir ?? undefined)
        : getTranscribePreview(audioPath, PREVIEW_LIMIT, outputDir ?? undefined),
    enabled: hasSourceRef && hasTranscript,
  });

  const { data: allVersions } = useQuery({
    queryKey: queryKeys.allVersions(sourceRef),
    queryFn: () => getAllVersions(audioPath, outputDir),
    enabled: hasSourceRef && hasTranscript,
  });

  const { data: indexEntries } = useQuery({
    queryKey: queryKeys.episodeCollections(sourceRef, showName),
    queryFn: () => getEpisodeCollections(audioPath, showName, outputDir),
    enabled: hasSourceRef && !!showName && !!episode.indexed,
  });

  const invalidateAll = useCallback(() => {
    invalidateAfterStep(queryClient, null, { audioPath: sourceRef });
  }, [sourceRef, queryClient]);

  return {
    speakerMap,
    episodeSpeakers,
    previewSegments,
    previewStep,
    allVersions,
    indexEntries,
    invalidateAll,
  };
}
