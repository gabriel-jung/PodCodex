/**
 * Version bucketing and transcribe-run pairing for the episode Overview.
 *
 * Pure functions of the version list, extracted from `OverviewTab`, where
 * they sat inside the component between five queries and the layout: the
 * two most fix-heavy rules in the hub could not be exercised without
 * rendering, and every change to the layout scrolled past them.
 */

import type { VersionEntry } from "@/api/types";

/** Parquet by-products a transcribe run writes beside its transcript. */
export const TRANSCRIBE_INTERMEDIATE_STEPS = new Set([
  "segments",
  "diarization",
  "diarized_segments",
  "speaker_map",
]);

/**
 * A diarized run writes its two transcript versions back-to-back (seconds
 * apart). Two transcripts further apart than this belong to separate runs —
 * catches a json/subtitle import, which has no intermediates to split runs.
 */
export const RUN_GAP_MS = 2 * 60 * 1000;

export interface VersionGroups {
  transcript: VersionEntry[];
  corrected: VersionEntry[];
  translations: Record<string, VersionEntry[]>;
  synthesize: VersionEntry[];
  other: VersionEntry[];
}

/** Bucket every version by step. Languages are passed in because a
 *  translation's step *is* its language code, so there is nothing in the
 *  version itself that distinguishes one from an unknown step. */
export function groupVersions(
  versions: VersionEntry[] | undefined,
  languages: readonly string[],
): VersionGroups {
  const groups: VersionGroups = {
    transcript: [],
    corrected: [],
    translations: {},
    synthesize: [],
    other: [],
  };
  for (const lang of languages) groups.translations[lang] = [];
  if (!versions) return groups;
  for (const v of versions) {
    if (v.step === "transcript") groups.transcript.push(v);
    else if (v.step === "corrected") groups.corrected.push(v);
    else if (v.step === "synthesize") groups.synthesize.push(v);
    else if (v.step && languages.includes(v.step)) {
      (groups.translations[v.step] ??= []).push(v);
    } else {
      groups.other.push(v);
    }
  }
  return groups;
}

export interface RunPairing {
  /** Intermediates filed under each transcript version's id, newest first. */
  childrenByTranscriptId: Map<string, VersionEntry[]>;
  /** Intermediates with no transcript to belong to; shown under "other". */
  orphanIntermediates: VersionEntry[];
}

/**
 * File each transcribe intermediate under the transcript(s) of its own run.
 *
 * A diarized batch run emits TWO transcript versions (undiarized and
 * diarized) from a single set of intermediates, so a plain "closest later
 * transcript" pairing files the diarization intermediates under the
 * undiarized one. Grouping by run instead: raw whisper `segments` belong
 * under every transcript of the run; diarization-derived intermediates
 * belong under the diarized transcript(s), falling back to all of them when
 * none is tagged. Orphans (no transcript in the run) are returned separately.
 */
export function pairIntermediatesWithRuns(groups: VersionGroups): RunPairing {
  const stream = [
    ...groups.transcript.map((v) => ({ v, inter: false })),
    ...groups.other
      .filter((v) => v.step && TRANSCRIBE_INTERMEDIATE_STEPS.has(v.step))
      .map((v) => ({ v, inter: true })),
  ].sort((a, b) => a.v.timestamp.localeCompare(b.v.timestamp));

  const map = new Map<string, VersionEntry[]>();
  const orphans: VersionEntry[] = [];
  let pending: VersionEntry[] = [];
  let runTx: VersionEntry[] = [];

  const flush = () => {
    const diarizedTx = runTx.filter(
      (t) => (t.params as { diarize?: unknown } | undefined)?.diarize === true,
    );
    for (const inter of pending) {
      const targets =
        inter.step !== "segments" && diarizedTx.length > 0 ? diarizedTx : runTx;
      if (targets.length === 0) {
        orphans.push(inter);
        continue;
      }
      for (const t of targets) {
        const arr = map.get(t.id) ?? [];
        arr.push(inter);
        map.set(t.id, arr);
      }
    }
    pending = [];
    runTx = [];
  };

  for (const { v, inter } of stream) {
    if (inter) {
      if (runTx.length > 0) flush(); // an intermediate after a run begins the next
      pending.push(v);
    } else {
      // A transcript far in time from the current run's transcripts is a
      // separate run — a json/subtitle import emits no intermediates, so
      // without this it would absorb the previous run's intermediates.
      const prev = runTx[runTx.length - 1];
      if (prev && Date.parse(v.timestamp) - Date.parse(prev.timestamp) > RUN_GAP_MS) {
        flush();
      }
      runTx.push(v);
    }
  }
  flush();

  for (const arr of map.values()) {
    arr.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
  }
  return { childrenByTranscriptId: map, orphanIntermediates: orphans };
}

const newestFirst = (a: VersionEntry, b: VersionEntry) =>
  b.timestamp.localeCompare(a.timestamp);

/** Every text version of the episode, newest first. No synth: that is audio,
 *  and the "All transcript versions" table is about text. */
export function transcriptVersionsOf(groups: VersionGroups): VersionEntry[] {
  return [
    ...groups.transcript,
    ...groups.corrected,
    ...Object.values(groups.translations).flat(),
  ].sort(newestFirst);
}

/** Everything the transcript table does not show: synthesized audio, steps
 *  with no home, and intermediates whose run has no transcript. */
export function otherFilesOf(
  groups: VersionGroups,
  orphanIntermediates: VersionEntry[],
): VersionEntry[] {
  return [
    ...groups.synthesize,
    ...groups.other.filter(
      (v) => !v.step || !TRANSCRIBE_INTERMEDIATE_STEPS.has(v.step),
    ),
    ...orphanIntermediates,
  ].sort(newestFirst);
}

/** The activity feed shows every step event, synth included. */
export function recentActivityOf(
  transcriptVersions: VersionEntry[],
  groups: VersionGroups,
): VersionEntry[] {
  return [...transcriptVersions, ...groups.synthesize].sort(newestFirst);
}
