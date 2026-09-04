import { describe, expect, it } from "vitest";
import type { VersionEntry } from "@/api/types";
import {
  groupVersions,
  otherFilesOf,
  pairIntermediatesWithRuns,
  recentActivityOf,
  transcriptVersionsOf,
} from "./versionGroups";

/**
 * The run pairing is the rule that decides which transcribe by-products hang
 * under which transcript version in the Overview hub, and it has two
 * awkward cases a reader cannot verify by inspection: a diarized run emits
 * two transcripts from one set of intermediates, and an import emits a
 * transcript with none at all.
 */

let clock = 0;
function v(step: string, opts: Partial<VersionEntry> & { at?: number } = {}): VersionEntry {
  const at = opts.at ?? clock++;
  return {
    id: opts.id ?? `${step}-${at}`,
    step,
    timestamp: new Date(Date.UTC(2026, 0, 1, 0, 0, at)).toISOString(),
    params: opts.params ?? {},
  } as VersionEntry;
}

/** A version this many minutes after the epoch used above. */
function minutesLater(
  step: string,
  minutes: number,
  params?: Record<string, unknown>,
  id?: string,
) {
  return {
    id: id ?? `${step}-m${minutes}`,
    step,
    timestamp: new Date(Date.UTC(2026, 0, 1, 0, minutes, 0)).toISOString(),
    params: params ?? {},
  } as VersionEntry;
}

describe("groupVersions", () => {
  it("buckets each step, and seeds an empty list per known language", () => {
    const groups = groupVersions(
      [v("transcript"), v("corrected"), v("fr"), v("synthesize"), v("segments")],
      ["fr", "de"],
    );
    expect(groups.transcript).toHaveLength(1);
    expect(groups.corrected).toHaveLength(1);
    expect(groups.synthesize).toHaveLength(1);
    expect(groups.translations.fr).toHaveLength(1);
    expect(groups.translations.de).toEqual([]);
    expect(groups.other.map((x) => x.step)).toEqual(["segments"]);
  });

  it("files a translation whose language is not in the list under other", () => {
    // A language pruned from the episode while its version survives.
    const groups = groupVersions([v("es")], ["fr"]);
    expect(groups.other.map((x) => x.step)).toEqual(["es"]);
  });

  it("returns empty groups for no versions at all", () => {
    const groups = groupVersions(undefined, ["fr"]);
    expect(groups.transcript).toEqual([]);
    expect(groups.translations.fr).toEqual([]);
  });
});

describe("pairIntermediatesWithRuns", () => {
  it("files a run's intermediates under its transcript", () => {
    clock = 0;
    const segments = v("segments");
    const transcript = v("transcript");
    const groups = groupVersions([segments, transcript], []);

    const { childrenByTranscriptId, orphanIntermediates } =
      pairIntermediatesWithRuns(groups);

    expect(childrenByTranscriptId.get(transcript.id)).toEqual([segments]);
    expect(orphanIntermediates).toEqual([]);
  });

  it("gives whisper segments to both transcripts of a diarized run, and the diarization only to the diarized one", () => {
    clock = 0;
    const segments = v("segments");
    const diarization = v("diarization");
    const plain = v("transcript", { id: "plain", params: { diarize: false } });
    const diarized = v("transcript", { id: "diarized", params: { diarize: true } });
    const groups = groupVersions([segments, diarization, plain, diarized], []);

    const { childrenByTranscriptId } = pairIntermediatesWithRuns(groups);

    expect(childrenByTranscriptId.get("plain")?.map((x) => x.step)).toEqual(["segments"]);
    expect(childrenByTranscriptId.get("diarized")?.map((x) => x.step)?.sort()).toEqual([
      "diarization",
      "segments",
    ]);
  });

  it("gives everything to all transcripts when none is tagged diarized", () => {
    clock = 0;
    const groups = groupVersions(
      [v("segments"), v("diarization"), v("transcript", { id: "only" })],
      [],
    );

    const { childrenByTranscriptId } = pairIntermediatesWithRuns(groups);

    expect(childrenByTranscriptId.get("only")).toHaveLength(2);
  });

  it("does not let an import absorb the previous run's intermediates", () => {
    // The import is far enough after the run to be a separate one, and it
    // writes no intermediates of its own.
    const groups = groupVersions(
      [
        minutesLater("segments", 0),
        minutesLater("transcript", 1),
        minutesLater("transcript", 30),
      ],
      [],
    );

    const { childrenByTranscriptId } = pairIntermediatesWithRuns(groups);

    expect(childrenByTranscriptId.get("transcript-m1")).toHaveLength(1);
    expect(childrenByTranscriptId.get("transcript-m30")).toBeUndefined();
  });

  it("keeps the two transcripts of one run together despite the gap rule", () => {
    const groups = groupVersions(
      [
        minutesLater("segments", 0),
        minutesLater("transcript", 1, { diarize: false }, "plain"),
        minutesLater("transcript", 1, { diarize: true }, "diarized"),
      ],
      [],
    );

    const { childrenByTranscriptId, orphanIntermediates } =
      pairIntermediatesWithRuns(groups);

    expect(childrenByTranscriptId.size).toBe(2);
    expect(orphanIntermediates).toEqual([]);
  });

  it("reports an intermediate with no transcript as an orphan", () => {
    clock = 0;
    const groups = groupVersions([v("segments")], []);

    const { childrenByTranscriptId, orphanIntermediates } =
      pairIntermediatesWithRuns(groups);

    expect(childrenByTranscriptId.size).toBe(0);
    expect(orphanIntermediates.map((x) => x.step)).toEqual(["segments"]);
  });

  it("sorts each transcript's children newest first", () => {
    clock = 0;
    const groups = groupVersions(
      [v("segments"), v("diarization"), v("transcript", { id: "t" })],
      [],
    );

    const kids = pairIntermediatesWithRuns(groups).childrenByTranscriptId.get("t")!;

    expect(kids[0].timestamp >= kids[1].timestamp).toBe(true);
  });
});

describe("the three list views", () => {
  it("transcriptVersionsOf is text only, newest first", () => {
    clock = 0;
    const groups = groupVersions(
      [v("transcript"), v("corrected"), v("fr"), v("synthesize")],
      ["fr"],
    );

    const list = transcriptVersionsOf(groups);

    expect(list.map((x) => x.step)).toEqual(["fr", "corrected", "transcript"]);
  });

  it("otherFilesOf carries synth, unknown steps and orphans, but no filed intermediates", () => {
    clock = 0;
    const groups = groupVersions([v("synthesize"), v("segments"), v("mystery")], []);
    const orphan = v("diarization");

    const list = otherFilesOf(groups, [orphan]);

    expect(list.map((x) => x.step).sort()).toEqual([
      "diarization",
      "mystery",
      "synthesize",
    ]);
  });

  it("recentActivityOf adds synth back to the text versions", () => {
    clock = 0;
    const groups = groupVersions([v("transcript"), v("synthesize")], []);
    const text = transcriptVersionsOf(groups);

    const feed = recentActivityOf(text, groups);

    expect(feed.map((x) => x.step)).toEqual(["synthesize", "transcript"]);
  });
});
