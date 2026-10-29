import { Fragment } from "react";
import { Star } from "lucide-react";
import type { Episode } from "@/api/types";
import {
  isEdited,
  plainStatus,
  reviewStatus,
  translationsStatus,
  type PanelStatus,
} from "@/lib/stepStatus";
import { STAGE_CLASSES, type StageKey } from "@/lib/stageClasses";
import { shortVersionId } from "@/lib/utils";

interface Entry {
  verb: string;
  hue: string;
  needsReview: boolean;
  partial: boolean;
  verified: boolean;
  outdated: boolean;
  title: string;
}

function entry(
  status: PanelStatus,
  verb: string,
  stage: StageKey,
  title?: string,
  partial = false,
  verified = false,
): Entry | null {
  if (status === "none") return null;
  return {
    verb,
    hue: STAGE_CLASSES[stage].text,
    needsReview: status === "review",
    partial,
    verified,
    outdated: false,
    title: partial
      ? `${verb} · some batches were rejected by the LLM`
      : (title ?? (status === "ready" ? verb : `${verb} · awaiting review`)),
  };
}

function EntrySpan({ entry }: { entry: Entry }) {
  return (
    <span title={entry.title} className={entry.hue}>
      {entry.verified && (
        <Star
          className="inline w-2.5 h-2.5 mr-0.5 -mt-px"
          fill="currentColor"
          aria-hidden="true"
        />
      )}
      {entry.verb}
      {entry.needsReview && (
        <span className="opacity-70"> (needs review)</span>
      )}
      {entry.partial && (
        <span className="text-destructive"> (partially failed)</span>
      )}
      {entry.outdated && <span className="opacity-70"> (outdated)</span>}
    </span>
  );
}

function translationsEntry(ep: Episode): Entry | null {
  const langs = ep.translations;
  const status = translationsStatus(langs, ep.provenance);
  if (status === "none") return null;
  const someEdited = langs.some((l) => isEdited(ep.provenance?.[l]));
  const partial = langs.some((l) => ep.llm_failed_steps?.includes(l));
  const title =
    status === "ready"
      ? `translated (${langs.join(", ")})`
      : someEdited
        ? `translated (${langs.join(", ")}) · some awaiting review`
        : `translated (${langs.join(", ")}) · awaiting review`;
  return entry(status, "translated", "translate", title, partial);
}

// A verified pointer marks one transcript/corrected version as the final
// source, so the underlying step stops mattering: collapse both the
// "transcribed" and "corrected" chips into a single "verified" chip.
function verifiedEntry(ep: Episode): Entry {
  return {
    verb: "verified",
    hue: "text-verified",
    needsReview: false,
    partial: false,
    verified: true,
    outdated: false,
    title: `verified version · ${ep.verified!.step} (v${shortVersionId(ep.verified!.version_id)})`,
  };
}

// The index records which source version it was cut from; after an edit,
// search keeps serving the old text until the episode is re-indexed.
function indexEntry(ep: Episode): Entry | null {
  const base = entry(plainStatus(!!ep.indexed), "indexed", "index");
  if (!base || !ep.index_outdated) return base;
  return {
    ...base,
    outdated: true,
    title: "indexed · the transcript changed since, re-index to search the current text",
  };
}

export function StatusChips({ ep }: { ep: Episode }) {
  const sourceEntries: (Entry | null)[] = ep.verified
    ? [verifiedEntry(ep)]
    : [
        entry(reviewStatus(!!ep.transcribed, ep.provenance?.transcript), "transcribed", "transcribe"),
        entry(
          reviewStatus(!!ep.corrected, ep.provenance?.corrected),
          "corrected",
          "correct",
          undefined,
          !!ep.llm_failed_steps?.includes("corrected"),
        ),
      ];
  const entries: Entry[] = [
    ...sourceEntries,
    translationsEntry(ep),
    entry(plainStatus(!!ep.synthesized), "synthesized", "synth"),
    indexEntry(ep),
  ].filter((x): x is Entry => x !== null);

  if (entries.length === 0) return null;

  return (
    <p className="text-2xs leading-snug flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5">
      {entries.map((e, i) => (
        <Fragment key={i}>
          {i > 0 && <span aria-hidden="true" className="text-muted-foreground/40">·</span>}
          <EntrySpan entry={e} />
        </Fragment>
      ))}
    </p>
  );
}
