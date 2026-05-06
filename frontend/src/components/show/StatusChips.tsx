import { Fragment } from "react";
import type { Episode } from "@/api/types";
import {
  isEdited,
  plainStatus,
  reviewStatus,
  translationsStatus,
  type PanelStatus,
} from "@/lib/stepStatus";
import { STAGE_CLASSES, type StageKey } from "@/lib/stageClasses";

interface Entry {
  verb: string;
  hue: string;
  needsReview: boolean;
  title: string;
}

function entry(status: PanelStatus, verb: string, stage: StageKey, title?: string): Entry | null {
  if (status === "none") return null;
  return {
    verb,
    hue: STAGE_CLASSES[stage].text,
    needsReview: status === "review",
    title: title ?? (status === "ready" ? verb : `${verb} · awaiting review`),
  };
}

function EntrySpan({ entry }: { entry: Entry }) {
  return (
    <span title={entry.title} className={entry.hue}>
      {entry.verb}
      {entry.needsReview && (
        <span className="opacity-70"> (needs review)</span>
      )}
    </span>
  );
}

function translationsEntry(ep: Episode): Entry | null {
  const langs = ep.translations;
  const status = translationsStatus(langs, ep.provenance);
  if (status === "none") return null;
  const someEdited = langs.some((l) => isEdited(ep.provenance?.[l]));
  const title =
    status === "ready"
      ? `translated (${langs.join(", ")})`
      : someEdited
        ? `translated (${langs.join(", ")}) · some awaiting review`
        : `translated (${langs.join(", ")}) · awaiting review`;
  return entry(status, "translated", "translate", title);
}

export function StatusChips({ ep }: { ep: Episode }) {
  const entries: Entry[] = [
    entry(reviewStatus(!!ep.transcribed, ep.provenance?.transcript), "transcribed", "transcribe"),
    entry(reviewStatus(!!ep.corrected, ep.provenance?.corrected), "corrected", "correct"),
    translationsEntry(ep),
    entry(plainStatus(!!ep.synthesized), "synthesized", "synth"),
    entry(plainStatus(!!ep.indexed), "indexed", "index"),
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
