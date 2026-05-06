import { Fragment } from "react";
import type { Episode } from "@/api/types";
import { isEdited } from "@/lib/stepStatus";
import { STAGE_CLASSES, type StageKey } from "@/lib/stageClasses";

interface Entry {
  verb: string;
  hue: string;
  needsReview: boolean;
  title: string;
}

function reviewable(
  present: boolean,
  provenance: unknown,
  verb: string,
  stage: StageKey,
): Entry | null {
  if (!present) return null;
  const edited = isEdited(provenance);
  return {
    verb,
    hue: STAGE_CLASSES[stage].text,
    needsReview: !edited,
    title: edited ? verb : `${verb} · awaiting review`,
  };
}

function plain(present: boolean, verb: string, stage: StageKey): Entry | null {
  if (!present) return null;
  return { verb, hue: STAGE_CLASSES[stage].text, needsReview: false, title: verb };
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

export function StatusChips({ ep }: { ep: Episode }) {
  const langs = ep.translations;
  const anyTranslationEdited = langs.some((l) => isEdited(ep.provenance?.[l]));
  const allTranslationsEdited = langs.length > 0 && langs.every((l) => isEdited(ep.provenance?.[l]));

  const entries: Entry[] = [
    reviewable(!!ep.transcribed, ep.provenance?.transcript, "transcribed", "transcribe"),
    reviewable(!!ep.corrected, ep.provenance?.corrected, "corrected", "correct"),
    langs.length > 0
      ? {
          verb: "translated",
          hue: STAGE_CLASSES.translate.text,
          needsReview: !allTranslationsEdited,
          title:
            allTranslationsEdited
              ? `translated (${langs.join(", ")})`
              : anyTranslationEdited
                ? `translated (${langs.join(", ")}) · some awaiting review`
                : `translated (${langs.join(", ")}) · awaiting review`,
        }
      : null,
    plain(!!ep.synthesized, "synthesized", "synth"),
    plain(!!ep.indexed, "indexed", "index"),
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
