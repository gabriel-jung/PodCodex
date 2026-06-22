import { Fragment } from "react";
import type { ShowSummary } from "@/api/types";
import { STAGE_CLASSES, type StageKey } from "@/lib/stageClasses";

interface Props {
  show: ShowSummary;
  /** Render all stages on one wrapping flow (list-view rows). Default = vertical stack. */
  dense?: boolean;
}

interface Entry {
  count: number;
  waiting: number;
  verb: string;
  hue: string;
  title: string;
}

function reviewable(
  total: number | null | undefined,
  edited: number | null | undefined,
  verb: string,
  stage: StageKey,
): Entry | null {
  const t = total ?? 0;
  if (t <= 0) return null;
  const e = Math.min(t, edited ?? 0);
  const w = t - e;
  const title = w === 0
    ? `${t} ${verb} · all reviewed`
    : `${t} ${verb} · ${e === 0 ? "" : `${w} `}waiting for review`;
  return { count: t, waiting: w, verb, hue: STAGE_CLASSES[stage].text, title };
}

function plain(
  total: number | null | undefined,
  verb: string,
  stage: StageKey,
  hueClass?: string,
): Entry | null {
  const t = total ?? 0;
  if (t <= 0) return null;
  return { count: t, waiting: 0, verb, hue: hueClass ?? STAGE_CLASSES[stage].text, title: `${t} ${verb}` };
}

function EntrySpan({ entry }: { entry: Entry }) {
  return (
    <span title={entry.title} className={entry.hue}>
      <span className="font-mono tabular-nums">{entry.count}</span>{" "}
      {entry.verb}
      {entry.waiting > 0 && (
        <span className="opacity-70">
          {" "}(<span className="font-mono tabular-nums">{entry.waiting}</span> to review)
        </span>
      )}
    </span>
  );
}

export default function ShowProgressStrip({ show, dense }: Props) {
  const reviewables = [
    reviewable(show.transcribed_count, show.transcribed_edited_count, "transcribed", "transcribe"),
    reviewable(show.corrected_count, show.corrected_edited_count, "corrected", "correct"),
    reviewable(show.translated_count, show.translated_edited_count, "translated", "translate"),
  ].filter((x): x is Entry => x !== null);

  const plains = [
    plain(show.synthesized_count, "synthesized", "synth"),
    plain(show.indexed_count, "indexed", "index"),
    plain(show.verified_count, "verified", "correct", "text-verified"),
  ].filter((x): x is Entry => x !== null);

  if (reviewables.length === 0 && plains.length === 0) return null;

  if (dense) {
    const all = [...reviewables, ...plains];
    return (
      <p className="text-2xs leading-snug flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5">
        {all.map((e, i) => (
          <Fragment key={i}>
            {i > 0 && <span aria-hidden="true" className="text-muted-foreground/40">·</span>}
            <EntrySpan entry={e} />
          </Fragment>
        ))}
      </p>
    );
  }

  return (
    <ul className="text-2xs leading-snug space-y-0.5">
      {reviewables.map((e, i) => (
        <li key={`r${i}`} className="truncate">
          <EntrySpan entry={e} />
        </li>
      ))}
      {plains.length > 0 && (
        <li className="truncate flex flex-wrap items-baseline gap-x-1.5">
          {plains.map((e, i) => (
            <Fragment key={`p${i}`}>
              {i > 0 && <span aria-hidden="true" className="text-muted-foreground/40">·</span>}
              <EntrySpan entry={e} />
            </Fragment>
          ))}
        </li>
      )}
    </ul>
  );
}
