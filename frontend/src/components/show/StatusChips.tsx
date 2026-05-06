import type { Episode } from "@/api/types";
import { isEdited } from "@/lib/stepStatus";
import { STAGE_CLASSES, type StageKey } from "@/lib/stageClasses";

type StepStatus = "none" | "partial" | "done";

type ChipStage = Exclude<StageKey, "index">;

/** Provenance-derived review status. Backend "outdated" freshness is hidden
 *  here (reserved for the planned advanced filter) — outdated raw output and
 *  fresh raw output both surface as `partial`.
 *    - not present      → none
 *    - present + edited → done    (solid fill)
 *    - present + raw    → partial (outlined, awaiting review)
 */
function resolveStatus(present: boolean, provenance: unknown): StepStatus {
  if (!present) return "none";
  if (isEdited(provenance)) return "done";
  return "partial";
}

function Chip({ status, stage, label, title }: { status: StepStatus; stage: ChipStage; label: string; title: string }) {
  if (status === "none") return null;
  const c = STAGE_CLASSES[stage];
  const base = "text-2xs leading-none px-1.5 py-0.5 rounded-full font-medium";
  if (status === "partial") {
    return (
      <span className={`${base} border ${c.border} ${c.text}`} title={`${title} (raw, awaiting review)`}>
        {label}
      </span>
    );
  }
  return (
    <span className={`${base} ${c.bg} ${c.text}`} title={title} role="img" aria-label={title}>
      {label}
    </span>
  );
}

export function StatusChips({ ep, compact }: { ep: Episode; compact?: boolean }) {
  return (
    <div className="flex gap-1 items-center flex-wrap">
      <Chip
        status={resolveStatus(!!ep.transcribed, ep.provenance?.transcript)}
        stage="transcribe"
        label={compact ? "T" : "Transcribed"}
        title="Transcribed"
      />
      <Chip
        status={resolveStatus(!!ep.corrected, ep.provenance?.corrected)}
        stage="correct"
        label={compact ? "AI" : "Corrected"}
        title="Corrected"
      />
      {compact && ep.translations.length > 0 ? (
        ep.translations.map((lang) => (
          <Chip
            key={lang}
            status={resolveStatus(true, ep.provenance?.[lang])}
            stage="translate"
            label={lang.toUpperCase()}
            title={`Translated (${lang})`}
          />
        ))
      ) : (
        <Chip
          status={resolveStatus(
            ep.translations.length > 0,
            ep.provenance?.[ep.translations[0] ?? ""],
          )}
          stage="translate"
          label="Translated"
          title={ep.translations.length > 0 ? `Translated (${ep.translations.join(", ")})` : "Translated"}
        />
      )}
      {ep.synthesized && (
        <span className="text-2xs leading-none px-1.5 py-0.5 rounded-full font-medium bg-stage-synth/15 text-stage-synth" title="Synthesized" role="img" aria-label="Synthesized">
          {compact ? "S" : "Synth"}
        </span>
      )}
      {ep.indexed && (
        <span className="text-2xs leading-none px-1.5 py-0.5 rounded-full font-medium bg-warning/15 text-warning" title="Indexed" role="img" aria-label="Indexed">
          {compact ? "I" : "Indexed"}
        </span>
      )}
    </div>
  );
}
