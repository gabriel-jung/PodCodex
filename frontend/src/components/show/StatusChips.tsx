import type { Episode } from "@/api/types";
import { isEdited, type BackendStepStatus } from "@/lib/stepStatus";

type StepStatus = "none" | "partial" | "outdated" | "done";

type StageKey = "transcribe" | "correct" | "translate" | "synth";

const STAGE_CLASSES: Record<StageKey, { bg: string; text: string; border: string }> = {
  transcribe: { bg: "bg-stage-transcribe/15", text: "text-stage-transcribe", border: "border-stage-transcribe" },
  correct:    { bg: "bg-stage-correct/15",    text: "text-stage-correct",    border: "border-stage-correct" },
  translate:  { bg: "bg-stage-translate/15",  text: "text-stage-translate",  border: "border-stage-translate" },
  synth:      { bg: "bg-stage-synth/15",      text: "text-stage-synth",      border: "border-stage-synth" },
};

/** Combine "freshness" status from backend (`raw | outdated | done`) with
 *  provenance-derived "review" status:
 *    - not present        → none
 *    - present + outdated → outdated (dashed)
 *    - present + edited   → done     (solid fill)
 *    - present + raw only → partial  (outlined — awaiting review)
 */
function resolveStatus(
  backendStatus: BackendStepStatus | string | undefined,
  present: boolean,
  provenance: unknown,
): StepStatus {
  if (!present) return "none";
  if (backendStatus === "outdated") return "outdated";
  return isEdited(provenance) ? "done" : "partial";
}

function Chip({ status, stage, label, title }: { status: StepStatus; stage: StageKey; label: string; title: string }) {
  if (status === "none") return null;
  const c = STAGE_CLASSES[stage];
  const base = "text-2xs leading-none px-1.5 py-0.5 rounded-full font-medium";
  if (status === "outdated") {
    return (
      <span className={`${base} border border-dashed ${c.border} ${c.text}`} title={`${title} (outdated)`}>
        {label}
      </span>
    );
  }
  if (status === "partial") {
    return (
      <span className={`${base} border ${c.border} ${c.text}`} title={`${title} (raw — awaiting review)`}>
        {label}
      </span>
    );
  }
  return (
    <span className={`${base} ${c.bg} ${c.text}`} title={title}>
      {label}
    </span>
  );
}

export function StatusChips({ ep, compact }: { ep: Episode; compact?: boolean }) {
  return (
    <div className="flex gap-1 items-center flex-wrap">
      <Chip
        status={resolveStatus(ep.transcribe_status, !!ep.transcribed, ep.provenance?.transcript)}
        stage="transcribe"
        label={compact ? "T" : "Transcribed"}
        title="Transcribed"
      />
      <Chip
        status={resolveStatus(ep.correct_status, !!ep.corrected, ep.provenance?.corrected)}
        stage="correct"
        label={compact ? "AI" : "Corrected"}
        title="Corrected"
      />
      {compact && ep.translations.length > 0 ? (
        ep.translations.map((lang) => (
          <Chip
            key={lang}
            status={resolveStatus(ep.translate_status, true, ep.provenance?.[lang])}
            stage="translate"
            label={lang.toUpperCase()}
            title={`Translated (${lang})`}
          />
        ))
      ) : (
        <Chip
          status={resolveStatus(
            ep.translate_status,
            ep.translations.length > 0,
            ep.provenance?.[ep.translations[0] ?? ""],
          )}
          stage="translate"
          label="Translated"
          title={ep.translations.length > 0 ? `Translated (${ep.translations.join(", ")})` : "Translated"}
        />
      )}
      {ep.synthesized && (
        <span className="text-2xs leading-none px-1.5 py-0.5 rounded-full font-medium bg-stage-synth/15 text-stage-synth" title="Synthesized">
          {compact ? "S" : "Synth"}
        </span>
      )}
      {ep.indexed && (
        <span className="text-2xs leading-none px-1.5 py-0.5 rounded-full font-medium bg-warning/15 text-warning" title="Indexed">
          {compact ? "I" : "Indexed"}
        </span>
      )}
    </div>
  );
}
