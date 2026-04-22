import type { Episode } from "@/api/types";
import { isEdited, type BackendStepStatus } from "@/lib/stepStatus";

type StepStatus = "none" | "partial" | "outdated" | "done";

const BORDER_MAP: Record<string, string> = {
  "bg-blue-500/15": "border-blue-500",
  "bg-purple-500/15": "border-purple-500",
  "bg-teal-500/15": "border-teal-500",
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

function Chip({ status, label, color, textColor, title }: { status: StepStatus; label: string; color: string; textColor: string; title: string }) {
  if (status === "none") return null;
  const borderColor = BORDER_MAP[color] ?? "border-border";
  const base = "text-2xs leading-none px-1.5 py-0.5 rounded-full font-medium";
  if (status === "outdated") {
    return (
      <span className={`${base} border border-dashed ${borderColor} ${textColor}`} title={`${title} (outdated)`}>
        {label}
      </span>
    );
  }
  if (status === "partial") {
    return (
      <span className={`${base} border ${borderColor} ${textColor}`} title={`${title} (raw — awaiting review)`}>
        {label}
      </span>
    );
  }
  return (
    <span className={`${base} ${color} ${textColor}`} title={title}>
      {label}
    </span>
  );
}

export function StatusChips({ ep, compact }: { ep: Episode; compact?: boolean }) {
  return (
    <div className="flex gap-1 items-center flex-wrap">
      <Chip
        status={resolveStatus(ep.transcribe_status, !!ep.transcribed, ep.provenance?.transcript)}
        label={compact ? "T" : "Transcribed"}
        color="bg-blue-500/15"
        textColor="text-blue-500"
        title="Transcribed"
      />
      <Chip
        status={resolveStatus(ep.correct_status, !!ep.corrected, ep.provenance?.corrected)}
        label={compact ? "AI" : "Corrected"}
        color="bg-purple-500/15"
        textColor="text-purple-500"
        title="Corrected"
      />
      {compact && ep.translations.length > 0 ? (
        ep.translations.map((lang) => (
          <Chip
            key={lang}
            status={resolveStatus(ep.translate_status, true, ep.provenance?.[lang])}
            label={lang.toUpperCase()}
            color="bg-teal-500/15"
            textColor="text-teal-500"
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
          label="Translated"
          color="bg-teal-500/15"
          textColor="text-teal-500"
          title={ep.translations.length > 0 ? `Translated (${ep.translations.join(", ")})` : "Translated"}
        />
      )}
      {ep.synthesized && (
        <span className="text-2xs leading-none px-1.5 py-0.5 rounded-full font-medium bg-orange-500/15 text-orange-500" title="Synthesized">
          {compact ? "S" : "Synth"}
        </span>
      )}
      {ep.indexed && (
        <span className="text-2xs leading-none px-1.5 py-0.5 rounded-full font-medium bg-warning/15 text-warning" title="Indexed">
          {compact ? "I" : "Indexed"}
        </span>
      )}
      {ep.no_subtitles && !ep.transcribed && (
        <span className="text-2xs leading-none px-1.5 py-0.5 rounded-full font-medium bg-muted text-muted-foreground" title="No subtitles available on YouTube">
          {compact ? "—" : "No subs"}
        </span>
      )}
    </div>
  );
}
