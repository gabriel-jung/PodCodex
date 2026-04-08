import type { Episode } from "@/api/types";

type StepStatus = "none" | "outdated" | "done";

const BORDER_MAP: Record<string, string> = {
  "bg-blue-500/15": "border-blue-500",
  "bg-purple-500/15": "border-purple-500",
  "bg-teal-500/15": "border-teal-500",
};

function Chip({ status, label, color, textColor, title }: { status: StepStatus; label: string; color: string; textColor: string; title: string }) {
  if (status === "none") return null;
  const borderColor = BORDER_MAP[color] ?? "border-border";
  if (status === "outdated") {
    return (
      <span
        className={`text-2xs leading-none px-1.5 py-0.5 rounded-full border border-dashed font-medium ${borderColor} ${textColor}`}
        title={`${title} (outdated)`}
      >
        {label}
      </span>
    );
  }
  return (
    <span className={`text-2xs leading-none px-1.5 py-0.5 rounded-full font-medium ${color} ${textColor}`} title={title}>
      {label}
    </span>
  );
}

export function StatusChips({ ep, compact }: { ep: Episode; compact?: boolean }) {
  const translateStatus: StepStatus = ep.translations.length > 0
    ? ((ep.translate_status as StepStatus) || "done")
    : "none";

  return (
    <div className="flex gap-1 items-center flex-wrap">
      <Chip
        status={(ep.transcribe_status as StepStatus) || (ep.transcribed ? "done" : "none")}
        label={compact ? "T" : "Transcribed"}
        color="bg-blue-500/15"
        textColor="text-blue-500"
        title="Transcribed"
      />
      <Chip
        status={(ep.correct_status as StepStatus) || (ep.corrected ? "done" : "none")}
        label={compact ? "AI" : "Corrected"}
        color="bg-purple-500/15"
        textColor="text-purple-500"
        title="Corrected"
      />
      {compact && ep.translations.length > 0 ? (
        ep.translations.map((lang) => (
          <Chip
            key={lang}
            status={translateStatus}
            label={lang.toUpperCase()}
            color="bg-teal-500/15"
            textColor="text-teal-500"
            title={`Translated (${lang})`}
          />
        ))
      ) : (
        <Chip
          status={translateStatus}
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
