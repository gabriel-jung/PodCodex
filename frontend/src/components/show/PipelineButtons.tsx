import { useState } from "react";
import type { Episode } from "@/api/types";
import { Button } from "@/components/ui/button";
import StepConfigEditor, { STEPS, type StepKey, type TranscribeSource } from "./StepConfigEditor";
import { episodeNeedsStep } from "@/lib/stepStatus";

/** Count episodes that need this step (prerequisites met + not already up-to-date). */
function countCanRun(episodes: Episode[], step: StepKey): number {
  return episodes.filter((e) => {
    const hasPrereq = step === "transcribe"
      ? (!!e.audio_path || !!e.has_subtitles)
      : !!e.transcribed;
    return hasPrereq && episodeNeedsStep(e, step);
  }).length;
}

export default function PipelineButtons({
  disabled,
  episodes,
  showLanguage,
  onRun,
}: {
  disabled: boolean;
  episodes: Episode[];
  showLanguage: string;
  onRun: (step: StepKey, filteredEpisodes?: Episode[], sourceVersionIds?: Record<string, string>, transcribeSource?: TranscribeSource, force?: boolean) => void;
}) {
  const [confirmStep, setConfirmStep] = useState<StepKey | null>(null);

  const handleConfirm = (filteredEpisodes?: Episode[], sourceVersionIds?: Record<string, string>, transcribeSource?: TranscribeSource, force?: boolean) => {
    if (confirmStep) onRun(confirmStep, filteredEpisodes, sourceVersionIds, transcribeSource, force);
    setConfirmStep(null);
  };

  // count=0 → outline (open editor to reprocess); count>0 → highlighted.
  const stepsWithCount = STEPS.map((s) => ({ ...s, count: countCanRun(episodes, s.key) }));

  return (
    <>
      <div className="flex items-center gap-1">
        {stepsWithCount.map(({ key, label, icon: Icon, count }) => {
          const hasWork = count > 0;
          return (
            <Button
              key={key}
              onClick={() => setConfirmStep(key)}
              disabled={disabled}
              variant={hasWork ? "default" : "outline"}
              size="sm"
              className={`text-xs h-7 px-2 ${hasWork ? "" : "opacity-60"}`}
              title={hasWork
                ? `${label} ${count} episode${count === 1 ? "" : "s"}`
                : `${label} — all up to date (open to reprocess)`}
            >
              <Icon className="w-3 h-3" />
              <span>{label}</span>
              {hasWork && <span className="tabular-nums opacity-80">{count}</span>}
            </Button>
          );
        })}
      </div>

      {confirmStep && (
        <StepConfigEditor
          step={confirmStep}
          episodes={episodes}
          showLanguage={showLanguage}
          onRun={handleConfirm}
          onClose={() => setConfirmStep(null)}
        />
      )}
    </>
  );
}
