import { useState } from "react";
import type { Episode } from "@/api/types";
import { Button } from "@/components/ui/button";
import { ChevronDown, Zap } from "lucide-react";
import StepConfigEditor, { STEPS, type StepKey, type TranscribeSource } from "./StepConfigEditor";

/** Count episodes that need this step (prerequisites met + not already up-to-date). */
function countCanRun(episodes: Episode[], step: StepKey): number {
  return episodes.filter((e) => {
    switch (step) {
      case "transcribe": return (!!e.audio_path && e.transcribe_status !== "done") || !!e.has_subtitles;
      case "correct":    return !!e.transcribed && e.correct_status !== "done";
      case "translate":  return !!e.transcribed && e.translate_status !== "done";
      case "index":      return !!e.transcribed && !e.indexed;
      default:           return true;
    }
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
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmStep, setConfirmStep] = useState<StepKey | null>(null);

  const handleClick = (key: StepKey) => {
    setMenuOpen(false);
    setConfirmStep(key);
  };

  const handleConfirm = (filteredEpisodes?: Episode[], sourceVersionIds?: Record<string, string>, transcribeSource?: TranscribeSource, force?: boolean) => {
    if (confirmStep) onRun(confirmStep, filteredEpisodes, sourceVersionIds, transcribeSource, force);
    setConfirmStep(null);
  };

  return (
    <>
      <div className="relative">
        <Button
          onClick={() => setMenuOpen(!menuOpen)}
          disabled={disabled}
          variant="default"
          size="sm"
          className="text-xs h-7 px-3"
        >
          <Zap className="w-3 h-3" /> Process <ChevronDown className="w-3 h-3 ml-0.5" />
        </Button>
        {menuOpen && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setMenuOpen(false)} />
            <div className="absolute right-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg py-1 min-w-[180px]">
              {STEPS.map(({ key, label, icon: Icon }) => {
                const count = countCanRun(episodes, key);
                return (
                  <button
                    key={key}
                    onClick={() => handleClick(key)}
                    className={`w-full flex items-center gap-2 px-3 py-1.5 text-xs transition ${
                      count > 0 ? "hover:bg-accent" : "text-muted-foreground"
                    }`}
                  >
                    <Icon className="w-3 h-3" />
                    <span className="flex-1 text-left">{label}</span>
                    <span className="tabular-nums">{count}</span>
                  </button>
                );
              })}
            </div>
          </>
        )}
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
