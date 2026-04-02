import { useState } from "react";
import type { Episode } from "@/api/types";
import { Button } from "@/components/ui/button";
import { ChevronDown } from "lucide-react";
import StepConfigEditor, { STEPS, type StepKey } from "./StepConfigEditor";

export default function PipelineButtons({
  disabled,
  episodes,
  showLanguage,
  onRun,
}: {
  disabled: boolean;
  episodes: Episode[];
  showLanguage: string;
  onRun: (step: StepKey) => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmStep, setConfirmStep] = useState<StepKey | null>(null);

  const handleClick = (key: StepKey) => {
    setMenuOpen(false);
    setConfirmStep(key);
  };

  const handleConfirm = () => {
    if (confirmStep) onRun(confirmStep);
    setConfirmStep(null);
  };

  return (
    <>
      <div className="relative">
        <Button
          onClick={() => setMenuOpen(!menuOpen)}
          disabled={disabled}
          variant="outline"
          size="sm"
          className="text-xs h-7 px-2"
        >
          Pipeline <ChevronDown className="w-3 h-3 ml-1" />
        </Button>
        {menuOpen && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setMenuOpen(false)} />
            <div className="absolute right-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg py-1 min-w-[140px]">
              {STEPS.map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => handleClick(key)}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-accent transition"
                >
                  <Icon className="w-3 h-3" /> {label}
                </button>
              ))}
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
