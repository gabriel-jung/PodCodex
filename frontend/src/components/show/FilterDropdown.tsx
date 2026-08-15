import { useState } from "react";
import { Button } from "@/components/ui/button";
import { SlidersHorizontal } from "lucide-react";
import { useEpisodeStore } from "@/stores";
import { langLabel } from "@/lib/utils";
import {
  statesForStep,
  type StepFilterState,
  type StepFilterStep,
} from "@/lib/stepStatus";

const STEP_LABELS: Record<StepFilterStep, string> = {
  transcribe: "Transcribe",
  correct: "Correct",
  translate: "Translate",
  index: "Index",
  synthesize: "Synthesize",
};

const STATE_LABELS: Record<StepFilterState, string> = {
  missing: "Not started",
  done: "Done",
  raw: "Needs review",
  edited: "Edited",
  outdated: "Outdated",
};

/** `languages` are the translation languages present in this show, used to
 *  narrow a translate filter (e.g. "missing translation in French"). */
export default function FilterDropdown({ languages = [] }: { languages?: string[] }) {
  const minDurationMinutes = useEpisodeStore((s) => s.minDurationMinutes);
  const setMinDurationMinutes = useEpisodeStore((s) => s.setMinDurationMinutes);
  const maxDurationMinutes = useEpisodeStore((s) => s.maxDurationMinutes);
  const setMaxDurationMinutes = useEpisodeStore((s) => s.setMaxDurationMinutes);
  const titleInclude = useEpisodeStore((s) => s.titleInclude);
  const setTitleInclude = useEpisodeStore((s) => s.setTitleInclude);
  const titleExclude = useEpisodeStore((s) => s.titleExclude);
  const setTitleExclude = useEpisodeStore((s) => s.setTitleExclude);
  const stepFilterStep = useEpisodeStore((s) => s.stepFilterStep);
  const stepFilterState = useEpisodeStore((s) => s.stepFilterState);
  const stepFilterLang = useEpisodeStore((s) => s.stepFilterLang);
  const setStepFilter = useEpisodeStore((s) => s.setStepFilter);
  const [open, setOpen] = useState(false);
  const activeCount = [
    minDurationMinutes > 0,
    maxDurationMinutes > 0,
    titleInclude.length > 0,
    titleExclude.length > 0,
    stepFilterStep !== "",
  ].filter(Boolean).length;

  const clearAll = () => {
    setMinDurationMinutes(0);
    setMaxDurationMinutes(0);
    setTitleInclude("");
    setTitleExclude("");
    setStepFilter("");
  };

  // Switching to a step that lacks the current state (e.g. "outdated" then
  // Index) would filter to nothing, so fall back to the step's first state.
  const onStepChange = (next: StepFilterStep | "") => {
    if (!next) return setStepFilter("");
    const allowed = statesForStep(next);
    const state = allowed.includes(stepFilterState) ? stepFilterState : allowed[0];
    setStepFilter(next, state);
  };

  return (
    <div className="relative">
      <Button
        onClick={() => setOpen(!open)}
        variant={activeCount > 0 ? "secondary" : "ghost"}
        size="sm"
        className="text-xs h-7 px-2 gap-1"
      >
        <SlidersHorizontal className="w-3 h-3" />
        Filters
        {activeCount > 0 && <span className="bg-primary text-primary-foreground rounded-full px-1 text-2xs">{activeCount}</span>}
      </Button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute left-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg p-3 min-w-[240px] space-y-3">
            <div className="space-y-2">
              <span className="text-xs font-medium">Duration</span>
              <div className="flex items-center gap-2">
                <input
                  type="number" min={0} step={5}
                  value={minDurationMinutes || ""}
                  onChange={(e) => setMinDurationMinutes(Math.max(0, Number(e.target.value)))}
                  placeholder="min"
                  className="input w-16 text-xs text-center"
                />
                <span className="text-xs text-muted-foreground">to</span>
                <input
                  type="number" min={0} step={5}
                  value={maxDurationMinutes || ""}
                  onChange={(e) => setMaxDurationMinutes(Math.max(0, Number(e.target.value)))}
                  placeholder="max"
                  className="input w-16 text-xs text-center"
                />
                <span className="text-xs text-muted-foreground">min</span>
              </div>
            </div>
            <div className="space-y-2">
              <span className="text-xs font-medium">Title contains</span>
              <input
                value={titleInclude}
                onChange={(e) => setTitleInclude(e.target.value)}
                placeholder="word or phrase..."
                className="input w-full text-xs"
              />
            </div>
            <div className="space-y-2">
              <span className="text-xs font-medium">Title excludes</span>
              <input
                value={titleExclude}
                onChange={(e) => setTitleExclude(e.target.value)}
                placeholder="word or phrase..."
                className="input w-full text-xs"
              />
            </div>
            <div className="space-y-2">
              <span className="text-xs font-medium">Pipeline step</span>
              <div className="flex items-center gap-2">
                <select
                  value={stepFilterStep}
                  onChange={(e) => onStepChange(e.target.value as StepFilterStep | "")}
                  className="input flex-1 text-xs"
                  aria-label="Filter by pipeline step"
                >
                  <option value="">Any step</option>
                  {(Object.keys(STEP_LABELS) as StepFilterStep[]).map((s) => (
                    <option key={s} value={s}>{STEP_LABELS[s]}</option>
                  ))}
                </select>
                {stepFilterStep && (
                  <select
                    value={stepFilterState}
                    onChange={(e) =>
                      setStepFilter(stepFilterStep, e.target.value as StepFilterState)
                    }
                    className="input flex-1 text-xs"
                    aria-label="Filter by step state"
                  >
                    {statesForStep(stepFilterStep).map((st) => (
                      <option key={st} value={st}>{STATE_LABELS[st]}</option>
                    ))}
                  </select>
                )}
              </div>
              {stepFilterStep === "translate" && languages.length > 0 && (
                <select
                  value={stepFilterLang}
                  onChange={(e) =>
                    setStepFilter("translate", stepFilterState, e.target.value)
                  }
                  className="input w-full text-xs"
                  aria-label="Filter by translation language"
                >
                  <option value="">Any language</option>
                  {languages.map((l) => (
                    <option key={l} value={l}>{langLabel(l)}</option>
                  ))}
                </select>
              )}
            </div>
            {activeCount > 0 && (
              <Button onClick={() => { clearAll(); setOpen(false); }} variant="ghost" size="sm" className="text-xs w-full">
                Clear all filters
              </Button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
