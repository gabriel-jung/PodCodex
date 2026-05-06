/**
 * Shared stage→Tailwind class map. Single source of truth for the pipeline
 * stage palette used by StatusChips, ShowProgressStrip, and the Episode
 * Overview hub. Includes `index` (warning hue) so consumers can iterate the
 * full pipeline without a separate special case.
 */

export type StageKey = "transcribe" | "correct" | "translate" | "synth" | "index";

export interface StageClasses {
  bg: string;
  text: string;
  border: string;
  /** Left-border accent (`border-l-*/60`) — used by StageCard hub. */
  borderL: string;
  /** Solid fill (used for status dots) — full saturation, no /15 alpha. */
  dot: string;
}

export const STAGE_CLASSES: Record<StageKey, StageClasses> = {
  transcribe: { bg: "bg-stage-transcribe/15", text: "text-stage-transcribe", border: "border-stage-transcribe", borderL: "border-l-stage-transcribe/60", dot: "bg-stage-transcribe" },
  correct:    { bg: "bg-stage-correct/15",    text: "text-stage-correct",    border: "border-stage-correct",    borderL: "border-l-stage-correct/60",    dot: "bg-stage-correct"    },
  translate:  { bg: "bg-stage-translate/15",  text: "text-stage-translate",  border: "border-stage-translate",  borderL: "border-l-stage-translate/60",  dot: "bg-stage-translate"  },
  synth:      { bg: "bg-stage-synth/15",      text: "text-stage-synth",      border: "border-stage-synth",      borderL: "border-l-stage-synth/60",      dot: "bg-stage-synth"      },
  index:      { bg: "bg-warning/15",          text: "text-warning",          border: "border-warning",          borderL: "border-l-warning/60",          dot: "bg-warning"          },
};
