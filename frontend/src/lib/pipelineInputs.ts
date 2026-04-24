/**
 * Shared helpers for pipeline-step input-version selection.
 *
 * Used by both the per-episode panels (Correct/Translate) and the batch
 * StepConfigEditor so the "which versions are valid as input" logic lives
 * in one place.
 */

import type { VersionEntry } from "@/api/types";

export type PipelineInputStep = "transcribe" | "correct" | "translate" | "index";

/** Which version steps are valid inputs for each pipeline step, in priority order. */
export const INPUT_STEPS: Record<PipelineInputStep, string[]> = {
  transcribe: [],
  correct: ["transcript"],
  translate: ["corrected", "transcript"],
  index: ["corrected", "transcript"],
};

const INPUT_STEP_SETS: Record<PipelineInputStep, Set<string>> = Object.fromEntries(
  Object.entries(INPUT_STEPS).map(([k, v]) => [k, new Set(v)]),
) as Record<PipelineInputStep, Set<string>>;

/** Filter versions to only those valid as input for a given pipeline step. */
export function filterVersionsForStep(
  versions: VersionEntry[],
  step: PipelineInputStep,
): VersionEntry[] {
  const valid = INPUT_STEP_SETS[step];
  return valid.size > 0 ? versions.filter((v) => !!v.step && valid.has(v.step)) : versions;
}
