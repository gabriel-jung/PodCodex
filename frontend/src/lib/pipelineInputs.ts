/**
 * Shared helpers for pipeline-step input-version selection.
 *
 * Used by both the per-episode panels (Correct/Translate) and the batch
 * StepConfigEditor so the "which versions are valid as input" logic lives
 * in one place.
 */

import type { VersionEntry } from "@/api/types";
import { isEdited } from "@/lib/utils";

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

/** Order versions for "default pick" across multiple input steps, the same
 *  order the backend reads them in (`core/versions.py` `_source_candidates`
 *  and the canonical ladder). Priority: the verified version (when present
 *  and its step is a valid input for `step`), then step rank, then within a
 *  step that step's own rule: edited first for corrected versions, strictly
 *  newest for transcripts (a fresh transcribe supersedes an older edit).
 *  Example for `index` (steps = ["corrected", "transcript"]):
 *    verified > edited corrected > corrected > newest transcript.
 *  Stable: within a tier, input order (newest first) is preserved. */
export function sortVersionsForDefault(
  versions: VersionEntry[],
  step?: PipelineInputStep,
  verified?: { step: string; version_id: string } | null,
): VersionEntry[] {
  const priority = step ? INPUT_STEPS[step] : [];
  const validSteps = step ? INPUT_STEP_SETS[step] : null;
  const verifiedActive =
    !!verified && (!validSteps || validSteps.has(verified.step));
  const rank = (v: VersionEntry) => {
    if (verifiedActive && v.id === verified!.version_id) return -1;
    const stepIdx = v.step ? priority.indexOf(v.step) : -1;
    const stepKey = stepIdx === -1 ? priority.length : stepIdx;
    // Transcripts have no edited tier (see the doc comment above).
    const editedTier = v.step === "transcript" ? 0 : isEdited(v) ? 0 : 1;
    return stepKey * 2 + editedTier;
  };
  return [...versions].sort((a, b) => rank(a) - rank(b));
}
