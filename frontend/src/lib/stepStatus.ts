// Re-export from lib/utils so the predicate has a single definition.
import { isEdited } from "./utils";
export { isEdited };

import type { Episode } from "@/api/types";
import type { PipelineInputStep } from "./pipelineInputs";

/** Three-way review status used by Overview StageCards and PipelinePanel
 *  headers. Stays in sync with `STAGE_CARD_CLASSES` vocabulary. */
export type PanelStatus = "ready" | "review" | "none";

/** Status for a step that has a review state (transcribe/correct/translate).
 *  Present + edited → ready, present + raw → review, absent → none. */
export function reviewStatus(present: boolean, provenance: unknown): PanelStatus {
  if (!present) return "none";
  return isEdited(provenance) ? "ready" : "review";
}

/** Status for a terminal step without a review concept (synth/index). */
export function plainStatus(present: boolean): PanelStatus {
  return present ? "ready" : "none";
}

/** Aggregate status across multiple translation langs — "ready" only when
 *  every present lang is edited; "review" if any is raw; "none" when empty. */
export function translationsStatus(
  translations: readonly string[],
  provenance: Record<string, unknown> | null | undefined,
): PanelStatus {
  if (translations.length === 0) return "none";
  return translations.every((l) => isEdited(provenance?.[l])) ? "ready" : "review";
}

/** True if the episode still needs work for the given step. */
export function episodeNeedsStep(ep: Episode, step: PipelineInputStep): boolean {
  switch (step) {
    case "transcribe": return ep.transcribe_status !== "done";
    case "correct":    return ep.correct_status !== "done";
    case "translate":  return ep.translate_status !== "done";
    case "index":      return !ep.indexed;
    default:           return true;
  }
}
