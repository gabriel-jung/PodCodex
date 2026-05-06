// Re-export from lib/utils so the predicate has a single definition.
import { isEdited } from "./utils";
export { isEdited };

/** Freshness status returned by the backend for transcribe/correct/translate.
 *  Freshness = "does the saved version's params still match the effective
 *  defaults?", independent of review state (see `isEdited`). */
export type BackendStepStatus = "none" | "outdated" | "done";

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
