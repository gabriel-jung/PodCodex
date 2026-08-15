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

// ── Per-step episode-list filtering ────────────────────────────────────────

/** Steps the episode list can filter on. */
export type StepFilterStep =
  | "transcribe"
  | "correct"
  | "translate"
  | "index"
  | "synthesize";

/** Per-step state an episode can be in. */
export type StepFilterState = "missing" | "done" | "raw" | "edited" | "outdated";

/** Steps with no review concept: content either exists or it doesn't. */
const TERMINAL_STEPS: readonly StepFilterStep[] = ["index", "synthesize"];

/** States offered for a step. Terminal steps have no raw/edited/outdated. */
export function statesForStep(step: StepFilterStep): readonly StepFilterState[] {
  return TERMINAL_STEPS.includes(step)
    ? (["missing", "done"] as const)
    : (["missing", "done", "raw", "edited", "outdated"] as const);
}

/** Resolve a step to (present, edited, outdated) for one episode.
 *
 * `lang` narrows `translate` to a single language; without it, translate is
 * aggregated across every language the episode has (see translationsStatus).
 */
function resolveStep(
  ep: Episode,
  step: StepFilterStep,
  lang: string,
): { present: boolean; edited: boolean; outdated: boolean } {
  const prov = (ep.provenance ?? {}) as Record<string, unknown>;
  switch (step) {
    case "transcribe":
      return {
        present: ep.transcribed,
        edited: isEdited(prov.transcript),
        outdated: ep.transcribe_status === "outdated",
      };
    case "correct":
      return {
        present: ep.corrected,
        edited: isEdited(prov.corrected),
        outdated: ep.correct_status === "outdated",
      };
    case "translate": {
      const langs = ep.translations ?? [];
      const present = lang ? langs.includes(lang) : langs.length > 0;
      const edited = lang
        ? isEdited(prov[lang])
        : translationsStatus(langs, prov) === "ready";
      // Outdated is a per-episode aggregate; the backend does not break it
      // down per language.
      return { present, edited, outdated: ep.translate_status === "outdated" };
    }
    case "index":
      return { present: ep.indexed, edited: false, outdated: false };
    case "synthesize":
      return { present: ep.synthesized, edited: false, outdated: false };
  }
}

/**
 * True when the episode is in `state` for `step`.
 *
 * Drives the episode-list step filter. Reads the same `*_status` fields and
 * provenance the pipeline buttons and StageCards use, so "needs correcting"
 * here and a lit-up Correct button always agree.
 */
export function matchesStepFilter(
  ep: Episode,
  step: StepFilterStep,
  state: StepFilterState,
  lang = "",
): boolean {
  const { present, edited, outdated } = resolveStep(ep, step, lang);
  switch (state) {
    case "missing":  return !present;
    case "done":     return present;
    case "raw":      return present && !edited;
    case "edited":   return present && edited;
    case "outdated": return outdated;
  }
}
