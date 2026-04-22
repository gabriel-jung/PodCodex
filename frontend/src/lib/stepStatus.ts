/** Matches backend `is_edited`: a version counts as reviewed when it was
 *  either saved through the editor (`manual_edit`) or produced by the
 *  validated-output path (`type === "validated"`, e.g. an applied manual
 *  LLM pass). Raw model output does not count. */
export function isEdited(provenance: unknown): boolean {
  const p = provenance as { manual_edit?: unknown; type?: unknown } | undefined;
  return p?.manual_edit === true || p?.type === "validated";
}

/** Freshness status returned by the backend for transcribe/correct/translate.
 *  Freshness = "does the saved version's params still match the effective
 *  defaults?", independent of review state (see `isEdited`). */
export type BackendStepStatus = "none" | "outdated" | "done";
