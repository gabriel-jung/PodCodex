// Re-export from lib/utils so the predicate has a single definition.
export { isEdited } from "./utils";

/** Freshness status returned by the backend for transcribe/correct/translate.
 *  Freshness = "does the saved version's params still match the effective
 *  defaults?", independent of review state (see `isEdited`). */
export type BackendStepStatus = "none" | "outdated" | "done";
