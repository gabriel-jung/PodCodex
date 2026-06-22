/** Shared helpers for the per-episode verified-version pointer.
 *
 *  A verified pointer marks one transcript/corrected version as the episode's
 *  final source. Keep the predicate and the "what it's for" wording here so the
 *  many UI surfaces that show the marker (StatusChips, EpisodeRow,
 *  ShowProgressStrip, the Overview preview card + versions table,
 *  SegmentContextDialog, the editor's VersionControlBar) stay consistent.
 */

export interface VerifiedPointer {
  step: string;
  version_id: string;
}

/** What the verified version is used for. Shared so the explanatory tooltip
 *  reads the same everywhere. */
export const VERIFIED_CAPTION =
  "canonical source for translate / search / synth";

/** True when `versionId` is the episode's verified version. Pass `step` to also
 *  require the pointer targets that step; version ids are unique per step, so
 *  omitting it is safe (pass it only where the step is cheaply known). */
export function isVerifiedVersion(
  verified: VerifiedPointer | null | undefined,
  versionId: string | null | undefined,
  step?: string,
): boolean {
  if (!verified || !versionId) return false;
  if (step !== undefined && verified.step !== step) return false;
  return verified.version_id === versionId;
}
