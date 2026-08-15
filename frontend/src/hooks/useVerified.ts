/** Hook for setting/clearing the per-episode verified-version pointer.
 *
 *  The verified pointer is a singleton per episode (lives in pipeline_db).
 *  Setting it on any version replaces the previous pointer; passing
 *  `(null, null)` clears it. Downstream consumers — translate, index,
 *  synthesize, RAG, bot retrieval — read the pointer as their canonical
 *  source ahead of the latest-version cascade.
 */

import { useMutation } from "@tanstack/react-query";
import { setVerifiedVersion } from "@/api/search";
import { invalidateSpeakerViews } from "@/api/cacheInvalidation";
import { queryKeys } from "@/api/queryKeys";

export type VerifiableStep = "transcript" | "corrected";

export function useSetVerifiedVersion(
  audioPath: string | null | undefined,
  outputDir: string | null | undefined,
) {
  return useMutation({
    mutationFn: (input: { step: VerifiableStep | null; versionId: string | null }) =>
      setVerifiedVersion(audioPath, outputDir, input.step, input.versionId),
    // The pointer flows through unified_episodes and the all-versions list,
    // changes the resolved best-source for downstream panels, and shifts the
    // per-show verified_count surfaced in ShowProgressStrip. It also decides
    // the canonical transcript, which is what the speaker views resolve.
    meta: {
      invalidates: [
        queryKeys.episodesAll(),
        queryKeys.allVersions(audioPath ?? outputDir),
        queryKeys.bestSourceSegments(audioPath),
        queryKeys.shows(),
        invalidateSpeakerViews,
      ],
    },
  });
}
