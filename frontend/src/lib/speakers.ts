/**
 * Speaker label constants and helpers.
 *
 * Mirrors `src/podcodex/core/_utils.py`:
 *   - NARRATOR_SPEAKER, BREAK_SPEAKER
 *   - UNKNOWN_SPEAKERS frozenset
 *
 * A cross-language drift test (tests/test_frontend_constants_sync.py)
 * parses this file and asserts the values match. Update both sides together.
 */

/** Default label for unattributed segments. */
export const NARRATOR_SPEAKER = "Narrator";

/** Sentinel speaker for break markers inserted between contiguous turns. */
export const BREAK_SPEAKER = "[BREAK]";

/** Sentinel for segments the user marked for removal in the editor. */
export const REMOVE_SPEAKER = "[remove]";

/** Diarization placeholders that count as "no real name yet". */
export const UNKNOWN_SPEAKERS: ReadonlySet<string> = new Set([
  "UNKNOWN",
  "UNK",
  "None",
  "none",
  "",
]);

/** True when this label belongs to a real (named) speaker, not a sentinel. */
export const isRealSpeaker = (sp: string): boolean =>
  !!sp && sp !== BREAK_SPEAKER && !UNKNOWN_SPEAKERS.has(sp);

/**
 * Resolve a segment's speaker for synth display/grouping.
 *   - real name → unchanged
 *   - empty / UNKNOWN / UNK / etc. → NARRATOR_SPEAKER
 *   - [BREAK] → null (caller should drop this segment)
 */
export const resolveSynthSpeaker = (sp: string): string | null => {
  if (sp === BREAK_SPEAKER) return null;
  if (isRealSpeaker(sp)) return sp;
  return NARRATOR_SPEAKER;
};
