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

/** Placeholder written when diarization is skipped. Deliberately not a
 *  plausible human name; see the Python constant for why. */
export const NARRATOR_SPEAKER = "NoDiarization";

/** The value NARRATOR_SPEAKER had before 0.2.10, still treated as a
 *  placeholder so libraries written by older versions keep working. */
export const LEGACY_NARRATOR_SPEAKER = "Narrator";

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

/** Matches raw diarizer output like SPEAKER_00, SPEAKER_12. */
const DIARIZER_DEFAULT_RE = /^SPEAKER_\d+$/;

/** True for a diarizer placeholder id ("", SPEAKER_00): a turn the diarizer
 *  separated but nobody has named yet. Note this excludes NARRATOR_SPEAKER,
 *  which means "never diarized at all" rather than "unnamed turn". */
export const isDiarizerPlaceholder = (sp: string): boolean =>
  sp === "" || DIARIZER_DEFAULT_RE.test(sp);

/** True when the label is a placeholder rather than a name someone chose. */
export const isDefaultSpeaker = (sp: string): boolean =>
  sp === NARRATOR_SPEAKER ||
  sp === LEGACY_NARRATOR_SPEAKER ||
  UNKNOWN_SPEAKERS.has(sp) ||
  DIARIZER_DEFAULT_RE.test(sp);

/**
 * True when a speaker list carries no information: exactly one speaker, and
 * it is a placeholder.
 *
 * That is what a non-diarized transcript or a subtitle import without
 * `<v Speaker>` tags produces, and repeating "Narrator" on every row, in the
 * episode list and in the airtime line then says nothing. A single *named*
 * speaker is different: the user chose that name, so it stays visible.
 */
export const isSoloDefaultSpeaker = (names: readonly string[]): boolean =>
  names.length === 1 && isDefaultSpeaker(names[0]);

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
