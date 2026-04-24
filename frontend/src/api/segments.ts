import type { AudioSegment } from "@/stores";
import type { Segment } from "./types";
import { getSegments } from "./transcribe";
import { getCorrectSegments } from "./correct";

/** Fetch the latest reviewed segments for playback surfaces.
 *  Prefers corrected; falls back to transcribed when corrected is
 *  missing or empty. */
export async function getBestSegments(audioPath: string): Promise<Segment[]> {
  try {
    const corrected = await getCorrectSegments(audioPath);
    if (Array.isArray(corrected) && corrected.length > 0) return corrected;
  } catch { /* fall through */ }
  return getSegments(audioPath);
}

export function toAudioSegments(segments: Segment[]): AudioSegment[] {
  return segments
    .filter((s): s is Segment & { start: number; end: number } =>
      typeof s.start === "number" && typeof s.end === "number")
    .map((s) => ({
      start: s.start,
      end: s.end,
      speaker: s.speaker ?? "",
      text: s.text ?? "",
    }));
}
