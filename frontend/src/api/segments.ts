import type { AudioSegment } from "@/stores";
import type { Segment } from "./types";
import { json } from "./client";

/** Fetch the canonical source segments for playback surfaces.
 *  Honors the verified pointer (when set) then falls back through
 *  corrected and transcript. Single facility shared with backend
 *  `_resolve_source_segments(auto)` so the audio overlay and panels
 *  cannot disagree. */
export async function getBestSegments(audioPath: string): Promise<Segment[]> {
  const params = new URLSearchParams({ audio_path: audioPath });
  return json<Segment[]>(`/api/shows/best-source-segments?${params}`);
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
