import type { Episode } from "@/api/types";

/** Build an Episode shell for the standalone `/file/$path` route, where
 * no show metadata exists — only an audio file path on disk. */
export function standaloneEpisode(audioFilePath: string): Episode {
  const stem = audioFilePath.split("/").pop()?.replace(/\.[^.]+$/, "") || null;
  return {
    id: audioFilePath,
    title: stem || "Audio",
    stem,
    pub_date: null,
    description: "",
    audio_url: null,
    duration: 0,
    episode_number: null,
    audio_path: audioFilePath,
    downloaded: true,
    removed: false,
    transcribed: false,
    corrected: false,
    indexed: false,
    synthesized: false,
    has_subtitles: false,
    translations: [],
    artwork_url: "",
    provenance: {},
    files: [],
    transcribe_status: "none",
    correct_status: "none",
    translate_status: "none",
  };
}
