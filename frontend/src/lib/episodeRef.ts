import type { Episode } from "@/api/types";

export interface EpisodeSourceRef {
  audioPath: string | null;
  outputDir: string | null;
  /** Single identifier for query keys: audio_path if present, else output_dir. */
  sourceRef: string | null;
  hasSourceRef: boolean;
  /** Episode has output_dir but no audio file (e.g. YouTube subtitle import). */
  noAudio: boolean;
}

export function getEpisodeSourceRef(episode: Episode | null | undefined): EpisodeSourceRef {
  const audioPath = episode?.audio_path ?? null;
  const outputDir = episode?.output_dir ?? null;
  const sourceRef = audioPath ?? outputDir;
  return {
    audioPath,
    outputDir,
    sourceRef,
    hasSourceRef: !!sourceRef,
    noAudio: !audioPath && !!outputDir,
  };
}
