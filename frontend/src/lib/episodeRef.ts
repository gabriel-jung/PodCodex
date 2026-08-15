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

/** Stable on-disk stem identifier for an episode. Falls back to ``id`` when
 *  ``stem`` is empty or absent (some legacy entries pre-date the stem field
 *  being populated; treating ``""`` as falsy is intentional). */
export function getEpisodeStem(episode: Episode): string {
  return episode.stem || episode.id;
}

/**
 * The path the batch API identifies this episode by.
 *
 * The `.virtual` suffix tells the backend the episode has no audio on disk
 * but does have an output_dir to resume from (subtitle-only imports). This is
 * the key space of `BatchRequest.audio_paths` *and* of its
 * `source_version_ids` map, so anything building either must go through here:
 * keying a version map by `audio_path || id` instead silently loses every
 * subtitle-only episode, and the backend then falls back to its own default
 * version rather than the one the user picked.
 */
export function getEpisodeBatchPath(episode: Episode): string | null {
  if (episode.audio_path) return episode.audio_path;
  if (!episode.output_dir) return null;
  return episode.output_dir.replace(/\/+$/, "") + ".virtual";
}
