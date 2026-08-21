import type {
  AppConfig,
  BroadcastPreviewOut,
  CreateFromRSSResponse,
  CreateFromYouTubeResponse,
  CreateLocalShowResponse,
  Episode,
  EpisodeSpeakersResponse,
  EpisodeStatus,
  FilesImportResponse,
  PipelineAppDefaults,
  PipelineConfig,
  PodcastSearchResult,
  RSSEpisodeOut,
  ShowMeta,
  ShowSummary,
  SpeakerRosterResponse,
  TaskResponse,
} from "./types";
import { ApiError, json, rawFetch } from "./client";

const enc = encodeURIComponent;

// ── Config ──────────────────────────────────

export const getConfig = () => json<AppConfig>("/api/config");
export const getPipelineConfig = () => json<PipelineConfig>("/api/pipeline-config");

export const updateConfig = (cfg: AppConfig) =>
  json<AppConfig>("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });

/** Persist the app-wide pipeline defaults (Settings → Pipeline). Its own
 *  endpoint so a defaults save can't carry stale copies of other config
 *  scalars. */
export const putPipelineDefaults = (defaults: PipelineAppDefaults) =>
  json<PipelineAppDefaults>("/api/config/pipeline-defaults", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(defaults),
  });

export interface FfmpegValidateResponse {
  ok: boolean;
  path: string | null;
  version: string;
  error: string;
}

export const validateFfmpegPath = (path: string) =>
  json<FfmpegValidateResponse>("/api/config/validate-ffmpeg", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });

// ── Podcast search ──────────────────────────

export const searchPodcasts = (query: string, limit = 8) =>
  json<PodcastSearchResult[]>(`/api/podcasts/search?q=${enc(query)}&limit=${limit}`);

// ── Shows ───────────────────────────────────

export const listShows = () => json<ShowSummary[]>("/api/shows/");

export const createFromRSS = (rssUrl: string, savePath: string, folderName?: string, artworkUrl?: string, name?: string, language?: string) =>
  json<CreateFromRSSResponse>("/api/shows/from-rss", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rss_url: rssUrl, save_path: savePath, folder_name: folderName || "", artwork_url: artworkUrl || "", name: name || "", language: language || "" }),
  });

export const registerShow = (path: string) =>
  json<{ status: string; path: string }>("/api/shows/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });

export const importLocalFile = (filePath: string, name?: string, folder?: string) =>
  json<FilesImportResponse>("/api/shows/files/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_path: filePath, name: name ?? null, folder: folder ?? null }),
  });

export const createLocalShow = (name: string) =>
  json<CreateLocalShowResponse>("/api/shows/create-local", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });

/** The suggested free name from an import-collision 409, else null. */
export function conflictSuggestion(err: unknown): string | null {
  if (!(err instanceof ApiError) || err.status !== 409) return null;
  const suggested = (err.body as { detail?: { suggested?: unknown } } | null)
    ?.detail?.suggested;
  return typeof suggested === "string" ? suggested : null;
}

/** Remove the show's cover. On a feed-backed show the next refresh restores
 *  the feed's own artwork; on a local show the UI falls back to the default. */
export const deleteShowArtwork = (folder: string) =>
  json<{ status: string }>(`/api/shows/artwork?show_folder=${enc(folder)}`, {
    method: "DELETE",
  });

export async function uploadShowArtwork(folder: string, file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  await rawFetch(`/api/shows/artwork?show_folder=${enc(folder)}`, {
    method: "POST",
    body: form,
  });
}

export const getShowMeta = (folder: string) =>
  json<ShowMeta>(`/api/shows/${enc(folder)}/meta`);

export const getSpeakerRoster = (folder: string) =>
  json<SpeakerRosterResponse>(`/api/shows/${enc(folder)}/speakers/roster`);

export const getEpisodeSpeakers = (folder: string, stem: string) =>
  json<EpisodeSpeakersResponse>(
    `/api/shows/${enc(folder)}/episode/${enc(stem)}/speakers`,
  );

export const previewBroadcastNumber = (folder: string, pattern: string) =>
  json<BroadcastPreviewOut>(
    `/api/shows/${enc(folder)}/broadcast-preview?pattern=${enc(pattern)}`,
  );

/** Writable half of ShowMeta: the server derives `accepts_imports` and
 *  `last_feed_update` per request and ignores them on PUT, so callers must
 *  not have to invent values for them. */
export type ShowMetaUpdate = Omit<ShowMeta, "accepts_imports" | "last_feed_update">;

export const updateShowMeta = (folder: string, meta: ShowMetaUpdate) =>
  json<{ status: string }>(`/api/shows/${enc(folder)}/meta`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(meta),
  });

export const moveShow = (folder: string, newPath: string, moveFiles: boolean) =>
  json<{ status: string; new_path: string }>(`/api/shows/${enc(folder)}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ new_path: newPath, move_files: moveFiles }),
  });

/** Outcome of a whole-episode delete, per store.
 *
 *  `status: "partial"` means nothing was fully removed and the episode is
 *  still listed: either the search index could not be reached (in which case
 *  nothing at all was touched) or a file could not be removed. Retrying is
 *  always safe and is the intended recovery. */
export interface DeleteEpisodeResult {
  status: "deleted" | "partial";
  collections: number;
  output_dir_removed: boolean;
  audio_removed: boolean;
  db_row_removed: boolean;
  warnings: string[];
}

/** Delete an episode outright: chunks, output dir, audio copy, DB row.
 *
 *  Distinct from `deleteAudioFile`, which only frees disk and leaves the
 *  transcripts in place. */
export const deleteEpisode = (folder: string, stem: string) =>
  json<DeleteEpisodeResult>(`/api/shows/${enc(folder)}/episodes/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stem }),
  });

export const deleteShow = (folder: string, deleteFiles = false) =>
  json<{ status: string; files_deleted: boolean }>(`/api/shows/${enc(folder)}/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ delete_files: deleteFiles }),
  });

// ── Episodes (unified: local + RSS merged) ──

export const getEpisodes = (folder: string) =>
  json<Episode[]>(`/api/shows/${enc(folder)}/unified`);

/** Live pipeline state only, keyed by stem. The cheap poll counterpart to
 *  `getEpisodes`. */
export const getEpisodeStatus = (folder: string) =>
  json<EpisodeStatus[]>(`/api/shows/${enc(folder)}/status`);

// ── RSS actions ─────────────────────────────

export const refreshRSS = (folder: string) =>
  json<{ status: string }>(`/api/shows/${enc(folder)}/rss/fetch`, {
    method: "POST",
  });

export const downloadEpisodes = (folder: string, guids: string[], force = false) =>
  json<TaskResponse>(
    `/api/shows/${enc(folder)}/rss/download${force ? "?force=true" : ""}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(guids),
    },
  );

// ── YouTube actions ────────────────────────

export const createFromYouTube = (
  youtubeUrl: string,
  savePath: string,
  folderName?: string,
  artworkUrl?: string,
  name?: string,
  language?: string,
) =>
  json<CreateFromYouTubeResponse>("/api/shows/from-youtube", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      youtube_url: youtubeUrl,
      save_path: savePath,
      folder_name: folderName || "",
      artwork_url: artworkUrl || "",
      name: name || "",
      language: language || "",
    }),
  });

export const refreshYouTube = (folder: string) =>
  json<RSSEpisodeOut[]>(`/api/shows/${enc(folder)}/youtube/fetch`, {
    method: "POST",
  });

export const downloadYouTubeEpisodes = (
  folder: string,
  videoIds?: string[],
  importSubs = false,
  subLang = "en",
  force = false,
) =>
  json<TaskResponse>(
    `/api/shows/${enc(folder)}/youtube/download${force ? "?force=true" : ""}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_ids: videoIds ?? null,
        import_subs: importSubs,
        sub_lang: subLang,
      }),
    },
  );

export const importYouTubeSubs = (
  folder: string,
  videoIds: string[],
  lang = "en",
) =>
  json<TaskResponse>(`/api/shows/${enc(folder)}/youtube/import-subs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_ids: videoIds, lang }),
  });
