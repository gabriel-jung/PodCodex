import type {
  AppConfig,
  CreateFromRSSResponse,
  CreateFromYouTubeResponse,
  Episode,
  PipelineConfig,
  PipelineDefaults,
  PodcastSearchResult,
  RSSEpisodeOut,
  ShowMeta,
  ShowSummary,
  TaskResponse,
} from "./types";
import { json } from "./base";

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

// ── Podcast search ──────────────────────────

export const searchPodcasts = (query: string, limit = 8) =>
  json<PodcastSearchResult[]>(`/api/podcasts/search?q=${enc(query)}&limit=${limit}`);

// ── Shows ───────────────────────────────────

export const listShows = () => json<ShowSummary[]>("/api/shows");

export const createFromRSS = (rssUrl: string, savePath: string, folderName?: string, artworkUrl?: string, name?: string) =>
  json<CreateFromRSSResponse>("/api/shows/from-rss", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rss_url: rssUrl, save_path: savePath, folder_name: folderName || "", artwork_url: artworkUrl || "", name: name || "" }),
  });

export const registerShow = (path: string) =>
  json<{ status: string; path: string }>("/api/shows/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });

export const getShowMeta = (folder: string) =>
  json<ShowMeta>(`/api/shows/${enc(folder)}/meta`);

export const updateShowMeta = (folder: string, meta: ShowMeta) =>
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

export const deleteShow = (folder: string, deleteFiles = false) =>
  json<{ status: string; files_deleted: boolean }>(`/api/shows/${enc(folder)}/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ delete_files: deleteFiles }),
  });

// ── Episodes (unified: local + RSS merged) ──

export const getEpisodes = (folder: string, defaults?: PipelineDefaults) => {
  const params = defaults ? `?defaults=${enc(JSON.stringify(defaults))}` : "";
  return json<Episode[]>(`/api/shows/${enc(folder)}/unified${params}`);
};

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
) =>
  json<TaskResponse>(`/api/shows/${enc(folder)}/youtube/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_ids: videoIds ?? null,
      import_subs: importSubs,
      sub_lang: subLang,
    }),
  });

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
