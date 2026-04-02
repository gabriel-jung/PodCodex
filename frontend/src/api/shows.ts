import type {
  AppConfig,
  CreateFromRSSResponse,
  Episode,
  PipelineConfig,
  PipelineDefaults,
  PodcastSearchResult,
  ShowMeta,
  ShowSummary,
  TaskResponse,
} from "./types";
import { json } from "./base";

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
  json<PodcastSearchResult[]>(`/api/podcasts/search?q=${encodeURIComponent(query)}&limit=${limit}`);

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
  json<ShowMeta>(`/api/shows/${encodeURIComponent(folder)}/meta`);

export const updateShowMeta = (folder: string, meta: ShowMeta) =>
  json<{ status: string }>(`/api/shows/${encodeURIComponent(folder)}/meta`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(meta),
  });

export const moveShow = (folder: string, newPath: string, moveFiles: boolean) =>
  json<{ status: string; new_path: string }>(`/api/shows/${encodeURIComponent(folder)}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ new_path: newPath, move_files: moveFiles }),
  });

// ── Episodes (unified: local + RSS merged) ──

export const getEpisodes = (folder: string, defaults?: PipelineDefaults) => {
  const params = defaults ? `?defaults=${encodeURIComponent(JSON.stringify(defaults))}` : "";
  return json<Episode[]>(`/api/shows/${encodeURIComponent(folder)}/unified${params}`);
};

// ── RSS actions ─────────────────────────────

export const refreshRSS = (folder: string) =>
  json<{ status: string }>(`/api/shows/${encodeURIComponent(folder)}/rss/fetch`, {
    method: "POST",
  });

export const downloadEpisodes = (folder: string, guids: string[]) =>
  json<TaskResponse>(
    `/api/shows/${encodeURIComponent(folder)}/rss/download`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(guids),
    },
  );
