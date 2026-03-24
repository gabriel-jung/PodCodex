/** Typed API client for the PodCodex backend. */

import type {
  AppConfig,
  CreateFromRSSResponse,
  DownloadResult,
  Episode,
  HealthResponse,
  PodcastSearchResult,
  Segment,
  ShowMeta,
  ShowSummary,
} from "./types";

const BASE = "";  // proxied by Vite in dev, same origin in prod

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// ── Health ──────────────────────────────────

export const getHealth = () => json<HealthResponse>("/api/health");

// ── Config ──────────────────────────────────

export const getConfig = () => json<AppConfig>("/api/config");

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

export const createFromRSS = (rssUrl: string, savePath: string, folderName?: string, artworkUrl?: string) =>
  json<CreateFromRSSResponse>("/api/shows/from-rss", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rss_url: rssUrl, save_path: savePath, folder_name: folderName || "", artwork_url: artworkUrl || "" }),
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

// ── Episodes (unified: local + RSS merged) ──

export const getEpisodes = (folder: string) =>
  json<Episode[]>(`/api/shows/${encodeURIComponent(folder)}/unified`);

// ── RSS actions ─────────────────────────────

export const refreshRSS = (folder: string) =>
  json<unknown>(`/api/shows/${encodeURIComponent(folder)}/rss/fetch`, {
    method: "POST",
  });

export const downloadEpisodes = (folder: string, guids: string[]) =>
  json<DownloadResult[]>(
    `/api/shows/${encodeURIComponent(folder)}/rss/download`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(guids),
    },
  );

// ── Transcripts ─────────────────────────────

export const getSegments = (audioPath: string) =>
  json<Segment[]>(`/api/transcribe/segments?audio_path=${encodeURIComponent(audioPath)}`);

export const getSegmentsRaw = (audioPath: string) =>
  json<Segment[]>(`/api/transcribe/segments/raw?audio_path=${encodeURIComponent(audioPath)}`);

// ── Filesystem ──────────────────────────────

export interface DirEntry {
  name: string;
  path: string;
  is_show: boolean;
  has_audio: boolean;
}

export interface FileEntry {
  name: string;
  path: string;
}

export interface DirListing {
  path: string;
  parent: string | null;
  dirs: DirEntry[];
  files: FileEntry[];
  error: string | null;
}

export const listDirectory = (path: string, showFiles = false) =>
  json<DirListing>(`/api/fs/list?path=${encodeURIComponent(path)}&show_files=${showFiles}`);

// ── Audio ───────────────────────────────────

export const audioFileUrl = (path: string) =>
  `/api/audio/file?path=${encodeURIComponent(path)}`;

export const audioClipUrl = (path: string, start: number, end: number, padding = 0.3) =>
  `/api/audio/clip?path=${encodeURIComponent(path)}&start=${start}&end=${end}&padding=${padding}`;
