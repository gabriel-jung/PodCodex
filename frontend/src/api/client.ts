/** Typed API client for the PodCodex backend. */

import type {
  AppConfig,
  CreateFromRSSResponse,
  DownloadResult,
  Episode,
  HealthResponse,
  PodcastSearchResult,
  PolishRequest,
  Segment,
  ShowMeta,
  ShowSummary,
  TaskResponse,
  TranscribeRequest,
  TranslateRequest,
  VersionInfo,
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

export const saveSegments = (audioPath: string, segments: Segment[]) =>
  json<{ status: string; count: number }>(`/api/transcribe/segments?audio_path=${encodeURIComponent(audioPath)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(segments),
  });

export const getTranscribeVersionInfo = (audioPath: string) =>
  json<VersionInfo>(`/api/transcribe/version-info?audio_path=${encodeURIComponent(audioPath)}`);

export const getSpeakerMap = (audioPath: string) =>
  json<Record<string, string>>(`/api/transcribe/speaker-map?audio_path=${encodeURIComponent(audioPath)}`);

export const saveSpeakerMap = (audioPath: string, mapping: Record<string, string>) =>
  json<{ status: string }>(`/api/transcribe/speaker-map?audio_path=${encodeURIComponent(audioPath)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(mapping),
  });

export const startTranscribe = (req: TranscribeRequest) =>
  json<TaskResponse>("/api/transcribe/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export async function uploadTranscript(audioPath: string, file: File): Promise<{ status: string; count: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`/api/transcribe/upload?audio_path=${encodeURIComponent(audioPath)}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// ── Polish ─────────────────────────────────

export const getPolishSegments = (audioPath: string) =>
  json<Segment[]>(`/api/polish/segments?audio_path=${encodeURIComponent(audioPath)}`);

export const getPolishSegmentsRaw = (audioPath: string) =>
  json<Segment[]>(`/api/polish/segments/raw?audio_path=${encodeURIComponent(audioPath)}`);

export const savePolishSegments = (audioPath: string, segments: Segment[]) =>
  json<{ status: string; count: number }>(`/api/polish/segments?audio_path=${encodeURIComponent(audioPath)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(segments),
  });

export const getPolishVersionInfo = (audioPath: string) =>
  json<VersionInfo>(`/api/polish/version-info?audio_path=${encodeURIComponent(audioPath)}`);

export const startPolish = (req: PolishRequest) =>
  json<TaskResponse>("/api/polish/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const getPolishManualPrompts = (params: {
  audio_path: string;
  context?: string;
  source_lang?: string;
  batch_minutes?: number;
  engine?: string;
}) =>
  json<{ batch_index: number; prompt: string; segment_count: number }[]>(
    "/api/polish/manual-prompts",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params) },
  );

export const applyPolishManual = (params: { audio_path: string; corrections: unknown[] }) =>
  json<{ status: string; count: number }>("/api/polish/apply-manual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

// ── Translate ──────────────────────────────

export const getTranslateSegments = (audioPath: string, lang: string) =>
  json<Segment[]>(`/api/translate/segments?audio_path=${encodeURIComponent(audioPath)}&lang=${encodeURIComponent(lang)}`);

export const getTranslateSegmentsRaw = (audioPath: string, lang: string) =>
  json<Segment[]>(`/api/translate/segments/raw?audio_path=${encodeURIComponent(audioPath)}&lang=${encodeURIComponent(lang)}`);

export const saveTranslateSegments = (audioPath: string, lang: string, segments: Segment[]) =>
  json<{ status: string; count: number }>(`/api/translate/segments?audio_path=${encodeURIComponent(audioPath)}&lang=${encodeURIComponent(lang)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(segments),
  });

export const getTranslateVersionInfo = (audioPath: string, lang: string) =>
  json<VersionInfo>(`/api/translate/version-info?audio_path=${encodeURIComponent(audioPath)}&lang=${encodeURIComponent(lang)}`);

export const getTranslateLanguages = (audioPath: string) =>
  json<string[]>(`/api/translate/languages?audio_path=${encodeURIComponent(audioPath)}`);

export const startTranslate = (req: TranslateRequest) =>
  json<TaskResponse>("/api/translate/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const getTranslateManualPrompts = (params: {
  audio_path: string;
  context?: string;
  source_lang?: string;
  target_lang?: string;
  batch_minutes?: number;
}) =>
  json<{ batch_index: number; prompt: string; segment_count: number }[]>(
    "/api/translate/manual-prompts",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params) },
  );

export const applyTranslateManual = (params: { audio_path: string; lang: string; corrections: unknown[] }) =>
  json<{ status: string; count: number }>("/api/translate/apply-manual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

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

export const deleteAudioFile = (path: string) =>
  json<{ status: string; path: string }>(`/api/audio/file?path=${encodeURIComponent(path)}`, {
    method: "DELETE",
  });
