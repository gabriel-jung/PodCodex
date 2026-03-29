/** Typed API client for the PodCodex backend. */

import type {
  AppConfig,
  AssembleRequest,
  CreateFromRSSResponse,
  DirListing,
  DownloadResult,
  Episode,
  ExtrasResponse,
  ExtractVoicesRequest,
  GenerateRequest,
  GeneratedSegment,
  HealthResponse,
  IndexRequest,
  IndexStatus,
  ModelsResponse,
  PipelineConfig,
  PodcastSearchResult,
  PolishRequest,
  SearchRequest,
  SearchResult,
  Segment,
  ShowMeta,
  SyncRequest,
  ShowSummary,
  SynthesisStatus,
  TaskResponse,
  TranscribeRequest,
  TranslateRequest,
  VersionInfo,
  VoiceSample,
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

// ── System ─────────────────────────────────

export const getExtras = () => json<ExtrasResponse>("/api/system/extras");

export const getActiveTask = (audioPath: string) =>
  json<{
    task_id: string;
    status: string;
    progress: number;
    message: string;
    steps?: string[];
    log?: string[];
    result?: Record<string, unknown>;
    error?: string;
  } | null>(
    `/api/tasks/active?audio_path=${encodeURIComponent(audioPath)}`,
  );

export const installExtra = (extra: string) =>
  json<TaskResponse>("/api/system/install-extra", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ extra }),
  });

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

// ── Episodes (unified: local + RSS merged) ──

export const getEpisodes = (folder: string) =>
  json<Episode[]>(`/api/shows/${encodeURIComponent(folder)}/unified`);

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

export const skipPolish = (audioPath: string) =>
  json<{ status: string; count: number }>("/api/polish/skip", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_path: audioPath }),
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

export const listDirectory = (path: string, showFiles = false) =>
  json<DirListing>(`/api/fs/list?path=${encodeURIComponent(path)}&show_files=${showFiles}`);

export const createDirectory = (path: string, name: string) =>
  json<{ path: string | null; error: string | null }>(
    `/api/fs/mkdir?path=${encodeURIComponent(path)}&name=${encodeURIComponent(name)}`,
    { method: "POST" },
  );

// ── Audio ───────────────────────────────────

export const audioFileUrl = (path: string) =>
  `/api/audio/file?path=${encodeURIComponent(path)}`;

export const deleteAudioFile = (path: string) =>
  json<{ status: string; path: string }>(`/api/audio/file?path=${encodeURIComponent(path)}`, {
    method: "DELETE",
  });

// ── Synthesize ─────────────────────────────

export const getSynthesisStatus = (audioPath: string) =>
  json<SynthesisStatus>(`/api/synthesize/status?audio_path=${encodeURIComponent(audioPath)}`);

export const startExtractVoices = (req: ExtractVoicesRequest) =>
  json<TaskResponse>("/api/synthesize/extract-voices", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const getVoiceSamples = (audioPath: string) =>
  json<Record<string, VoiceSample[]>>(`/api/synthesize/voice-samples?audio_path=${encodeURIComponent(audioPath)}`);

export async function uploadVoiceSample(audioPath: string, speaker: string, file: File): Promise<VoiceSample & { speaker: string }> {
  const form = new FormData();
  form.append("audio_path", audioPath);
  form.append("speaker", speaker);
  form.append("file", file);
  const res = await fetch("/api/synthesize/upload-sample", { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

export const extractSelectedSamples = (audioPath: string, selections: { speaker: string; start: number; end: number; text: string }[]) =>
  json<{ status: string; speakers: number; total_samples: number; samples: Record<string, VoiceSample[]> }>(
    "/api/synthesize/extract-selected",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio_path: audioPath, selections }),
    },
  );

export const startGenerateTTS = (req: GenerateRequest) =>
  json<TaskResponse>("/api/synthesize/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const getGeneratedSegments = (audioPath: string) =>
  json<GeneratedSegment[]>(`/api/synthesize/generated-segments?audio_path=${encodeURIComponent(audioPath)}`);

export const assembleEpisode = (req: AssembleRequest) =>
  json<{ path: string; duration: number }>("/api/synthesize/assemble", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

// ── Index ──────────────────────────────────

export const getIndexConfig = () =>
  json<{
    models: Record<string, { label: string; description: string }>;
    chunking_strategies: Record<string, string>;
    defaults: { model: string; chunking: string; chunk_size: number; threshold: number };
  }>("/api/index/config");

export const getIndexStatus = (audioPath: string, show: string) =>
  json<{ combinations: IndexStatus[]; db_exists: boolean }>(
    `/api/index/status?audio_path=${encodeURIComponent(audioPath)}&show=${encodeURIComponent(show)}`,
  );

export const startIndex = (req: IndexRequest) =>
  json<TaskResponse>("/api/index/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

// ── Search ─────────────────────────────────

export const getSearchConfig = () =>
  json<{
    models: Record<string, { label: string; description: string }>;
    chunking_strategies: Record<string, string>;
    defaults: { model: string; chunking: string; alpha: number; top_k: number };
  }>("/api/search/config");

export const searchQuery = (req: SearchRequest) =>
  json<SearchResult[]>("/api/search/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const syncToQdrant = (req: SyncRequest) =>
  json<{ task_id: string }>("/api/search/sync", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const exactSearch = (req: SearchRequest) =>
  json<SearchResult[]>("/api/search/exact", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const randomQuote = (req: {
  folder?: string | null;
  audio_path?: string | null;
  show: string;
  model?: string;
  chunking?: string;
  episode?: string | null;
  speaker?: string | null;
}) =>
  json<SearchResult | null>("/api/search/random", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const getIndexStats = (folder: string, show: string = "") =>
  json<{
    collections: { collection: string; model: string; chunking: string; episodes: number; chunks: number }[];
    total_episodes: number;
    total_chunks: number;
  }>(`/api/search/stats?folder=${encodeURIComponent(folder)}&show=${encodeURIComponent(show)}`);

// ── Models ────────────────────────────────

export const getModels = () => json<ModelsResponse>("/api/models");

export const deleteModel = (modelId: string) =>
  json<{ status: string; model_id: string }>(`/api/models/${encodeURIComponent(modelId)}`, {
    method: "DELETE",
  });

// ── Export ─────────────────────────────────

export const exportTextUrl = (audioPath: string, source = "transcript", outputDir?: string) => {
  const params = new URLSearchParams({ audio_path: audioPath, source });
  if (outputDir) params.set("output_dir", outputDir);
  return `/api/export/text?${params}`;
};

export const exportSrtUrl = (audioPath: string, source = "transcript", outputDir?: string) => {
  const params = new URLSearchParams({ audio_path: audioPath, source });
  if (outputDir) params.set("output_dir", outputDir);
  return `/api/export/srt?${params}`;
};

export const exportVttUrl = (audioPath: string, source = "transcript", outputDir?: string) => {
  const params = new URLSearchParams({ audio_path: audioPath, source });
  if (outputDir) params.set("output_dir", outputDir);
  return `/api/export/vtt?${params}`;
};

export const exportZipUrl = (audioPath: string, outputDir?: string) => {
  const params = new URLSearchParams({ audio_path: audioPath });
  if (outputDir) params.set("output_dir", outputDir);
  return `/api/export/zip?${params}`;
};
