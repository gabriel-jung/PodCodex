import type { BatchRequest, IndexRequest, IndexStatus, SearchRequest, SearchResult, TaskResponse, VersionEntry } from "./types";
import { json } from "./client";

// ── Batch ──────────────────────────────────

export const startBatch = (req: BatchRequest) =>
  json<TaskResponse>("/api/batch/start", {
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

/** Fetch all versions across all steps for an episode (newest first). */
export const getAllVersions = (audioPath?: string | null, outputDir?: string | null) => {
  const params = new URLSearchParams();
  if (audioPath) params.set("audio_path", audioPath);
  if (outputDir) params.set("output_dir", outputDir);
  return json<VersionEntry[]>(`/api/shows/versions?${params}`);
};

export interface EpisodeCollection {
  collection: string;
  model: string;
  chunker: string;
  source: string;
  chunk_count: number;
}

/** Index entries this episode currently lives in (one per collection). */
export const getEpisodeCollections = (audioPath: string, show: string) => {
  const params = new URLSearchParams({ audio_path: audioPath, show });
  return json<EpisodeCollection[]>(`/api/index/episode-collections?${params}`);
};

export const deleteEpisodeCollection = (
  audioPath: string,
  show: string,
  collection: string,
) => {
  const params = new URLSearchParams({
    audio_path: audioPath,
    show,
    collection,
  });
  return json<{ status: string; still_indexed: boolean }>(
    `/api/index/episode?${params}`,
    { method: "DELETE" },
  );
};

export interface InspectChunkSpeakerTurn {
  speaker: string;
  start: number;
  end: number;
  text: string;
}

export interface InspectChunk {
  chunk_index: number;
  start: number;
  end: number;
  dominant_speaker: string;
  text: string;
  source: string;
  speakers?: InspectChunkSpeakerTurn[];
  vector_norm: number;
  vector_zero_frac: number;
  vector_min: number;
  vector_max: number;
  vector_mean: number;
  vector_std: number;
}

export interface InspectSummary {
  dim: number;
  n_chunks: number;
  n_dead_chunks: number;
  n_collapsed_chunks: number;
  n_with_zeros: number;
  zero_warn_threshold: number;
}

export interface InspectResponse {
  collection: string;
  model: string;
  chunking: string;
  episode: string;
  summary: InspectSummary;
  chunks: InspectChunk[];
}

export const getIndexInspect = (
  audioPath: string,
  show: string,
  model: string,
  chunking: string,
) => {
  const params = new URLSearchParams({ audio_path: audioPath, show, model, chunking });
  return json<InspectResponse>(`/api/index/inspect?${params}`);
};

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

export const exactSearch = (req: SearchRequest) =>
  json<SearchResult[]>("/api/search/exact", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const randomQuote = (req: {
  show: string;
  model?: string;
  chunking?: string;
  episode?: string | null;
  episodes?: string[] | null;
  speaker?: string | null;
  source?: string | null;
  pub_date_min?: string | null;
  pub_date_max?: string | null;
}) =>
  json<SearchResult | null>("/api/search/random", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const listIndexedSpeakers = (
  show: string,
  opts: { model?: string; chunking?: string } = {},
) => {
  const qs = new URLSearchParams({ show });
  if (opts.model) qs.set("model", opts.model);
  if (opts.chunking) qs.set("chunking", opts.chunking);
  return json<string[]>(`/api/search/speakers?${qs}`);
};

export const getIndexStats = (show: string = "") =>
  json<{
    collections: { collection: string; model: string; chunking: string; episodes: number; chunks: number; sources: string[] }[];
    total_episodes: number;
    total_chunks: number;
  }>(`/api/search/stats?show=${encodeURIComponent(show)}`);
