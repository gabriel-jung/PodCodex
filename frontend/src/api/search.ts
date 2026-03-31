import type { BatchRequest, IndexRequest, IndexStatus, SearchRequest, SearchResult, SyncRequest, TaskResponse } from "./types";
import { json } from "./base";

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
