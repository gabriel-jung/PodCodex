/**
 * TypeScript types for the PodCodex API.
 *
 * Generated types (from Pydantic models) are re-exported from generated-types.ts.
 * Frontend-only types (not backed by Pydantic) are defined here.
 */

// ── Re-export all generated types ────────────────────────
export type {
  AppConfig,
  AssembleRequest,
  BatchRequest,
  CreateFromRSSRequest,
  CreateFromRSSResponse,
  CreateFromYouTubeRequest,
  CreateFromYouTubeResponse,
  EpisodeOut,
  ExtractVoicesRequest,
  GenerateRequest,
  IndexRequest,
  PipelineDefaultsSchema as PipelineDefaults,
  PolishApplyManualRequest,
  PolishManualPromptsRequest,
  PolishRequest,
  RSSEpisodeOut,
  SearchRequest,
  Segment,
  ShowMeta,
  ShowSummary,
  TaskResponse,
  TranscribeRequest,
  TranslateApplyManualRequest,
  TranslateManualPromptsRequest,
  TranslateRequest,
  UnifiedEpisodeOut,
} from "./generated-types";

// ── Frontend-only types (not backed by Pydantic models) ──

export interface HealthResponse {
  status: string;
  capabilities: Record<string, boolean>;
}

export interface ExtraInfo {
  description: string;
  installed: boolean;
  capabilities: string[];
}

export interface ExtrasResponse {
  extras: Record<string, ExtraInfo>;
  capabilities: Record<string, boolean>;
}

/** Unified episode used throughout the frontend (aliased from generated). */
export type Episode = import("./generated-types").UnifiedEpisodeOut;

export interface DownloadResult {
  stem: string;
  audio_path: string | null;
  status: "downloaded" | "exists" | "failed" | "no_audio";
}

export interface PodcastSearchResult {
  name: string;
  artist: string;
  feed_url: string;
  artwork_url: string;
}

export interface VersionEntry {
  id: string;
  timestamp: string;
  type: "raw" | "validated";
  model: string | null;
  params: Record<string, unknown>;
  content_hash: string;
  segment_count: number;
  manual_edit: boolean;
}

// ── Pipeline config (from Python constants, not Pydantic) ─

export interface LLMProviderSpec {
  url: string;
  model: string;
  label: string;
  env_var?: string;
}

export interface PipelineConfig {
  whisper_models: Record<string, string>;
  default_whisper_model: string;
  tts_model_sizes: Record<string, string>;
  default_tts_model_size: string;
  assemble_strategies: Record<string, string>;
  llm_providers: Record<string, LLMProviderSpec>;
  default_ollama_model: string;
  default_source_lang: string;
  default_target_lang: string;
  detected_keys?: Record<string, string>;
}

// ── Synthesize (response shapes, not Pydantic) ───────────

export interface VoiceSample {
  file: string;
  duration: number;
  text: string;
}

export interface GeneratedSegment {
  speaker: string;
  text: string;
  start: number;
  end: number;
  audio_file: string;
  duration: number;
  voice_sample?: string;
  generated_at?: string;
}

export interface SynthesisStatus {
  voice_samples_extracted: boolean;
  tts_segments_generated: boolean;
  synthesized: boolean;
}

// ── Index (response shapes) ──────────────────────────────

export interface IndexStatus {
  model: string;
  chunking: string;
  indexed: boolean;
  chunk_count: number;
}

export interface CollectionInfo {
  name: string;
  model: string;
  chunker: string;
  episode_count: number;
}

// ── Search (response shape) ──────────────────────────────

export interface SearchResult {
  text: string;
  episode: string;
  speaker: string;
  start: number;
  end: number;
  score: number;
  source: string;
  speakers: { speaker: string; text: string; start: number; end: number }[] | null;
}

// ── Filesystem ───────────────────────────────────────────

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

// ── Models ───────────────────────────────────────────────

export interface CachedModel {
  id: string;
  name: string;
  size_bytes: number;
  size_mb: number;
  path: string;
}

export interface VRAMStatus {
  total_mb: number;
  used_mb: number;
  reserved_mb: number;
  free_mb: number;
  device: string;
}

export interface ModelsResponse {
  models: CachedModel[];
  cache_dir: string;
  vram: VRAMStatus | null;
}
