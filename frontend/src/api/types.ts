/**
 * TypeScript types for the PodCodex API.
 *
 * Generated types (from Pydantic models) are re-exported from generated-types.ts.
 * Frontend-only types (not backed by Pydantic) are defined here.
 */

// ── Re-export all generated types ────────────────────────
// AssembleStrategy is a bare ``Literal`` alias on the Python side, not a
// Pydantic model. The schema generator only emits BaseModel-derived
// interfaces, so we mirror the alias here. Keep in sync with
// ``src/podcodex/core/constants.py:AssembleStrategy``.
export type AssembleStrategy = "silence" | "original_timing";

export type {
  AppConfig,
  AssembleRequest,
  BatchFix,
  BatchRequest,
  BroadcastPreviewOut,
  CreateFromRSSResponse,
  CreateFromYouTubeResponse,
  FilesImportResponse,
  GenerateRequest,
  IndexRequest,
  PipelineDefaultsSchema as PipelineDefaults,
  CorrectRequest,
  RSSEpisodeOut,
  SearchRequest,
  Segment,
  ShowMeta,
  ShowSummary,
  TaskResponse,
  TranscribeRequest,
  TranslateRequest,
  UnifiedEpisodeOut,
} from "./generated-types";

export { AUDIO_EXTENSIONS } from "./generated-types";

// ── Frontend-only types (not backed by Pydantic models) ──

export interface HealthResponse {
  status: string;
  capabilities: Record<string, boolean>;
  /** "bundle" when running as the frozen PyInstaller sidecar; "dev" when
   *  uvicorn is running from a venv. Frontend uses this to hide tabs
   *  (e.g. Plugins) whose actions only make sense with a venv. */
  mode: "bundle" | "dev";
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

export interface OllamaCheckResponse {
  /** Python `ollama` package importable in the backend process. */
  installed: boolean;
  /** Daemon responded to `list()` at `host`. */
  reachable: boolean;
  /** Resolved host (honors `OLLAMA_HOST`, default `http://localhost:11434`). */
  host: string;
  /** Pulled model names, sorted. Empty when unreachable or none pulled. */
  models: string[];
  /** Reason the daemon was unreachable. `null` when reachable. */
  error: string | null;
}

/** Unified episode used throughout the frontend (aliased from generated). */
export type Episode = import("./generated-types").UnifiedEpisodeOut;

export interface PodcastSearchResult {
  name: string;
  artist: string;
  feed_url: string;
  artwork_url: string;
}

export interface VersionEntry {
  id: string;
  step?: string;
  timestamp: string;
  type: "raw" | "validated";
  model: string | null;
  params: Record<string, unknown>;
  content_hash: string;
  segment_count: number;
  manual_edit: boolean;
}

// ── Pipeline config (from Python constants, not Pydantic) ─

export interface PipelineConfig {
  whisper_models: Record<string, string>;
  default_whisper_model: string;
  tts_model_sizes: Record<string, string>;
  default_tts_model_size: string;
  assemble_strategies: Record<string, string>;
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

// ── Search (response shape) ──────────────────────────────

export interface SearchResult {
  text: string;
  episode: string;
  episode_stem: string;
  episode_number: number | null;
  audio_path: string;
  output_dir: string;
  speaker: string;
  start: number;
  end: number;
  score: number;
  source: string;
  pub_date: string;
  speakers: { speaker: string; text: string; start: number; end: number }[] | null;
  accent_match: boolean;
  fuzzy_match: boolean;
  match_text: string | null;
}

// ── Speaker roster ───────────────────────────────────────

export interface SpeakerEpisodeEntry {
  stem: string;
  title: string;
  segment_count: number;
  total_seconds: number;
}

export interface SpeakerRosterEntry {
  name: string;
  is_known: boolean;
  episode_count: number;
  segment_count: number;
  total_seconds: number;
  episodes: SpeakerEpisodeEntry[];
}

export interface SpeakerRosterResponse {
  speakers: SpeakerRosterEntry[];
  episodes_scanned: number;
  episodes_with_transcripts: number;
}

export interface EpisodeSpeakerEntry {
  name: string;
  total_seconds: number;
  pct: number; // share of episode duration (0-100); may total < 100 (music/gaps)
}

export interface EpisodeSpeakersResponse {
  speakers: EpisodeSpeakerEntry[]; // sorted by total_seconds desc
  episode_seconds: number;
  has_transcript: boolean;
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
