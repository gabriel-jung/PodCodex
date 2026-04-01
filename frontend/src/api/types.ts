/** TypeScript types matching the FastAPI schemas. */

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

export interface AppConfig {
  show_folders: string[];
  default_save_path: string;
}

export interface ShowSummary {
  name: string;
  path: string;
  episode_count: number;
  has_rss: boolean;
  artwork_url: string;
  last_rss_update: string | null;
}

export interface PipelineDefaults {
  model_size: string;
  diarize: boolean;
  llm_mode: string;
  llm_provider: string;
  llm_model: string;
  target_lang: string;
}

export interface ShowMeta {
  name: string;
  rss_url: string;
  language: string;
  speakers: string[];
  artwork_url: string;
  pipeline?: PipelineDefaults;
}

export interface Episode {
  id: string;
  title: string;
  stem: string | null;
  pub_date: string | null;
  description: string;
  audio_url: string | null;
  duration: number;
  episode_number: number | null;
  audio_path: string | null;
  downloaded: boolean;
  transcribed: boolean;
  polished: boolean;
  indexed: boolean;
  synthesized: boolean;
  translations: string[];
  artwork_url: string;
  provenance: Record<string, unknown>;
  transcribe_status: "none" | "outdated" | "done";
  polish_status: "none" | "outdated" | "done";
  translate_status: "none" | "outdated" | "done";
}

export interface DownloadResult {
  stem: string;
  audio_path: string | null;
  status: "downloaded" | "exists" | "failed" | "no_audio";
}

export interface Segment {
  speaker: string;
  text: string;
  start: number;
  end: number;
  flagged?: boolean;
}

export interface CreateFromRSSResponse {
  folder: string;
  name: string;
  episode_count: number;
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

export interface TaskResponse {
  task_id: string;
}

export interface TranscribeRequest {
  audio_path: string;
  output_dir?: string | null;
  model_size?: string;
  language?: string;
  batch_size?: number;
  force?: boolean;
  diarize?: boolean;
  hf_token?: string | null;
  num_speakers?: number | null;
  show?: string;
  episode?: string;
}

export interface PolishRequest {
  audio_path: string;
  output_dir?: string | null;
  mode?: string;
  provider?: string | null;
  model?: string;
  context?: string;
  source_lang?: string;
  batch_minutes?: number;
  engine?: string;
  api_base_url?: string;
  api_key?: string | null;
}

export interface TranslateRequest {
  audio_path: string;
  output_dir?: string | null;
  mode?: string;
  provider?: string | null;
  model?: string;
  context?: string;
  source_lang?: string;
  target_lang?: string;
  batch_minutes?: number;
  api_base_url?: string;
  api_key?: string | null;
}

// ── Batch ─────────────────────────────────

export interface BatchRequest {
  show_folder: string;
  audio_paths: string[];
  // Step toggles
  transcribe?: boolean;
  polish?: boolean;
  translate?: boolean;
  index?: boolean;
  // Transcribe config
  model_size?: string;
  language?: string;
  batch_size?: number;
  diarize?: boolean;
  hf_token?: string | null;
  num_speakers?: number | null;
  // LLM config (polish/translate)
  llm_mode?: string;
  llm_provider?: string | null;
  llm_model?: string;
  llm_api_base_url?: string;
  llm_api_key?: string | null;
  context?: string;
  source_lang?: string;
  target_lang?: string;
  llm_batch_minutes?: number;
  engine?: string;
  // Index config
  show_name?: string;
  index_model_keys?: string[];
  index_chunkings?: string[];
}

// ── Pipeline config (from Python constants) ─

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
  /** Masked env keys detected on the backend (e.g. { hf_token: "hf_k****", mistral: "sk-4****" }) */
  detected_keys?: Record<string, string>;
}

// ── Synthesize ─────────────────────────────

export interface ExtractVoicesRequest {
  audio_path: string;
  output_dir?: string | null;
  min_duration?: number | null;
  max_duration?: number | null;
  top_k?: number;
}

export interface GenerateRequest {
  audio_path: string;
  output_dir?: string | null;
  model_size?: string;
  language?: string;
  source_lang?: string | null;
  max_chunk_duration?: number;
  force?: boolean;
  only_speakers?: string[] | null;
}

export interface AssembleRequest {
  audio_path: string;
  output_dir?: string | null;
  strategy?: string;
  silence_duration?: number;
}

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

// ── Index ──────────────────────────────────

export interface IndexRequest {
  audio_path: string;
  output_dir?: string | null;
  show: string;
  source?: string;
  model_keys?: string[];
  chunkings?: string[];
  chunk_size?: number;
  threshold?: number;
  overwrite?: boolean;
}

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

// ── Search ─────────────────────────────────

export interface SearchRequest {
  query: string;
  audio_path?: string | null;
  folder?: string | null;
  output_dir?: string | null;
  show: string;
  model?: string;
  chunking?: string;
  top_k?: number;
  alpha?: number;
  episode?: string | null;
  speaker?: string | null;
}

export interface SyncRequest {
  folder: string;
  show: string;
  overwrite?: boolean;
  qdrant_url?: string | null;
}

// ── Filesystem ────────────────────────────

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

// ── Models ────────────────────────────────

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

// ── Search ─────────────────────────────────

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
