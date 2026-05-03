// AUTO-GENERATED — do not edit manually.
// Regenerate with: make types  (or: .venv/bin/python scripts/generate_types.py)
//
// Source: Pydantic models in src/podcodex/api/



export type Mode = "full" | "index-only";

export interface PipelineDefaultsSchema {
  model_size: string;
  diarize?: boolean | null;
  llm_mode: string;
  llm_provider_profile: string;
  llm_key_name: string;
  llm_model: string;
  target_lang: string;
}

export interface ShowMeta {
  name: string;
  rss_url: string;
  youtube_url: string;
  language: string;
  speakers: string[];
  artwork_url: string;
  pipeline: PipelineDefaultsSchema;
  last_feed_update?: string | null;
}

export interface EpisodeOut {
  stem: string;
  title: string;
  audio_path: string | null;
  output_dir: string;
  segments_ready: boolean;
  diarized: boolean;
  assigned: boolean;
  transcribed: boolean;
  corrected: boolean;
  indexed: boolean;
  synthesized: boolean;
  translations: string[];
}

export interface RSSEpisodeOut {
  guid: string;
  title: string;
  pub_date: string;
  description: string;
  audio_url: string;
  duration: number;
  episode_number?: number | null;
  season_number?: number | null;
  artwork_url: string;
  removed: boolean;
  feed_order?: number | null;
  local_stem?: string | null;
  downloaded: boolean;
}

export interface Segment {
  speaker: string;
  text: string;
  start: number;
  end: number;
  flagged: boolean;
}

export interface UnifiedEpisodeOut {
  id: string;
  title: string;
  stem?: string | null;
  pub_date?: string | null;
  description: string;
  audio_url?: string | null;
  duration: number;
  episode_number?: number | null;
  audio_path?: string | null;
  output_dir?: string | null;
  downloaded: boolean;
  removed: boolean;
  feed_order?: number | null;
  transcribed: boolean;
  corrected: boolean;
  indexed: boolean;
  synthesized: boolean;
  has_subtitles: boolean;
  translations: string[];
  artwork_url: string;
  provenance: Record<string, unknown>;
  segment_count?: number | null;
  files: string[];
  transcribe_status: string;
  correct_status: string;
  translate_status: string;
}

export interface CreateFromRSSRequest {
  rss_url: string;
  save_path: string;
  folder_name?: string;
  name?: string;
  artwork_url?: string;
  language?: string;
}

export interface RegisterShowRequest {
  path: string;
}

export interface CreateFromRSSResponse {
  folder: string;
  name: string;
  episode_count: number;
}

export interface CreateFromYouTubeRequest {
  youtube_url: string;
  save_path: string;
  folder_name?: string;
  name?: string;
  artwork_url?: string;
  language?: string;
}

export interface CreateFromYouTubeResponse {
  folder: string;
  name: string;
  episode_count: number;
}

export interface TaskResponse {
  task_id: string;
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
  has_youtube: boolean;
  artwork_url: string;
  last_rss_update?: string | null;
}

export interface MoveShowRequest {
  new_path: string;
  move_files?: boolean;
}

export interface TranscribeRequest {
  audio_path: string;
  output_dir?: string | null;
  model_size?: string;
  language?: string;
  batch_size?: number | null;
  force?: boolean;
  diarize?: boolean;
  hf_token?: string | null;
  num_speakers?: number | null;
  show?: string;
  episode?: string;
  clean?: boolean;
}

export interface CorrectRequest {
  audio_path: string;
  output_dir?: string | null;
  mode?: string;
  provider_profile?: string | null;
  key_name?: string | null;
  model?: string;
  context?: string;
  source_lang?: string;
  batch_minutes?: number;
  source_version_id?: string | null;
}

export interface CorrectManualPromptsRequest {
  audio_path?: string | null;
  output_dir?: string | null;
  context?: string;
  source_lang?: string;
  target_lang?: string;
  batch_minutes?: number;
  source_version_id?: string | null;
}

export interface CorrectApplyManualRequest {
  audio_path?: string | null;
  output_dir?: string | null;
  corrections: Record<string, unknown>[];
  lang?: string;
}

export interface EpisodeListItem {
  episode: string;
  episode_title: string;
  pub_date: string;
  episode_number?: number | null;
  chunk_count: number;
  duration: number;
}

export interface EpisodeMeta {
  episode: string;
  episode_title: string;
  pub_date: string;
  episode_number?: number | null;
  description: string;
  source: string;
  chunk_count: number;
  duration: number;
  speakers: string[];
}

export interface TranslateRequest {
  audio_path: string;
  output_dir?: string | null;
  mode?: string;
  provider_profile?: string | null;
  key_name?: string | null;
  model?: string;
  context?: string;
  source_lang?: string;
  batch_minutes?: number;
  source_version_id?: string | null;
  target_lang?: string;
}

export interface TranslateManualPromptsRequest {
  audio_path?: string | null;
  output_dir?: string | null;
  context?: string;
  source_lang?: string;
  target_lang?: string;
  batch_minutes?: number;
  source_version_id?: string | null;
}

export interface TranslateApplyManualRequest {
  audio_path?: string | null;
  output_dir?: string | null;
  corrections: Record<string, unknown>[];
  lang?: string;
}

export interface BatchRequest {
  show_folder: string;
  audio_paths: string[];
  transcribe?: boolean;
  correct?: boolean;
  translate?: boolean;
  index?: boolean;
  model_size?: string;
  language?: string;
  batch_size?: number | null;
  diarize?: boolean;
  clean?: boolean;
  hf_token?: string | null;
  num_speakers?: number | null;
  transcribe_source?: string;
  sub_lang?: string;
  llm_mode?: string;
  llm_provider_profile?: string | null;
  llm_key_name?: string | null;
  llm_model?: string;
  context?: string;
  source_lang?: string;
  target_lang?: string;
  llm_batch_minutes?: number;
  engine?: string;
  force?: boolean;
  show_name?: string;
  index_model_keys?: string[];
  index_chunkings?: string[];
}

export interface SearchRequest {
  query: string;
  show: string;
  model?: string;
  chunking?: string;
  top_k?: number;
  alpha?: number;
  episode?: string | null;
  episodes?: string[] | null;
  speaker?: string | null;
  source?: string | null;
  pub_date_min?: string | null;
  pub_date_max?: string | null;
}

export interface SearchResultSchema {
  text: string;
  episode: string;
  episode_stem: string;
  episode_number?: number | null;
  audio_path: string;
  speaker: string;
  start: number;
  end: number;
  score: number;
  source: string;
  pub_date: string;
  speakers?: Record<string, unknown>[] | null;
  accent_match: boolean;
  fuzzy_match: boolean;
  match_text?: string | null;
}

export interface ExactRequest {
  query: string;
  show: string;
  model?: string;
  chunking?: string;
  episode?: string | null;
  episodes?: string[] | null;
  speaker?: string | null;
  source?: string | null;
  pub_date_min?: string | null;
  pub_date_max?: string | null;
}

export interface RandomRequest {
  show: string;
  model?: string;
  chunking?: string;
  episode?: string | null;
  episodes?: string[] | null;
  speaker?: string | null;
  source?: string | null;
  pub_date_min?: string | null;
  pub_date_max?: string | null;
}

export interface IndexRequest {
  audio_path: string;
  output_dir?: string | null;
  show: string;
  source?: string;
  version_id?: string | null;
  model_keys?: string[];
  chunkings?: string[];
  chunk_size?: number;
  threshold?: number;
  overwrite?: boolean;
}

export interface ExtractVoicesRequest {
  audio_path: string;
  output_dir?: string | null;
  min_duration?: number | null;
  max_duration?: number | null;
  top_k?: number;
}

export interface ExtractSelectedRequest {
  audio_path: string;
  output_dir?: string | null;
  selections: Record<string, unknown>[];
}

export interface GenerateRequest {
  audio_path: string;
  output_dir?: string | null;
  model_size?: string;
  language?: string;
  source_lang?: string | null;
  source_version_id?: string | null;
  max_chunk_duration?: number;
  force?: boolean;
  only_speakers?: string[] | null;
  keep_segment_keys?: string[] | null;
}

export interface AssembleRequest {
  audio_path: string;
  output_dir?: string | null;
  strategy?: string;
  silence_duration?: number;
}

export interface YouTubeDownloadRequest {
  video_ids?: string[] | null;
  import_subs?: boolean;
  sub_lang?: string;
}

export interface YouTubeSubsRequest {
  video_ids: string[];
  lang?: string;
}

export interface CollectionEntry {
  name: string;
  model: string;
  chunker: string;
  dim: number;
  rows: number;
}

export interface ShowEntry {
  name: string;
  folder: string;
  audio_included: boolean;
  collections: CollectionEntry[];
}

export interface Manifest {
  schema_version: number;
  mode: Mode;
  podcodex_version: string;
  exported_at: string;
  shows: ShowEntry[];
}

export interface ArchivePreview {
  archive_path: string;
  manifest: Manifest;
  size_bytes: number;
  embedder_warnings: string[];
}

export interface ExportResult {
  output_path: string;
  size_bytes: number;
  mode: Mode;
  shows_exported: number;
  collections_exported: number;
  audio_included: boolean;
}

export interface ImportResult {
  shows_dir: string;
  mode: Mode;
  shows_imported: string[];
  collections_imported: string[];
  conflicts_resolved: Record<string, string>;
}

export interface ExportShowRequest {
  show_folder: string;
  output_path: string;
  with_audio?: boolean;
  index_only?: boolean;
}

export interface ExportIndexRequest {
  show_folders: string[];
  output_path: string;
}

export interface PreviewRequest {
  archive_path: string;
}

export interface ImportRequest {
  archive_path: string;
  shows_dir?: string | null;
  name?: string | null;
  on_conflict?: "auto" | "rename" | "replace" | "abort";
}

export interface APIKeyPublic {
  name: string;
  masked: string;
  suggested_provider?: string | null;
  source: "ui" | "env";
}

export interface ProviderProfile {
  name: string;
  type: "openai" | "anthropic" | "mistral" | "ollama" | "openai-compatible";
  base_url?: string | null;
  builtin: boolean;
}
