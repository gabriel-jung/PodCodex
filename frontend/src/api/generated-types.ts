// AUTO-GENERATED — do not edit manually.
// Regenerate with: make types  (or: .venv/bin/python scripts/generate_types.py)
//
// Source: Pydantic models in src/podcodex/api/



export interface PipelineDefaultsSchema {
  model_size?: string;
  diarize?: boolean;
  llm_mode?: string;
  llm_provider?: string;
  llm_model?: string;
  target_lang?: string;
}

export interface ShowMeta {
  name: string;
  rss_url?: string;
  youtube_url?: string;
  language?: string;
  speakers?: string[];
  artwork_url?: string;
  pipeline?: PipelineDefaultsSchema;
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
  description?: string;
  audio_url?: string;
  duration?: number;
  episode_number?: number | null;
  season_number?: number | null;
  artwork_url?: string;
  local_stem?: string | null;
  downloaded?: boolean;
}

export interface Segment {
  speaker?: string;
  text?: string;
  start?: number;
  end?: number;
  flagged?: boolean;
}

export interface UnifiedEpisodeOut {
  id: string;
  title: string;
  stem?: string | null;
  pub_date?: string | null;
  description?: string;
  audio_url?: string | null;
  duration?: number;
  episode_number?: number | null;
  audio_path?: string | null;
  output_dir?: string | null;
  downloaded?: boolean;
  transcribed?: boolean;
  corrected?: boolean;
  indexed?: boolean;
  synthesized?: boolean;
  has_subtitles?: boolean;
  translations?: string[];
  artwork_url?: string;
  provenance?: Record<string, unknown>;
  files?: string[];
  transcribe_status?: string;
  correct_status?: string;
  translate_status?: string;
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
  show_folders?: string[];
  default_save_path?: string;
}

export interface ShowSummary {
  name: string;
  path: string;
  episode_count?: number;
  has_rss?: boolean;
  has_youtube?: boolean;
  artwork_url?: string;
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
  batch_size?: number;
  force?: boolean;
  diarize?: boolean;
  hf_token?: string | null;
  num_speakers?: number | null;
  show?: string;
  episode?: string;
}

export interface CorrectRequest {
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

export interface CorrectSkipRequest {
  audio_path: string;
  output_dir?: string | null;
}

export interface CorrectManualPromptsRequest {
  audio_path: string;
  output_dir?: string | null;
  context?: string;
  source_lang?: string;
  batch_minutes?: number;
  engine?: string;
}

export interface CorrectApplyManualRequest {
  audio_path: string;
  output_dir?: string | null;
  corrections: Record<string, unknown>[];
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

export interface TranslateManualPromptsRequest {
  audio_path: string;
  output_dir?: string | null;
  context?: string;
  source_lang?: string;
  target_lang?: string;
  batch_minutes?: number;
}

export interface TranslateApplyManualRequest {
  audio_path: string;
  output_dir?: string | null;
  lang?: string;
  corrections: Record<string, unknown>[];
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
  batch_size?: number;
  diarize?: boolean;
  clean?: boolean;
  transcribe_source?: string;
  sub_lang?: string;
  hf_token?: string | null;
  num_speakers?: number | null;
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
  show_name?: string;
  index_model_keys?: string[];
  index_chunkings?: string[];
  force?: boolean;
}

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

export interface SearchResultSchema {
  text: string;
  episode: string;
  speaker: string;
  start: number;
  end: number;
  score: number;
  source: string;
  speakers?: Record<string, unknown>[] | null;
}

export interface ExactRequest {
  query: string;
  folder?: string | null;
  audio_path?: string | null;
  show: string;
  model?: string;
  chunking?: string;
  top_k?: number;
  episode?: string | null;
  speaker?: string | null;
}

export interface RandomRequest {
  folder?: string | null;
  audio_path?: string | null;
  show: string;
  model?: string;
  chunking?: string;
  episode?: string | null;
  speaker?: string | null;
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

export interface YouTubeDownloadRequest {
  video_ids?: string[] | null;
  import_subs?: boolean;
  sub_lang?: string;
}

export interface YouTubeSubsRequest {
  video_ids: string[];
  lang?: string;
}
