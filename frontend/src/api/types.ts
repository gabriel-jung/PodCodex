/** TypeScript types matching the FastAPI schemas. */

export interface HealthResponse {
  status: string;
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
}

export interface ShowMeta {
  name: string;
  rss_url: string;
  language: string;
  speakers: string[];
  artwork_url: string;
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
  translations: string[];
  artwork_url: string;
  raw_transcript: boolean;
  validated_transcript: boolean;
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

export interface VersionInfo {
  has_raw: boolean;
  has_validated: boolean;
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
  batch_size?: number;
  engine?: string;
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
  batch_size?: number;
}
