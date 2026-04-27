import type { DirListing } from "./types";
import { BASE, json } from "./client";

export const listDirectory = (
  path: string,
  showFiles = false,
  extensions?: string[],
) => {
  const ext =
    extensions && extensions.length > 0
      ? `&extensions=${encodeURIComponent(extensions.join(","))}`
      : "";
  return json<DirListing>(
    `/api/fs/list?path=${encodeURIComponent(path)}&show_files=${showFiles}${ext}`,
  );
};

export const createDirectory = (path: string, name: string) =>
  json<{ path: string | null; error: string | null }>(
    `/api/fs/mkdir?path=${encodeURIComponent(path)}&name=${encodeURIComponent(name)}`,
    { method: "POST" },
  );

export const openFolder = (path: string) =>
  json<{ error: string | null }>(
    `/api/fs/open?path=${encodeURIComponent(path)}`,
    { method: "POST" },
  );

export interface DriveInfo {
  label: string;
  path: string;
}

export const listDrives = () =>
  json<{ drives: DriveInfo[] }>("/api/fs/drives");

/** Delete a non-audio auxiliary file (subtitles, JSON exports, etc). */
export const deleteFile = (path: string) =>
  json<{ status: string; path: string }>(
    `/api/fs/file?path=${encodeURIComponent(path)}`,
    { method: "DELETE" },
  );

// ── Artwork ────────────────────────────────

export const artworkUrl = (showFolder: string) =>
  `${BASE}/api/shows/artwork?show_folder=${encodeURIComponent(showFolder)}`;

// ── Audio ───────────────────────────────────

export const audioFileUrl = (path: string) =>
  `${BASE}/api/audio/file?path=${encodeURIComponent(path)}`;

export const deleteAudioFile = (path: string) =>
  json<{ status: string; path: string }>(`/api/audio/file?path=${encodeURIComponent(path)}`, {
    method: "DELETE",
  });

// ── Export ─────────────────────────────────

export const exportTextUrl = (audioPath: string, source = "transcript", outputDir?: string) => {
  const params = new URLSearchParams({ audio_path: audioPath, source });
  if (outputDir) params.set("output_dir", outputDir);
  return `${BASE}/api/export/text?${params}`;
};

export const exportSrtUrl = (audioPath: string, source = "transcript", outputDir?: string) => {
  const params = new URLSearchParams({ audio_path: audioPath, source });
  if (outputDir) params.set("output_dir", outputDir);
  return `${BASE}/api/export/srt?${params}`;
};

export const exportVttUrl = (audioPath: string, source = "transcript", outputDir?: string) => {
  const params = new URLSearchParams({ audio_path: audioPath, source });
  if (outputDir) params.set("output_dir", outputDir);
  return `${BASE}/api/export/vtt?${params}`;
};

export const exportZipUrl = (audioPath: string, outputDir?: string) => {
  const params = new URLSearchParams({ audio_path: audioPath });
  if (outputDir) params.set("output_dir", outputDir);
  return `${BASE}/api/export/zip?${params}`;
};
