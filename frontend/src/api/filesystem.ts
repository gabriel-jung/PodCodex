import type { DirListing } from "./types";
import { BASE, json } from "./client";
import type { Platform } from "@/platform";

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

export type ExportFormat = "txt" | "srt" | "vtt" | "zip" | "audio";

export const saveExport = (req: {
  audio_path: string;
  output_dir?: string;
  source?: string;
  format: ExportFormat;
  dest: string;
}) =>
  json<{ status: string; path: string }>("/api/export/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

const _exportFallbackUrl = (
  format: ExportFormat,
  audioPath: string,
  source: string,
): string => {
  switch (format) {
    case "txt": return exportTextUrl(audioPath, source);
    case "srt": return exportSrtUrl(audioPath, source);
    case "vtt": return exportVttUrl(audioPath, source);
    case "zip": return exportZipUrl(audioPath);
    case "audio": return audioFileUrl(audioPath);
  }
};

/** Save an export to disk via native dialog (Tauri) or browser download (web). */
export async function saveExportFile(
  platform: Platform,
  args: {
    audioPath: string;
    format: ExportFormat;
    defaultName: string;
    source?: string;
  },
): Promise<void> {
  const ext = args.format === "audio"
    ? (args.audioPath.split(".").pop() || "mp3")
    : args.format;
  if (platform.isTauri) {
    const dest = await platform.fs.saveFileDialog({
      defaultPath: args.defaultName,
      extensions: [ext],
    });
    if (!dest) return;
    await saveExport({
      audio_path: args.audioPath,
      source: args.source,
      format: args.format,
      dest,
    });
    return;
  }
  const a = document.createElement("a");
  a.href = _exportFallbackUrl(args.format, args.audioPath, args.source ?? "transcript");
  a.download = args.defaultName;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
