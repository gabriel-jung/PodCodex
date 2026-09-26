import type { DirListing } from "./types";
import { BASE, json, withToken } from "./client";
import { episodeParams } from "./versions";
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
  json<{ drives: DriveInfo[]; home: string }>("/api/fs/drives");

/** Delete a non-audio auxiliary file (subtitles, JSON exports, etc). */
export const deleteFile = (path: string) =>
  json<{ status: string; path: string }>(
    `/api/fs/file?path=${encodeURIComponent(path)}`,
    { method: "DELETE" },
  );

// ── Artwork ────────────────────────────────

export const artworkUrl = (showFolder: string) =>
  withToken(`${BASE}/api/shows/artwork?show_folder=${encodeURIComponent(showFolder)}`);

// ── Audio ───────────────────────────────────

export const audioFileUrl = (path: string) =>
  withToken(`${BASE}/api/audio/file?path=${encodeURIComponent(path)}`);

export const deleteAudioFile = (path: string) =>
  json<{ status: string; path: string }>(`/api/audio/file?path=${encodeURIComponent(path)}`, {
    method: "DELETE",
  });

// ── Export ─────────────────────────────────

// One builder for every GET export URL. episodeParams leaves out an empty
// audio path (subtitle-only episodes export by their folder).
const exportUrl = (
  kind: "text" | "srt" | "vtt" | "zip",
  audioPath: string,
  outputDir?: string,
  extra?: { source?: string; version_id?: string },
) => withToken(`${BASE}/api/export/${kind}?${episodeParams(audioPath, outputDir, extra)}`);

export type ExportFormat = "txt" | "srt" | "vtt" | "zip" | "audio";

export const saveExport = (req: {
  audio_path: string;
  output_dir?: string;
  source?: string;
  /** Export this version instead of the step's default (the one on screen). */
  version_id?: string;
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
  outputDir?: string,
  versionId?: string,
): string => {
  const extra = { source, version_id: versionId };
  switch (format) {
    case "txt": return exportUrl("text", audioPath, outputDir, extra);
    case "srt": return exportUrl("srt", audioPath, outputDir, extra);
    case "vtt": return exportUrl("vtt", audioPath, outputDir, extra);
    case "zip": return exportUrl("zip", audioPath, outputDir);
    case "audio": return audioFileUrl(audioPath);
  }
};

/** Save an export to disk via native dialog (Tauri) or browser download (web).
 *  Either `audioPath` or `outputDir` must be provided — `outputDir` covers
 *  YouTube episodes whose audio hasn't been downloaded yet but whose
 *  per-episode folder still holds transcripts/subs to export. */
export async function saveExportFile(
  platform: Platform,
  args: {
    audioPath?: string;
    outputDir?: string;
    format: ExportFormat;
    defaultName: string;
    source?: string;
    versionId?: string;
  },
): Promise<void> {
  if (!args.audioPath && !args.outputDir) {
    throw new Error("saveExportFile requires audioPath or outputDir");
  }
  const ext = args.format === "audio"
    ? (args.audioPath?.split(".").pop() || "mp3")
    : args.format;
  if (platform.isTauri) {
    const dest = await platform.fs.saveFileDialog({
      defaultPath: args.defaultName,
      extensions: [ext],
    });
    if (!dest) return;
    await saveExport({
      audio_path: args.audioPath ?? "",
      output_dir: args.outputDir,
      source: args.source,
      version_id: args.versionId,
      format: args.format,
      dest,
    });
    return;
  }
  const a = document.createElement("a");
  a.href = _exportFallbackUrl(args.format, args.audioPath ?? "", args.source ?? "transcript", args.outputDir, args.versionId);
  a.download = args.defaultName;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
