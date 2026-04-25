/**
 * Bundle (.podcodex archive) export/import endpoints.
 *
 * The desktop frontend opens save/open dialogs via Tauri and passes the
 * chosen filesystem paths to the backend — no multipart upload.
 */

import type {
  ArchivePreview,
  ExportIndexRequest,
  ExportResult,
  ExportShowRequest,
  ImportRequest,
  ImportResult,
} from "./generated-types";
import { json } from "./client";

const _post = <T>(url: string, body: unknown): Promise<T> =>
  json<T>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export const exportShowBundle = (req: ExportShowRequest) =>
  _post<ExportResult>("/api/bundle/export-show", req);

export const exportIndexBundle = (req: ExportIndexRequest) =>
  _post<ExportResult>("/api/bundle/export-index", req);

export const previewBundle = (archive_path: string) =>
  _post<ArchivePreview>("/api/bundle/preview", { archive_path });

export const importBundle = (req: ImportRequest) =>
  _post<ImportResult>("/api/bundle/import", req);
