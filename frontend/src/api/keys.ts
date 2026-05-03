import { json, rawFetch } from "./client";
import type { APIKeyPublic } from "./generated-types";

export type { APIKeyPublic };

export interface APIKeysListResponse {
  path: string;
  keys: APIKeyPublic[];
}

export interface ScanResponse {
  added: string[];
  keys: APIKeyPublic[];
}

export const listApiKeys = () => json<APIKeysListResponse>("/api/keys");

export const createApiKey = (req: {
  name: string;
  value: string;
  suggested_provider?: string | null;
}) =>
  json<APIKeyPublic>("/api/keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const updateApiKey = (
  name: string,
  patch: { value?: string; suggested_provider?: string | null },
) =>
  json<APIKeyPublic>(`/api/keys/${encodeURIComponent(name)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });

export const deleteApiKey = (name: string) =>
  rawFetch(`/api/keys/${encodeURIComponent(name)}`, { method: "DELETE" }).then(
    () => undefined,
  );

export const scanEnvForKeys = () =>
  json<ScanResponse>("/api/keys/scan-env", { method: "POST" });
