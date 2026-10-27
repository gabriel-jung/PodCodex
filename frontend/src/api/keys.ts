import { json, rawFetch } from "./client";
import type {
  APIKeyCreateRequest,
  APIKeyPublic,
  APIKeysListResponse,
  APIKeysScanResponse,
  APIKeyUpdateRequest,
} from "./generated-types";

export type { APIKeyPublic, APIKeysListResponse };
export type ScanResponse = APIKeysScanResponse;

export const listApiKeys = () => json<APIKeysListResponse>("/api/keys");

export const createApiKey = (req: APIKeyCreateRequest) =>
  json<APIKeyPublic>("/api/keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const updateApiKey = (
  name: string,
  patch: APIKeyUpdateRequest,
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
