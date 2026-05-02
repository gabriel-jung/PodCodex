import { json, rawFetch } from "./client";
import type { ProviderProfile } from "./generated-types";

export type { ProviderProfile };
export type ProviderType = ProviderProfile["type"];

export interface ProviderProfilesListResponse {
  profiles: ProviderProfile[];
}

export const listProviderProfiles = () =>
  json<ProviderProfilesListResponse>("/api/provider-profiles");

export const createProviderProfile = (req: { name: string; base_url: string }) =>
  json<ProviderProfile>("/api/provider-profiles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const updateProviderProfile = (
  name: string,
  patch: { base_url?: string },
) =>
  json<ProviderProfile>(`/api/provider-profiles/${encodeURIComponent(name)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });

export const deleteProviderProfile = (name: string) =>
  rawFetch(`/api/provider-profiles/${encodeURIComponent(name)}`, {
    method: "DELETE",
  }).then(() => undefined);
