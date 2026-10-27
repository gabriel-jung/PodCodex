import { json, rawFetch } from "./client";
import type {
  ProviderProfile,
  ProviderProfileCreateRequest,
  ProviderProfilesListResponse,
  ProviderProfileUpdateRequest,
} from "./generated-types";

export type { ProviderProfile, ProviderProfilesListResponse };

export const listProviderProfiles = () =>
  json<ProviderProfilesListResponse>("/api/provider-profiles");

export const createProviderProfile = (req: ProviderProfileCreateRequest) =>
  json<ProviderProfile>("/api/provider-profiles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const updateProviderProfile = (
  name: string,
  patch: ProviderProfileUpdateRequest,
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
