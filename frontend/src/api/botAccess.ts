import { json, rawFetch } from "./client";
import type { ShowAccess, ShowPasswordSet } from "./generated-types";

export type { ShowAccess, ShowPasswordSet };

const jsonHeaders = { "Content-Type": "application/json" };

export const getShowAccessList = () =>
  json<ShowAccess[]>("/api/bot-access/passwords");

// `showId` is the password-table key: the show's id, or its display name for
// a show that has none yet (see ShowAccess.show_id).
export const getShowAccess = (showId: string) =>
  json<ShowAccess>(`/api/bot-access/password?${new URLSearchParams({ show_id: showId })}`);

export const setShowPassword = (showId: string, password?: string) =>
  json<ShowPasswordSet>(`/api/bot-access/password?${new URLSearchParams({ show_id: showId })}`, {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(password ? { password } : {}),
  });

export async function deleteShowPassword(showId: string): Promise<void> {
  await rawFetch(`/api/bot-access/password?${new URLSearchParams({ show_id: showId })}`, {
    method: "DELETE",
  });
}
