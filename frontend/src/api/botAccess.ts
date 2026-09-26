import { json, rawFetch } from "./client";
import type { ShowAccess, ShowPasswordSet } from "./generated-types";

export type { ShowAccess, ShowPasswordSet };

const jsonHeaders = { "Content-Type": "application/json" };

export const getShowAccessList = () =>
  json<ShowAccess[]>("/api/bot-access/passwords");

export const getShowAccess = (show: string) =>
  json<ShowAccess>(`/api/bot-access/password?${new URLSearchParams({ show })}`);

export const setShowPassword = (show: string, password?: string) =>
  json<ShowPasswordSet>(`/api/bot-access/password?${new URLSearchParams({ show })}`, {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(password ? { password } : {}),
  });

export async function deleteShowPassword(show: string): Promise<void> {
  await rawFetch(`/api/bot-access/password?${new URLSearchParams({ show })}`, {
    method: "DELETE",
  });
}
