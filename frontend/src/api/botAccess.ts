import { BASE, json } from "./client";

export interface ShowAccess {
  show: string;
  is_protected: boolean;
}

export interface ShowPasswordSet {
  show: string;
  password: string;
  generated: boolean;
}

const jsonHeaders = { "Content-Type": "application/json" };

export const getShowAccessList = () =>
  json<ShowAccess[]>("/api/bot-access/passwords");

export const getShowAccess = (show: string) =>
  json<ShowAccess>(`/api/bot-access/passwords/${encodeURIComponent(show)}`);

export const setShowPassword = (show: string, password?: string) =>
  json<ShowPasswordSet>(`/api/bot-access/passwords/${encodeURIComponent(show)}`, {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(password ? { password } : {}),
  });

export async function deleteShowPassword(show: string): Promise<void> {
  const res = await fetch(
    `${BASE}/api/bot-access/passwords/${encodeURIComponent(show)}`,
    { method: "DELETE" },
  );
  if (!res.ok) {
    throw new Error(`${res.status}: ${await res.text()}`);
  }
}
