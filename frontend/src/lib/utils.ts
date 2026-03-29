import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { Episode, ShowMeta } from "@/api/types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(...inputs));
}

/** Format seconds as "HH:MM:SS" or "MM:SS". */
export function formatTime(seconds: number | undefined | null, showMs = true): string {
  if (seconds == null || isNaN(seconds)) return "00:00";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 100);
  const base = h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return showMs ? `${base}.${String(ms).padStart(2, "0")}` : base;
}

/** Format seconds as a human-friendly duration like "1h 23m" or "45m". */
export function formatDuration(seconds: number | undefined | null): string {
  if (!seconds || seconds <= 0) return "";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m > 0 ? `${m}m` : ""}`.trim();
  return `${m}m`;
}

/** Format an ISO date string as a short locale date. */
export function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

/** Format an ISO date string as a relative time ago string. */
export function timeAgo(dateStr: string | null | undefined): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return "";
  const diff = Date.now() - d.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return formatDate(dateStr);
}

/** Build a default LLM context string from show metadata and episode info. */
export function buildDefaultContext(episode: Episode, showMeta: ShowMeta | null | undefined): string {
  const parts: string[] = [];
  if (showMeta?.name) parts.push(showMeta.name);
  if (showMeta?.language) parts.push(`${showMeta.language} podcast`);
  if (showMeta?.speakers?.length) parts.push(`hosted by ${showMeta.speakers.join(" and ")}`);
  if (episode.title) parts.push(`episode: ${episode.title}`);
  return parts.join(", ");
}

/** Derive a show name from metadata or audio path. */
export function getShowName(showMeta: ShowMeta | null | undefined, audioPath: string | null): string {
  if (showMeta?.name) return showMeta.name;
  if (!audioPath) return "";
  const parts = audioPath.replace(/\\/g, "/").split("/");
  return parts[parts.length - 2] || parts[parts.length - 1] || "";
}

/** Shared CSS class for <select> elements across forms. */
export const selectClass = "bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm";

/** Extract error message from a mutation error, with safe casting. */
export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  return "An unknown error occurred";
}

/** Map a language name to its ISO 639-1 code for WhisperX. */
export function languageToISO(lang: string): string {
  const map: Record<string, string> = {
    english: "en", french: "fr", german: "de", spanish: "es",
    italian: "it", portuguese: "pt", dutch: "nl", russian: "ru",
    japanese: "ja", chinese: "zh", korean: "ko", arabic: "ar",
    hindi: "hi", turkish: "tr", polish: "pl", swedish: "sv",
    danish: "da", norwegian: "no", finnish: "fi", greek: "el",
    czech: "cs", romanian: "ro", hungarian: "hu", ukrainian: "uk",
    catalan: "ca", hebrew: "he", thai: "th", vietnamese: "vi",
    indonesian: "id", malay: "ms",
  };
  const lower = lang.toLowerCase().trim();
  return map[lower] || lower;
}
