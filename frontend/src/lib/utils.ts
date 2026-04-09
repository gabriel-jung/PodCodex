import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { Episode, ShowMeta, VersionEntry } from "@/api/types";

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

/** Strip HTML tags and decode common entities. */
export function stripHtml(html: string): string {
  const text = html.replace(/<[^>]*>/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, " ");
  return text.replace(/\s+/g, " ").trim();
}

/** Build a default LLM context string from show metadata and episode info. */
export function buildDefaultContext(episode: Episode, showMeta: ShowMeta | null | undefined): string {
  const parts: string[] = [];
  if (showMeta?.name) parts.push(showMeta.name);
  if (showMeta?.language) parts.push(`${showMeta.language} podcast`);
  if (showMeta?.speakers?.length) parts.push(`hosted by ${showMeta.speakers.join(" and ")}`);
  if (episode.title) parts.push(`episode: ${episode.title}`);
  let ctx = parts.join(", ");
  if (episode.description) ctx += `\nDescription: ${stripHtml(episode.description)}`;
  return ctx;
}

/** Derive a show name from metadata or audio path. */
export function getShowName(showMeta: ShowMeta | null | undefined, audioPath: string | null): string {
  if (showMeta?.name) return showMeta.name;
  if (!audioPath) return "";
  const parts = audioPath.replace(/\\/g, "/").split("/");
  return parts[parts.length - 2] || parts[parts.length - 1] || "";
}

/** True if any pipeline step is outdated (transcribe, correct, or translate). */
export function isOutdated(ep: { transcribe_status?: string; correct_status?: string; translate_status?: string }): boolean {
  return ep.transcribe_status === "outdated" || ep.correct_status === "outdated" || ep.translate_status === "outdated";
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
    hindi: "hi", turkish: "tr", correct: "co", swedish: "sv",
    danish: "da", norwegian: "no", finnish: "fi", greek: "el",
    czech: "cs", romanian: "ro", hungarian: "hu", ukrainian: "uk",
    catalan: "ca", hebrew: "he", thai: "th", vietnamese: "vi",
    indonesian: "id", malay: "ms",
  };
  const lower = lang.toLowerCase().trim();
  return map[lower] || lower;
}

// ── Subtitle languages (shared by download dropdowns) ────

export const SUB_LANGUAGES = [
  { code: "en", label: "English" },
  { code: "fr", label: "Français" },
  { code: "de", label: "Deutsch" },
  { code: "es", label: "Español" },
  { code: "it", label: "Italiano" },
  { code: "pt", label: "Português" },
  { code: "ja", label: "日本語" },
  { code: "ko", label: "한국어" },
  { code: "zh", label: "中文" },
  { code: "ar", label: "العربية" },
  { code: "ru", label: "Русский" },
] as const;

// ── Version formatting ────────────────────────────────────

/** Format a version's timestamp as a short date string. */
export function versionDate(v: VersionEntry): string {
  const d = new Date(v.timestamp);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export const SOURCE_LABELS: Record<string, string> = {
  whisper: "Whisper",
  "youtube-subtitles": "YouTube subtitles",
  upload: "Upload",
  import: "Import",
};

const STEP_LABELS: Record<string, string> = {
  transcript: "Transcript",
  corrected: "Corrected",
  segments: "Segments",
  diarization: "Diarization",
  diarized_segments: "Diarized segments",
};

/** Format a version's step as a display tag, e.g. "transcript", "translated · fr". */
export function stepTag(step: string, type?: string): string {
  const edited = type === "validated" ? " · edited" : "";
  if (step in STEP_LABELS) return `${STEP_LABELS[step]}${edited}`;
  return `translated · ${step}${edited}`;
};

/** Build a compact label for a version (model, provider, language info). */
export function versionLabel(v: VersionEntry): string {
  const p = v.params as Record<string, unknown>;
  if (p.skipped) return "Skipped (copy)";

  // Source chain: "Whisper/base, diarized → ollama → openai"
  const chain = p.source_chain as string[] | undefined;
  if (chain && chain.length > 0) {
    const formatted = chain.map((s) => {
      // Split "whisper/base, diarized" → map source part, keep rest
      const [main, ...rest] = s.split(", ");
      const [source, ...model] = main.split("/");
      const label = [SOURCE_LABELS[source] || source, ...model].join(" ");
      return rest.length > 0 ? `${label}, ${rest.join(", ")}` : label;
    });
    return formatted.join(" → ");
  }

  // Legacy / transcript: flat label
  const parts: string[] = [];
  if (p.source) parts.push(SOURCE_LABELS[String(p.source)] || String(p.source));
  if (v.model) parts.push(v.model);
  if (p.llm_provider) parts.push(String(p.llm_provider));
  else if (p.llm_mode === "manual") parts.push("Manual");
  else if (p.llm_mode) parts.push(String(p.llm_mode));
  if (p.language) parts.push(String(p.language));
  else if (p.source_lang && p.target_lang) parts.push(`${p.source_lang} → ${p.target_lang}`);
  else if (p.source_lang) parts.push(String(p.source_lang));
  if (p.diarize === true) parts.push("diarized");
  return parts.join(", ") || "Generated";
}

/** Params to hide from the version info box (internal / not user-relevant). */
const HIDDEN_VERSION_PARAMS = new Set(["meta", "batch_size", "batch_minutes", "engine", "input_source", "skipped", "source", "source_chain"]);

/** Format all params as key: value rows for a version info box. */
export function versionInfo(v: VersionEntry): { key: string; value: string }[] {
  const rows: { key: string; value: string }[] = [];
  if (v.model) rows.push({ key: "Model", value: v.model });
  rows.push({ key: "Type", value: v.type === "validated" ? "Edited" : "Generated" });
  rows.push({ key: "Segments", value: String(v.segment_count) });
  rows.push({ key: "Hash", value: v.content_hash.replace("sha256:", "").slice(0, 8) });
  const p = v.params as Record<string, unknown>;
  for (const [k, val] of Object.entries(p)) {
    if (HIDDEN_VERSION_PARAMS.has(k) || val === null || val === undefined) continue;
    const label = k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    rows.push({ key: label, value: typeof val === "boolean" ? (val ? "yes" : "no") : String(val) });
  }
  return rows;
}
