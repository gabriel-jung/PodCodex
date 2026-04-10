/**
 * Pipeline step registry and small helper components used by EpisodePage.
 *
 * Extracted to keep EpisodePage focused on layout and data fetching.
 */

import { useState } from "react";
import type { Episode } from "@/api/types";
import { formatDuration, formatDate } from "@/lib/utils";
import {
  Mic, Sparkles, Languages, AudioLines, Database, ExternalLink,
} from "lucide-react";

import TranscribePanel from "@/components/transcribe/TranscribePanel";
import CorrectPanel from "@/components/correct/CorrectPanel";
import TranslatePanel from "@/components/translate/TranslatePanel";
import SynthesizePanel from "@/components/synthesize/SynthesizePanel";
import IndexPanel from "@/components/index/IndexPanel";

// ── Types ────────────────────────────────────────────────────────────────

export type StepStatus = "done" | "partial" | false;

export interface PipelineStepDef {
  key: "transcribe" | "correct" | "translate" | "synthesize" | "index";
  label: string;
  rowLabel: string;
  icon: typeof Mic;
  section: "core" | "bonus";
  headerBadge: boolean;
  component: () => JSX.Element;
  status: (e: Episode) => StepStatus;
  matchFiles?: (e: Episode, f: string) => boolean;
  detail?: (e: Episode) => string | undefined;
  provenanceKey?: "transcript" | "corrected";
}

export type PipelineStepKey = PipelineStepDef["key"];
export type ActiveStep = PipelineStepKey | "info" | "search";

// ── Step registry ────────────────────────────────────────────────────────

export const PIPELINE_STEPS: PipelineStepDef[] = [
  {
    key: "transcribe",
    label: "Transcribe",
    rowLabel: "Transcribed",
    icon: Mic,
    section: "core",
    headerBadge: true,
    component: () => <TranscribePanel />,
    status: (e) => {
      if (e.transcribe_status === "outdated") return "partial";
      return e.transcribed ? "done" : false;
    },
    matchFiles: (_e, f) =>
      f.includes("transcript.") || f.includes("segments.") || f.includes("diarization.") || f.includes("speaker_map."),
    provenanceKey: "transcript",
  },
  {
    key: "correct",
    label: "Correct",
    rowLabel: "Corrected",
    icon: Sparkles,
    section: "core",
    headerBadge: true,
    component: () => <CorrectPanel />,
    status: (e) => {
      if (e.correct_status === "outdated") return "partial";
      return e.corrected ? "done" : false;
    },
    provenanceKey: "corrected",
  },
  {
    key: "index",
    label: "Index",
    rowLabel: "Indexed",
    icon: Database,
    section: "core",
    headerBadge: true,
    component: () => <IndexPanel />,
    status: (e) => (e.indexed ? "done" : false),
  },
  {
    key: "translate",
    label: "Translate",
    rowLabel: "Translated",
    icon: Languages,
    section: "bonus",
    headerBadge: false,
    component: () => <TranslatePanel />,
    status: (e) => {
      if (e.translate_status === "outdated") return "partial";
      return e.translations.length > 0 ? "done" : false;
    },
    matchFiles: (e, f) =>
      f.includes(".translated.") || e.translations.some((lang) => f.includes(`.${lang}.`)),
    detail: (e) => (e.translations.length > 0 ? e.translations.join(", ") : undefined),
  },
  {
    key: "synthesize",
    label: "Synthesize",
    rowLabel: "Synthesized",
    icon: AudioLines,
    section: "bonus",
    headerBadge: false,
    component: () => <SynthesizePanel />,
    status: (e) => (e.synthesized ? "done" : false),
    matchFiles: (_e, f) => f.includes(".synthesized."),
  },
];

export const STEP_BY_KEY: Record<PipelineStepKey, PipelineStepDef> = Object.fromEntries(
  PIPELINE_STEPS.map((s) => [s.key, s]),
) as Record<PipelineStepKey, PipelineStepDef>;

// ── Small components ─────────────────────────────────────────────────────

export function PipelineRow({ label, status, detail, provenance, files }: {
  label: string;
  status: "done" | "partial" | false;
  detail?: string;
  provenance?: Record<string, unknown>;
  files?: string[];
}) {
  const [expanded, setExpanded] = useState(false);
  const hasFiles = (files?.length ?? 0) > 0;
  const hasInfo = !!provenance || hasFiles;

  return (
    <div>
      <div className="flex items-center gap-3 text-sm">
        <span className={`w-2 h-2 rounded-full shrink-0 ${status === "done" ? "bg-success" : status === "partial" ? "bg-blue-500" : "bg-muted-foreground/30"}`} />
        <span className={status ? "text-foreground" : "text-muted-foreground"}>{label}</span>
        {detail && <span className="text-xs text-muted-foreground">{detail}</span>}
        {hasInfo && status && (
          <button onClick={() => setExpanded(!expanded)} className="text-xs text-muted-foreground hover:text-foreground transition">
            {hasFiles ? `${files!.length} file${files!.length !== 1 ? "s" : ""} ` : ""}{expanded ? "▾" : "▸"}
          </button>
        )}
      </div>
      {expanded && (
        <div className="mt-1 ml-5 space-y-0.5">
          {files?.map((f) => (
            <div key={f} className="text-xs text-muted-foreground font-mono truncate">{f}</div>
          ))}
        </div>
      )}
    </div>
  );
}

export function SidebarButton({ icon: Icon, label, expanded, onClick, active }: {
  icon: typeof Mic;
  label: string;
  expanded: boolean;
  onClick: () => void;
  active?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={expanded ? undefined : label}
      className={`w-full flex items-center gap-3 px-4 py-2.5 text-sm transition ${
        active
          ? "bg-accent text-accent-foreground"
          : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
      }`}
    >
      <Icon className="w-5 h-5 shrink-0" />
      {expanded && <span className="truncate">{label}</span>}
    </button>
  );
}

export function PipelineStatus({ episode }: { episode: Episode }) {
  const visible = PIPELINE_STEPS.filter((s) => s.headerBadge && s.status(episode));
  if (visible.length === 0) return null;
  return (
    <div className="flex gap-1.5">
      {visible.map((s) => {
        const status = s.status(episode);
        return (
          <span
            key={s.key}
            className={`text-2xs px-1.5 py-0.5 rounded-full ${
              status === "partial"
                ? "bg-blue-500/15 text-blue-500"
                : "bg-success/15 text-success"
            }`}
          >
            {s.rowLabel}
          </span>
        );
      })}
    </div>
  );
}

export function EpisodeDetails({ episode }: { episode: Episode }) {
  const youtubeUrl = /^[\w-]{11}$/.test(episode.id) ? `https://www.youtube.com/watch?v=${episode.id}` : null;
  if (episode.episode_number == null && !episode.pub_date && episode.duration <= 0 && !youtubeUrl) return null;
  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium">Details</h4>
      <div className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm max-w-md">
        {episode.episode_number != null && (
          <>
            <span className="text-muted-foreground">Episode</span>
            <span>#{episode.episode_number}</span>
          </>
        )}
        {episode.pub_date && (
          <>
            <span className="text-muted-foreground">Published</span>
            <span>{formatDate(episode.pub_date)}</span>
          </>
        )}
        {episode.duration > 0 && (
          <>
            <span className="text-muted-foreground">Duration</span>
            <span>{formatDuration(episode.duration)}</span>
          </>
        )}
        {youtubeUrl && (
          <>
            <span className="text-muted-foreground">Source</span>
            <a href={youtubeUrl} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline flex items-center gap-1">
              YouTube <ExternalLink className="w-3 h-3" />
            </a>
          </>
        )}
      </div>
    </div>
  );
}
