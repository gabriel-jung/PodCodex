/**
 * Pipeline step registry and small helper components used by EpisodePage.
 *
 * Extracted to keep EpisodePage focused on layout and data fetching.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Episode } from "@/api/types";
import { isManualEdit } from "@/lib/stepStatus";
import {
  Mic, Sparkles, Languages, AudioLines, Database,
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
  component: () => ReactNode;
  status: (e: Episode) => StepStatus;
  matchFiles?: (e: Episode, f: string) => boolean;
  detail?: (e: Episode) => string | undefined;
  provenanceKey?: "transcript" | "corrected";
}

/** Dot color tracks review progress, not freshness: a hand-edited version
 *  is "done" (green, reviewed), an untouched raw transcript/correction is
 *  "partial" (blue, needs review), absent is `false` (gray). */
function editedStatus(present: boolean, provenance: unknown): StepStatus {
  if (!present) return false;
  return isManualEdit(provenance) ? "done" : "partial";
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
    status: (e) => editedStatus(!!e.transcribed, e.provenance?.transcript),
    // Kept legacy flat-file patterns (`.transcript.`, `.segments.`, …) so
    // pre-version-DB episodes still surface their artifacts.
    matchFiles: (_e, f) =>
      f.includes("/transcript/") ||
      f.includes("/speaker_map/") ||
      f.includes(".transcript.") ||
      f.includes(".segments.") ||
      f.includes(".diarization.") ||
      f.includes(".diarized_segments.") ||
      f.includes(".speaker_map."),
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
    status: (e) => editedStatus(!!e.corrected, e.provenance?.corrected),
    matchFiles: (_e, f) => f.includes("/corrected/") || f.includes(".corrected."),
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
      f.includes(".translated.") ||
      e.translations.some((lang) => f.includes(`/${lang}/`) || f.includes(`.${lang}.`)),
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
    matchFiles: (_e, f) => f.includes("/synthesized/") || f.includes(".synthesized."),
  },
];

export const STEP_BY_KEY: Record<PipelineStepKey, PipelineStepDef> = Object.fromEntries(
  PIPELINE_STEPS.map((s) => [s.key, s]),
) as Record<PipelineStepKey, PipelineStepDef>;

// ── Small components ─────────────────────────────────────────────────────

export function PipelineRow({ label, status, detail, subtitle, provenance, files, onClick }: {
  label: string;
  status: "done" | "partial" | false;
  detail?: string;
  subtitle?: string;
  provenance?: Record<string, unknown>;
  files?: string[];
  onClick?: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const hasFiles = (files?.length ?? 0) > 0;
  const hasInfo = !!provenance || hasFiles;

  const LabelTag = onClick ? "button" : "span";
  return (
    <div>
      <div className="flex items-center gap-3 text-sm">
        <span className={`w-2 h-2 rounded-full shrink-0 ${status === "done" ? "bg-success" : status === "partial" ? "bg-blue-500" : "bg-muted-foreground/30"}`} />
        <span className="flex flex-col">
          <span className="flex items-center gap-2">
            <LabelTag
              onClick={onClick}
              className={`${status ? "text-foreground" : "text-muted-foreground"} ${onClick ? "hover:underline cursor-pointer" : ""}`}
            >
              {label}
            </LabelTag>
            {detail && <span className="text-xs text-muted-foreground">{detail}</span>}
            {hasInfo && status && (
              <button onClick={() => setExpanded(!expanded)} className="text-xs text-muted-foreground hover:text-foreground transition">
                {hasFiles ? `${files!.length} file${files!.length !== 1 ? "s" : ""} ` : ""}{expanded ? "▾" : "▸"}
              </button>
            )}
          </span>
          {subtitle && <span className="text-2xs text-muted-foreground">{subtitle}</span>}
        </span>
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

export function PipelineStatus({ episode }: { episode: Episode }) {
  const prevStatus = useRef<Partial<Record<PipelineStepKey, StepStatus>>>({});
  const timers = useRef<Partial<Record<PipelineStepKey, ReturnType<typeof setTimeout>>>>({});
  const [flashing, setFlashing] = useState<Set<PipelineStepKey>>(new Set());

  useEffect(() => {
    const justCompleted: PipelineStepKey[] = [];
    for (const s of PIPELINE_STEPS) {
      const prev = prevStatus.current[s.key];
      const curr = s.status(episode);
      if (prev !== undefined && prev !== "done" && curr === "done") {
        justCompleted.push(s.key);
      }
      prevStatus.current[s.key] = curr;
    }
    if (justCompleted.length === 0) return;
    setFlashing((f) => new Set([...f, ...justCompleted]));
    // Per-key timers so simultaneous or staggered completions don't clobber
    // each other's cleanup.
    for (const k of justCompleted) {
      if (timers.current[k]) clearTimeout(timers.current[k]);
      timers.current[k] = setTimeout(() => {
        setFlashing((f) => {
          const n = new Set(f);
          n.delete(k);
          return n;
        });
        delete timers.current[k];
      }, 900);
    }
  }, [episode]);

  useEffect(() => () => {
    for (const t of Object.values(timers.current)) if (t) clearTimeout(t);
  }, []);

  const visible = PIPELINE_STEPS.filter((s) => s.headerBadge && s.status(episode));
  if (visible.length === 0) return null;
  return (
    <div className="flex gap-1.5">
      {visible.map((s) => {
        const status = s.status(episode);
        const isFlashing = flashing.has(s.key);
        return (
          <span
            key={s.key}
            className={`text-2xs px-1.5 py-0.5 rounded-full ${
              status === "partial"
                ? "bg-blue-500/15 text-blue-500"
                : "bg-success/15 text-success"
            } ${isFlashing ? "complete-flash" : ""}`}
          >
            {s.rowLabel}
          </span>
        );
      })}
    </div>
  );
}
