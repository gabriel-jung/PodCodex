/**
 * Pipeline step registry and small helper components used by EpisodePage.
 *
 * Extracted to keep EpisodePage focused on layout and data fetching.
 */

import { lazy, useEffect, useRef, useState, type ReactNode } from "react";
import type { Episode } from "@/api/types";
import { isEdited } from "@/lib/stepStatus";
import {
  Mic, Sparkles, Languages, AudioLines, Database,
} from "lucide-react";

const TranscribePanel = lazy(() => import("@/components/transcribe/TranscribePanel"));
const CorrectPanel = lazy(() => import("@/components/correct/CorrectPanel"));
const TranslatePanel = lazy(() => import("@/components/translate/TranslatePanel"));
const SynthesizePanel = lazy(() => import("@/components/synthesize/SynthesizePanel"));
const IndexPanel = lazy(() => import("@/components/index/IndexPanel"));

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
  return isEdited(provenance) ? "done" : "partial";
}

function translateStatus(e: Episode): StepStatus {
  if (e.translations.length === 0) return false;
  if (e.translate_status === "outdated") return "partial";
  // Prefer provenance of the currently-targeted lang (first translation as
  // fallback) so partial ↔ done tracks whether that version was edited.
  const lang = e.translations[0];
  return isEdited(e.provenance?.[lang]) ? "done" : "partial";
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
    status: translateStatus,
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
                ? "bg-info/15 text-info"
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
