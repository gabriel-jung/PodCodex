import { useEffect, useId, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { Episode } from "@/api/types";
import { getAllVersions } from "@/api/search";
import type { VersionEntry } from "@/api/types";
import { queryKeys } from "@/api/queryKeys";
import { usePipelineConfig } from "@/hooks/usePipelineConfig";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import { useApiKeys } from "@/hooks/useApiKeys";
import { useProviderProfiles } from "@/hooks/useProviderProfiles";
import {
  TRANSCRIBE_PRESETS,
  CPU_MODELS,
  GPU_MODELS,
  CPU_LABELS,
  GPU_LABELS,
  LLM_PRESETS,
  INDEX_PRESETS,
  usePipelineConfigStore,
} from "@/stores/pipelineConfigStore";
import { Button } from "@/components/ui/button";
import { Mic, Sparkles, Languages, Database, ChevronDown, Play, Copy, Check, Settings as SettingsIcon } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import { languageToISO, errorMessage, selectClass, cn, versionLabel, versionOption, stepTag, SUB_LANGUAGES } from "@/lib/utils";
import { modelPlaceholderFor, modelsFor } from "@/lib/providerModels";
import { INPUT_STEPS, filterVersionsForStep, type PipelineInputStep } from "@/lib/pipelineInputs";
import PresetCards from "@/components/common/PresetCards";
import SectionHeader from "@/components/common/SectionHeader";
import HelpLabel from "@/components/common/HelpLabel";
import { getCorrectManualPrompts, applyCorrectManual } from "@/api/correct";
import { getTranslateManualPrompts, applyTranslateManual } from "@/api/translate";

export type StepKey = PipelineInputStep;

/** True if the episode still needs work for the given step. */
export function episodeNeedsStep(ep: Episode, step: StepKey): boolean {
  switch (step) {
    case "transcribe": return ep.transcribe_status !== "done";
    case "correct":    return ep.correct_status !== "done";
    case "translate":  return ep.translate_status !== "done";
    case "index":      return !ep.indexed;
    default:           return true;
  }
}

/** Group key for the input source selector. */
interface SourceVariant {
  key: string;
  label: string;
  count: number;
  episodes: Episode[];
}

interface SourceGroup {
  key: string;
  label: string;
  count: number;
  episodes: Episode[];
  variants: SourceVariant[];
}

interface PromptBatch {
  batch_index: number;
  prompt: string;
  segment_count: number;
}

/** Find the label for a source key (group or variant). */
function findSourceLabel(groups: SourceGroup[], key: string): string | undefined {
  for (const g of groups) {
    if (g.key === key) return g.label;
    const v = g.variants.find((v) => v.key === key);
    if (v) return v.label;
  }
}

/** Stable key for an episode across the manual workflow. */
function epKey(ep: Episode): string {
  return ep.audio_path || ep.id;
}

/** Check if all batches for an episode have been validated. */
function isEpDone(
  ep: Episode,
  prompts: Record<string, PromptBatch[]>,
  results: Record<string, Record<number, unknown[]>>,
): boolean {
  const k = epKey(ep);
  const pr = prompts[k] || [];
  const re = results[k] || {};
  return pr.length > 0 && pr.every((_, i) => re[i] != null);
}

/** Build source groups with variants.
 *  Top level groups by step (corrected, transcript). Each group has variants
 *  (e.g. "Manual", "ollama") that users can drill into. */
function buildSourceGroups(
  episodes: Episode[],
  versionsMap: Record<string, VersionEntry[]>,
  step: StepKey,
): SourceGroup[] {
  if (step === "transcribe") return [];
  const stepPriority = INPUT_STEPS[step];
  if (!stepPriority.length) return [];

  // Two-level: step → variant label → episodes
  const stepGroups = new Map<string, Map<string, Episode[]>>();
  const stepEpisodes = new Map<string, Set<string>>(); // dedup per step

  for (const ep of episodes) {
    const versions = filterVersionsForStep(versionsMap[epKey(ep)] || [], step);
    const seenSteps = new Set<string>();
    const seenVariants = new Set<string>();

    for (const v of versions) {
      const s = v.step ?? "unknown";
      const vLabel = versionLabel(v);
      const variantKey = `${s}:${vLabel}`;

      // Add to step-level group (dedup per episode)
      if (!seenSteps.has(s)) {
        seenSteps.add(s);
        if (!stepEpisodes.has(s)) stepEpisodes.set(s, new Set());
        stepEpisodes.get(s)!.add(epKey(ep));
      }

      // Add to variant (dedup per episode per variant)
      if (!seenVariants.has(variantKey)) {
        seenVariants.add(variantKey);
        if (!stepGroups.has(s)) stepGroups.set(s, new Map());
        const variants = stepGroups.get(s)!;
        if (!variants.has(vLabel)) variants.set(vLabel, []);
        variants.get(vLabel)!.push(ep);
      }
    }
  }

  // Build output sorted by step priority
  const epLookup = new Map(episodes.map((e) => [epKey(e), e]));
  return stepPriority
    .filter((s) => stepEpisodes.has(s))
    .map((s) => {
      const variantMap = stepGroups.get(s) || new Map();
      const allEps = Array.from(stepEpisodes.get(s) || []);
      const variants: SourceVariant[] = Array.from(variantMap.entries())
        .map(([label, eps]) => ({ key: `${s}:${label}`, label, count: eps.length, episodes: eps }))
        .sort((a, b) => b.count - a.count);
      return {
        key: s,
        label: stepTag(s),
        count: allEps.length,
        episodes: allEps.map((k) => epLookup.get(k)!).filter(Boolean),
        variants,
      };
    });
}

const STEPS = [
  { key: "transcribe", label: "Transcribe", icon: Mic },
  { key: "correct", label: "Correct with AI", icon: Sparkles },
  { key: "translate", label: "Translate", icon: Languages },
  { key: "index", label: "Index", icon: Database },
] as const;

export { STEPS };

/** Toggle button with radio-style indicator and optional description. */
function ToggleButton({ checked, onClick, title, children, description }: {
  checked: boolean;
  onClick: () => void;
  title?: string;
  children: React.ReactNode;
  description?: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`w-full flex items-start gap-2.5 text-left text-sm px-3 py-2 rounded-md border transition ${
        checked ? "border-primary bg-accent" : "border-border hover:border-primary/50"
      }`}
    >
      <div className={`w-3.5 h-3.5 mt-0.5 rounded-full border-2 transition shrink-0 ${checked ? "border-primary bg-primary" : "border-muted-foreground"}`} />
      <div className="flex-1 min-w-0">
        <div className={checked ? "font-medium" : ""}>{children}</div>
        {description && <div className="text-xs text-muted-foreground mt-0.5 font-normal">{description}</div>}
      </div>
    </button>
  );
}

export type TranscribeSource = "audio" | "subtitles";

export default function StepConfigEditor({ step, episodes, showLanguage, onRun, onClose }: {
  step: StepKey;
  episodes: Episode[];
  showLanguage: string;
  onRun: (filteredEpisodes?: Episode[], sourceVersionIds?: Record<string, string>, transcribeSource?: TranscribeSource, force?: boolean) => void;
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { tc, setTc, llm, setLLM, targetLang, setTargetLang } = usePipelineConfig();
  const { whisperModels, detectedKeys: detected } = useLLMProviders();
  const { profiles } = useProviderProfiles();
  const { keys } = useApiKeys();
  const apiProfiles = useMemo(() => profiles.filter((p) => p.type !== "ollama"), [profiles]);
  const modelDatalistId = useId();
  const transcribePreset = usePipelineConfigStore((s) => s.transcribePreset);
  const applyTranscribePreset = usePipelineConfigStore((s) => s.applyTranscribePreset);
  const llmPreset = usePipelineConfigStore((s) => s.llmPreset);
  const applyLLMPreset = usePipelineConfigStore((s) => s.applyLLMPreset);
  const indexPreset = usePipelineConfigStore((s) => s.indexPreset);
  const applyIndexPreset = usePipelineConfigStore((s) => s.applyIndexPreset);
  const hasAnySubs = step === "transcribe" && episodes.some((e) => e.has_subtitles);
  const [transcribeSource, setTranscribeSource] = useState<TranscribeSource>(hasAnySubs ? "subtitles" : "audio");
  const [sourceOpen, setSourceOpen] = useState<boolean | null>(null); // null = auto from group count
  const [customVersions, setCustomVersions] = useState<Record<string, string>>({}); // epKey → versionId

  // Language chips for transcribe — tc.language is the source of truth;
  // otherMode only remembers whether the "Other" chip is active so an empty
  // custom code still keeps the input visible.
  const topLanguages = SUB_LANGUAGES.slice(0, 5);
  const currentLang = tc.language || "";
  const isTopChip = topLanguages.some((l) => l.code === currentLang);
  const [otherMode, setOtherMode] = useState(!!currentLang && !isTopChip);
  const chipSelected = otherMode ? "other" : isTopChip ? currentLang : "";

  // Manual batch workflow state
  const [manualActive, setManualActive] = useState(false); // true = episode-by-episode page
  const [manualBatchCounts, setManualBatchCounts] = useState<Record<string, number>>({});
  const [manualPrompts, setManualPrompts] = useState<Record<string, PromptBatch[]>>({});
  const [manualCurrentEp, setManualCurrentEp] = useState(0);
  const [manualCurrentBatch, setManualCurrentBatch] = useState(0);
  const [manualResults, setManualResults] = useState<Record<string, Record<number, unknown[]>>>({});
  const [manualPasted, setManualPasted] = useState("");
  const [manualParseError, setManualParseError] = useState<string | null>(null);
  const [manualCopied, setManualCopied] = useState(false);
  const [manualGenerating, setManualGenerating] = useState(false);
  const [manualApplying, setManualApplying] = useState(false);
  const [manualApplyingEp, setManualApplyingEp] = useState<string | null>(null);
  const [manualApplied, setManualApplied] = useState<Set<string>>(new Set());
  const [manualError, setManualError] = useState<string | null>(null);

  // Sync languages from show metadata
  useEffect(() => {
    if (showLanguage && isLLMStep && llm.sourceLang !== showLanguage) setLLM({ sourceLang: showLanguage });
    const iso = languageToISO(showLanguage);
    if (iso && !tc.language) setTc({ language: iso });
  }, [showLanguage]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectFull = cn(selectClass, "py-1.5 w-full");
  const inputFieldClass = "input py-1.5 text-sm w-full";

  const stepInfo = STEPS.find((s) => s.key === step)!;
  const Icon = stepInfo.icon;

  // Filter to episodes that need this step (prerequisites met + not already done)
  const canRun = useMemo(() => episodes.filter((e) => {
    switch (step) {
      case "transcribe":
        if (transcribeSource === "subtitles") return !!e.has_subtitles;
        return !!e.audio_path;
      case "correct":    return !!e.transcribed;
      case "translate":  return !!e.transcribed;
      case "index":      return !!e.transcribed;
      default:           return true;
    }
  }), [episodes, step, transcribeSource]);
  const needsWork = useMemo(() => canRun.filter((e) => episodeNeedsStep(e, step)), [canRun, step]);
  const needsWorkIds = useMemo(() => new Set(needsWork.map(epKey)), [needsWork]);
  const cantRun = episodes.length - canRun.length;
  const cantRunReason = step === "transcribe" ? "without audio" : "not transcribed";

  // Fetch all versions per episode (for source groups + custom picker)
  const isLLMStep = step === "correct" || step === "translate";
  const { data: epVersionsMap } = useQuery({
    queryKey: ["epVersions", step, canRun.map(epKey).join(",")],
    queryFn: async () => {
      const map: Record<string, VersionEntry[]> = {};
      await Promise.all(canRun.map(async (ep) => {
        if (!ep.audio_path && !ep.output_dir) return;
        try {
          map[epKey(ep)] = await getAllVersions(ep.audio_path, ep.output_dir);
        } catch { map[epKey(ep)] = []; }
      }));
      return map;
    },
    enabled: step !== "transcribe",
    staleTime: 30_000,
  });

  // Source groups for the input selector (correct, translate, index)
  const sourceGroups = useMemo(
    () => buildSourceGroups(canRun, epVersionsMap || {}, step),
    [canRun, epVersionsMap, step],
  );
  const [selectedSource, setSelectedSource] = useState<string | null>(null); // null = all
  const [expandedStep, setExpandedStep] = useState<string | null>(null);
  // Derive which step is actually expanded (clear stale state if sourceGroups changed)
  const resolvedExpanded = expandedStep && sourceGroups.some((g) => g.key === expandedStep) ? expandedStep : null;
  // Auto-expand when multiple source groups, collapse when only one
  const sourceOpenResolved = sourceOpen ?? sourceGroups.length > 1;
  const filteredEpisodes = useMemo(() => {
    if (selectedSource === "custom") return canRun;
    if (!selectedSource) return canRun;
    // Check if it's a variant key ("step:label") or step key ("step")
    for (const g of sourceGroups) {
      if (g.key === selectedSource) return g.episodes;
      const v = g.variants.find((v) => v.key === selectedSource);
      if (v) return v.episodes;
    }
    return canRun;
  }, [selectedSource, sourceGroups, canRun]);

  /** Apply manual corrections for a single completed episode, then invalidate queries. */
  const applyEpisode = async (ep: Episode, results: Record<string, Record<number, unknown[]>>, prompts: Record<string, PromptBatch[]>) => {
    const k = epKey(ep);
    const pr = prompts[k] || [];
    const re = results[k] || {};
    const corrections: unknown[] = [];
    for (let i = 0; i < pr.length; i++) {
      const batch = re[i];
      if (Array.isArray(batch)) corrections.push(...batch);
    }
    const params = { audio_path: ep.audio_path || undefined, output_dir: ep.output_dir || undefined, corrections } as Record<string, unknown>;
    if (step === "translate") {
      await applyTranslateManual({ ...params, lang: targetLang } as Parameters<typeof applyTranslateManual>[0]);
    } else {
      await applyCorrectManual(params as Parameters<typeof applyCorrectManual>[0]);
    }
    queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
  };

  // Derived values for auto-generate effect
  const currentEpKey = filteredEpisodes[manualCurrentEp] ? epKey(filteredEpisodes[manualCurrentEp]) : "";
  const currentBatchCount = manualBatchCounts[currentEpKey] ?? 1;

  // Auto-generate prompts when entering manual view, navigating episodes, or changing batch count
  useEffect(() => {
    if (!manualActive || !currentEpKey) return;
    const ep = filteredEpisodes[manualCurrentEp];
    if (!ep || (!ep.audio_path && !ep.output_dir)) return;
    let cancelled = false;
    const timeout = setTimeout(async () => {
      setManualGenerating(true);
      setManualError(null);
      try {
        const epDuration = ep.duration || 3600;
        const batchMins = Math.ceil(epDuration / 60 / currentBatchCount);
        const params: Record<string, unknown> = {
          source_lang: llm.sourceLang,
          batch_minutes: batchMins,
          context: llm.context || undefined,
        };
        if (ep.audio_path) params.audio_path = ep.audio_path;
        if (ep.output_dir) params.output_dir = ep.output_dir;
        const cvId = customVersions[currentEpKey];
        if (cvId) params.source_version_id = cvId;
        let result: PromptBatch[];
        if (step === "translate") {
          params.target_lang = targetLang;
          result = await getTranslateManualPrompts(params as Parameters<typeof getTranslateManualPrompts>[0]);
        } else {
          result = await getCorrectManualPrompts(params as Parameters<typeof getCorrectManualPrompts>[0]);
        }
        if (!cancelled) {
          setManualPrompts((prev) => ({ ...prev, [currentEpKey]: result }));
          setManualCurrentBatch(0);
        }
      } catch (e) {
        if (!cancelled) setManualError(errorMessage(e));
      } finally {
        if (!cancelled) setManualGenerating(false);
      }
    }, 300);
    return () => { cancelled = true; clearTimeout(timeout); };
  }, [manualActive, currentEpKey, currentBatchCount, customVersions[currentEpKey]]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div className="bg-background border border-border rounded-lg shadow-xl w-full max-w-md mx-4" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border">
          <Icon className="w-4 h-4" />
          <span className="text-sm font-semibold">{stepInfo.label}</span>
          <span className="text-xs text-muted-foreground">
            {filteredEpisodes.length === episodes.length
              ? `${episodes.length} episode${episodes.length !== 1 ? "s" : ""}`
              : `${filteredEpisodes.length} of ${episodes.length} episode${episodes.length !== 1 ? "s" : ""}`}
          </span>
          <div className="flex-1" />
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-lg leading-none" aria-label="Close">&times;</button>
        </div>

        <div className="px-5 py-4 space-y-4 max-h-[70vh] overflow-y-auto">
          {/* ── Transcribe source selector (audio vs subtitles) ── */}
          {step === "transcribe" && hasAnySubs && (
            <div className="space-y-1">
              <SectionHeader>Source</SectionHeader>
              <div className="space-y-1 pl-1">
                {episodes.some((e) => !!e.audio_path) && (
                  <button
                    onClick={() => setTranscribeSource("audio")}
                    className={`w-full flex items-center gap-2 text-xs px-2 py-1 rounded transition ${
                      transcribeSource === "audio" ? "bg-accent font-medium" : "hover:bg-accent/50"
                    }`}
                  >
                    <span className="flex-1 text-left">Audio</span>
                    <span className="tabular-nums text-muted-foreground">{episodes.filter((e) => !!e.audio_path).length}</span>
                  </button>
                )}
                <button
                  onClick={() => setTranscribeSource("subtitles")}
                  className={`w-full flex items-center gap-2 text-xs px-2 py-1 rounded transition ${
                    transcribeSource === "subtitles" ? "bg-accent font-medium" : "hover:bg-accent/50"
                  }`}
                >
                  <span className="flex-1 text-left">Subtitles</span>
                  <span className="tabular-nums text-muted-foreground">{episodes.filter((e) => e.has_subtitles).length}</span>
                </button>
              </div>
            </div>
          )}

          {/* ── Nothing to do ── */}
          {canRun.length === 0 && (
            <div className="py-4 text-center">
              <p className="text-sm text-muted-foreground">
                {manualApplied.size > 0
                  ? `Done — ${manualApplied.size} episode${manualApplied.size !== 1 ? "s" : ""} processed.`
                  : cantRun > 0
                    ? `All ${episodes.length} selected episode${episodes.length !== 1 ? "s are" : " is"} ${cantRunReason}.`
                    : "Nothing to process."}
              </p>
            </div>
          )}

          {/* ── Preset cards (transcribe) ── */}
          {canRun.length > 0 && step === "transcribe" && transcribeSource === "audio" && (
            <PresetCards presets={TRANSCRIBE_PRESETS} active={transcribePreset} onSelect={applyTranscribePreset} />
          )}

          {/* ── Input source selector ── */}
          {canRun.length > 0 && sourceGroups.length > 0 && step !== "transcribe" && !manualActive && selectedSource !== "custom" && (
            <div className="space-y-1.5">
              <button
                onClick={() => setSourceOpen(!sourceOpenResolved)}
                className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition w-full"
              >
                <ChevronDown className={`w-3 h-3 transition-transform ${sourceOpenResolved ? "" : "-rotate-90"}`} />
                <span>
                  {filteredEpisodes.length === 1 ? "Input transcript" : "Input transcripts"}
                  {!sourceOpenResolved && (
                    <span className="font-normal ml-1">
                      - {Object.keys(customVersions).length > 0 ? "Custom" : selectedSource
                        ? (findSourceLabel(sourceGroups, selectedSource) || selectedSource)
                        : "Latest"} ({filteredEpisodes.length})
                    </span>
                  )}
                </span>
              </button>
              {sourceOpenResolved && (
                <div className="space-y-0.5 pl-4">
                  <button
                    onClick={() => { setSelectedSource(null); setExpandedStep(null); }}
                    title={sourceGroups.map((g) => `${g.label}: ${g.count}`).join("\n")}
                    className={`w-full flex items-center gap-2 text-xs px-2 py-1 rounded transition ${
                      selectedSource === null ? "bg-accent font-medium" : "hover:bg-accent/50"
                    }`}
                  >
                    <span className="flex-1 text-left">Latest</span>
                    <span className="tabular-nums text-muted-foreground">{canRun.length}</span>
                  </button>
                  {sourceGroups.map((g) => {
                    const isSelected = selectedSource === g.key || g.variants.some((v) => v.key === selectedSource);
                    const isExpanded = resolvedExpanded === g.key;
                    const hasVariants = g.variants.length > 1;
                    return (
                      <div key={g.key}>
                        <button
                          onClick={() => {
                            setSelectedSource(g.key);
                            setExpandedStep(hasVariants && !isExpanded ? g.key : null);
                          }}
                          title={g.episodes.map((e) => e.title).join("\n")}
                          className={`w-full flex items-center gap-2 text-xs px-2 py-1 rounded transition ${
                            isSelected ? "bg-accent font-medium" : "hover:bg-accent/50"
                          }`}
                        >
                          {hasVariants && (
                            <ChevronDown className={`w-2.5 h-2.5 transition-transform ${isExpanded ? "" : "-rotate-90"}`} />
                          )}
                          <span className="flex-1 text-left capitalize">{g.label}</span>
                          <span className="tabular-nums text-muted-foreground">{g.count}</span>
                        </button>
                        {isExpanded && g.variants.map((v) => (
                          <button
                            key={v.key}
                            onClick={() => setSelectedSource(v.key)}
                            title={v.episodes.map((e) => e.title).join("\n")}
                            className={`w-full flex items-center gap-2 text-xs px-2 py-1 pl-7 rounded transition ${
                              selectedSource === v.key ? "bg-accent font-medium" : "hover:bg-accent/50"
                            }`}
                          >
                            <span className="flex-1 text-left">{v.label}</span>
                            <span className="tabular-nums text-muted-foreground">{v.count}</span>
                          </button>
                        ))}
                      </div>
                    );
                  })}
                  {isLLMStep && (
                    <button
                      onClick={() => { setSelectedSource("custom"); setCustomVersions({}); setExpandedStep(null); }}
                      className={`w-full flex items-center gap-2 text-xs px-2 py-1 rounded transition ${
                        selectedSource === "custom" ? "bg-accent font-medium" : "hover:bg-accent/50"
                      }`}
                    >
                      <span className="flex-1 text-left">Custom (pick per episode)</span>
                    </button>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ── Preset cards (index) ── */}
          {canRun.length > 0 && step === "index" && (
            <PresetCards presets={INDEX_PRESETS} active={indexPreset} onSelect={applyIndexPreset} />
          )}

          {/* ── Custom version picker ── */}
          {canRun.length > 0 && selectedSource === "custom" && !manualActive && epVersionsMap && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">Choose which version to use as input for each episode.</p>
              <div className="space-y-1">
                {canRun.map((ep) => {
                  const ek = epKey(ep);
                  const versions = filterVersionsForStep(epVersionsMap[ek] || [], step);
                  const sel = customVersions[ek] || "";
                  return (
                    <div key={ek} className="border border-border/50 rounded px-3 py-2 space-y-1.5">
                      <div className="text-sm font-medium truncate" title={ep.title}>{ep.title}</div>
                      {versions.length > 0 ? (
                        <select
                          value={sel}
                          onChange={(e) => setCustomVersions((prev) => ({ ...prev, [ek]: e.target.value }))}
                          className={cn(selectClass, "text-xs w-full")}
                        >
                          {versions.map((v, i) => (
                            <option key={v.id} value={i === 0 ? "" : v.id}>
                              {versionOption(v)}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <p className="text-2xs text-muted-foreground italic">No versions available</p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── Manual episode-by-episode workflow ── */}
          {canRun.length > 0 && isLLMStep && manualActive && (() => {
            const eps = filteredEpisodes;
            const ep = eps[manualCurrentEp];
            if (!ep) return null;
            const ek = epKey(ep);
            const batchCount = manualBatchCounts[ek] ?? 1;
            const dur = ep.duration ? `${Math.round(ep.duration / 60)} min` : "";
            const epPrompts = manualPrompts[ek] || [];
            const hasPrompts = epPrompts.length > 0;
            const batch = epPrompts[manualCurrentBatch];
            const epResults = manualResults[ek] || {};
            const batchDone = epResults[manualCurrentBatch] != null;
            const epDoneCount = epPrompts.filter((_, i) => epResults[i] != null).length;
            const epAllDone = hasPrompts && epDoneCount === epPrompts.length;
            const totalEpsDone = eps.filter((e) => manualApplied.has(epKey(e))).length;

            return (
              <div className="space-y-3">
                {/* Episode navigation */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => { setManualCurrentEp(Math.max(0, manualCurrentEp - 1)); setManualCurrentBatch(0); setManualPasted(""); setManualParseError(null); setManualCopied(false); }}
                    disabled={manualCurrentEp === 0}
                    className="text-muted-foreground hover:text-foreground disabled:opacity-30 transition"
                    aria-label="Previous episode"
                  >
                    <ChevronDown className="w-3.5 h-3.5 rotate-90" />
                  </button>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate" title={ep.title}>{ep.title}</div>
                    <div className="text-xs text-muted-foreground">
                      Episode {manualCurrentEp + 1} of {eps.length}
                      {dur && ` - ${dur}`}
                    </div>
                  </div>
                  <button
                    onClick={() => { setManualCurrentEp(Math.min(eps.length - 1, manualCurrentEp + 1)); setManualCurrentBatch(0); setManualPasted(""); setManualParseError(null); setManualCopied(false); }}
                    disabled={manualCurrentEp >= eps.length - 1}
                    className="text-muted-foreground hover:text-foreground disabled:opacity-30 transition"
                    aria-label="Next episode"
                  >
                    <ChevronDown className="w-3.5 h-3.5 -rotate-90" />
                  </button>
                </div>

                {/* Overall progress (when multiple episodes) */}
                {eps.length > 1 && (
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full rounded-full bg-primary transition-all"
                        style={{ width: `${(totalEpsDone / eps.length) * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-muted-foreground shrink-0">{totalEpsDone}/{eps.length} done</span>
                  </div>
                )}

                {/* Batch count slider - only show once prompts are loaded */}
                {hasPrompts && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <label className="text-sm font-medium" title="Number of chunks to split the transcript into. Each chunk becomes one prompt.">Batches</label>
                      <input
                        type="range"
                        min={1}
                        max={10}
                        value={batchCount}
                        onChange={(e) => { const v = Number(e.target.value); setManualBatchCounts((prev) => ({ ...prev, [ek]: v })); }}
                        className="flex-1 accent-primary"
                      />
                      <span className="text-sm tabular-nums w-4 text-center">{batchCount}</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Number of prompts to split the episode into. Large models like Claude Opus 4.6 or GPT-4o can handle hours-long episodes in one batch.
                    </p>
                  </div>
                )}

                {/* No transcript available */}
                {!ep.audio_path && !ep.output_dir && (
                  <p className="text-xs text-muted-foreground">No transcript available for this episode.</p>
                )}

                {/* Loading state */}
                {manualGenerating && (
                  <p className="text-xs text-muted-foreground">Generating prompt...</p>
                )}

                {/* Batch navigation (if multiple) */}
                {hasPrompts && epPrompts.length > 1 && (
                  <div className="flex items-center gap-1.5">
                    {epPrompts.map((_, i) => (
                      <button
                        key={i}
                        onClick={() => { setManualCurrentBatch(i); setManualPasted(""); setManualParseError(null); setManualCopied(false); }}
                        className={`w-6 h-6 rounded text-2xs font-medium transition ${
                          i === manualCurrentBatch
                            ? "bg-primary text-primary-foreground"
                            : epResults[i] != null
                              ? "bg-success/20 text-success border border-success/30"
                              : "bg-secondary text-muted-foreground border border-border"
                        }`}
                      >
                        {i + 1}
                      </button>
                    ))}
                    <span className="text-xs text-muted-foreground ml-1">{epDoneCount}/{epPrompts.length} batches</span>
                  </div>
                )}

                {/* Prompt display + copy */}
                {batch && (
                  <div className="border border-border rounded">
                    <div className="flex items-center justify-between px-3 py-1.5 bg-secondary/50 border-b border-border">
                      <span className="text-xs text-muted-foreground">
                        {epPrompts.length > 1 ? `Batch ${manualCurrentBatch + 1} - ` : ""}{batch.segment_count} segments
                        {batchDone && <span className="text-success ml-2">validated</span>}
                      </span>
                      <Button
                        onClick={async () => {
                          await navigator.clipboard.writeText(batch.prompt);
                          setManualCopied(true);
                          setTimeout(() => setManualCopied(false), 2000);
                        }}
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2"
                      >
                        {manualCopied ? <Check className="w-3 h-3 text-success" /> : <Copy className="w-3 h-3" />}
                      </Button>
                    </div>
                    <pre className="p-3 text-xs max-h-40 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                      {batch.prompt}
                    </pre>
                  </div>
                )}

                {/* Paste + validate */}
                {batch && !batchDone && (
                  <div className="space-y-2">
                    <textarea
                      value={manualPasted}
                      onChange={(e) => { setManualPasted(e.target.value); setManualParseError(null); }}
                      placeholder="Paste LLM JSON response here..."
                      className="input text-xs w-full resize-y"
                      rows={4}
                    />
                    {manualParseError && <p className="text-destructive text-xs">{manualParseError}</p>}
                    <Button
                      onClick={async () => {
                        setManualParseError(null);
                        try {
                          const parsed = JSON.parse(manualPasted);
                          const arr = Array.isArray(parsed) ? parsed : [parsed];
                          const newResults = { ...manualResults, [ek]: { ...epResults, [manualCurrentBatch]: arr } };
                          setManualResults(newResults);
                          setManualPasted("");
                          // Check if all batches for this episode are now done
                          const updatedEpResults = newResults[ek] || {};
                          const allBatchesDone = epPrompts.length > 0 && epPrompts.every((_, i) => updatedEpResults[i] != null);
                          if (allBatchesDone) {
                            // Auto-apply corrections for this episode
                            setManualApplyingEp(ek);
                            setManualError(null);
                            try {
                              await applyEpisode(ep, newResults, manualPrompts);
                              setManualApplied((prev) => new Set(prev).add(ek));
                              // Auto-advance to next incomplete episode
                              const nextEp = eps.findIndex((e, i) => i > manualCurrentEp && !isEpDone(e, manualPrompts, newResults) && !manualApplied.has(epKey(e)));
                              if (nextEp >= 0) {
                                setManualCurrentEp(nextEp);
                                setManualCurrentBatch(0);
                                setManualCopied(false);
                              }
                            } catch (e) {
                              setManualError(errorMessage(e));
                            } finally {
                              setManualApplyingEp(null);
                            }
                          } else {
                            // Auto-advance to next incomplete batch within this episode
                            const nextBatch = epPrompts.findIndex((_, i) => i > manualCurrentBatch && updatedEpResults[i] == null);
                            if (nextBatch >= 0) {
                              setManualCurrentBatch(nextBatch);
                            }
                          }
                        } catch (e) {
                          setManualParseError(`Invalid JSON: ${errorMessage(e)}`);
                        }
                      }}
                      disabled={!manualPasted.trim() || manualApplyingEp === ek}
                      size="sm"
                    >
                      Validate
                    </Button>
                  </div>
                )}

                {batch && batchDone && !epAllDone && (
                  <div className="flex items-center gap-2 text-xs">
                    <Check className="w-3.5 h-3.5 text-success" />
                    <span className="text-success">Validated ({(epResults[manualCurrentBatch] as unknown[]).length} segments)</span>
                    <Button
                      onClick={() => {
                        const next = { ...epResults };
                        delete next[manualCurrentBatch];
                        setManualResults({ ...manualResults, [ek]: next });
                      }}
                      variant="ghost"
                      size="sm"
                      className="h-6 text-xs text-muted-foreground"
                    >
                      Redo
                    </Button>
                  </div>
                )}

                {manualApplyingEp === ek && (
                  <p className="text-xs text-muted-foreground">Applying corrections...</p>
                )}
                {manualApplied.has(ek) && (
                  <div className="flex items-center gap-2 text-xs text-success">
                    <Check className="w-3.5 h-3.5" />
                    Applied
                  </div>
                )}

                {manualError && <p className="text-destructive text-xs">{manualError}</p>}
              </div>
            );
          })()}

          {/* ── LLM mode + config (correct / translate) ── */}
          {canRun.length > 0 && isLLMStep && !manualActive && selectedSource !== "custom" && (
            <>
              <PresetCards presets={LLM_PRESETS} active={llmPreset} onSelect={applyLLMPreset} />

              {/* Language fields (always visible) */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <HelpLabel label="Source language" help="Language of the original transcript." />
                  <input value={llm.sourceLang} onChange={(e) => setLLM({ sourceLang: e.target.value })} className={inputFieldClass} />
                </div>
                {step === "translate" && (
                  <div className="space-y-1.5">
                    <HelpLabel label="Target language" help="Language to translate into." />
                    <input value={targetLang} onChange={(e) => setTargetLang(e.target.value)} className={inputFieldClass} />
                  </div>
                )}
              </div>

              {/* Inline config for local / cloud */}
              {llm.mode !== "manual" && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <HelpLabel label="Model" help="LLM model name to use for processing." />
                      <input
                        value={llm.model}
                        onChange={(e) => setLLM({ model: e.target.value })}
                        list={llm.mode === "api" && modelsFor(llm.providerProfile).length > 0 ? modelDatalistId : undefined}
                        placeholder={llm.mode === "api" ? modelPlaceholderFor(llm.providerProfile) : "e.g. gpt-4o-mini"}
                        className={inputFieldClass}
                      />
                      {llm.mode === "api" && modelsFor(llm.providerProfile).length > 0 && (
                        <datalist id={modelDatalistId}>
                          {modelsFor(llm.providerProfile).map((m) => (
                            <option key={m} value={m} />
                          ))}
                        </datalist>
                      )}
                    </div>
                    <div className="space-y-1.5">
                      <HelpLabel label="Batch (min)" help="Minutes of transcript per LLM request. Larger batches are faster but need more context. Large models (e.g. Opus, GPT-4o) can handle a full episode at once." />
                      <input type="number" value={llm.batchMinutes} onChange={(e) => setLLM({ batchMinutes: Number(e.target.value) })} min={1} step={5} className={inputFieldClass} />
                    </div>
                  </div>
                  {llm.mode === "api" && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <HelpLabel label="Provider" help="Pick a profile (built-in or custom). Manage in Settings → Credentials." />
                        <select
                          value={llm.providerProfile}
                          onChange={(e) => setLLM({ providerProfile: e.target.value })}
                          className={selectFull}
                        >
                          <option value="">Pick a profile…</option>
                          {apiProfiles.map((p) => (
                            <option key={p.name} value={p.name}>
                              {p.name}{p.builtin ? "" : " (custom)"}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="space-y-1.5">
                        <HelpLabel label="LLM API key" help="Pick a key from the pool. Add keys in Settings → Credentials." />
                        <select
                          value={llm.keyName}
                          onChange={(e) => setLLM({ keyName: e.target.value })}
                          className={selectFull}
                        >
                          <option value="">
                            {keys.length === 0 ? "No keys yet" : "Pick a key…"}
                          </option>
                          {keys.map((k) => (
                            <option key={k.name} value={k.name}>
                              {k.name}
                              {k.suggested_provider ? ` — ${k.suggested_provider}` : ""}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  )}
                </div>
              )}

            </>
          )}


          {/* ── Settings (transcribe / index) ── */}
          {canRun.length > 0 && !isLLMStep && !(step === "transcribe" && transcribeSource === "subtitles") && (
            <div className="space-y-4">
              {step === "transcribe" && (() => {
                const isCpu = transcribePreset === "cpu" || (transcribePreset === "" && CPU_MODELS.has(tc.modelSize));
                const entries = Object.keys(whisperModels).length > 0
                  ? Object.entries(whisperModels)
                  : [[tc.modelSize, tc.modelSize] as [string, string]];
                const filtered = isCpu
                  ? entries.filter(([key]) => CPU_MODELS.has(key))
                  : entries.filter(([key]) => GPU_MODELS.has(key));
                return (
                  <>
                    <div className="space-y-1.5">
                      <HelpLabel label="Model" help="Speech recognition model. Bigger models make fewer mistakes but are slower and need more GPU memory." />
                      <select value={tc.modelSize} onChange={(e) => setTc({ modelSize: e.target.value })} className={selectFull}>
                        {filtered.map(([key, label]) => (
                          <option key={key} value={key}>{key} — {isCpu ? CPU_LABELS[key] || label : GPU_LABELS[key] || label}</option>
                        ))}
                      </select>
                    </div>

                    <div className="space-y-1.5">
                      <HelpLabel label="Language" help="The spoken language of the audio. Auto-detect works for most cases; setting it explicitly improves accuracy and word-level alignment." />
                      <div className="flex flex-wrap gap-1.5">
                        {([{ code: "", label: "Auto" }, ...topLanguages] as const).map((l) => (
                          <button
                            key={l.code || "auto"}
                            onClick={() => { setTc({ language: l.code }); setOtherMode(false); }}
                            className={`px-2.5 py-1 text-xs rounded-md border transition ${chipSelected === l.code ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"}`}
                          >
                            {l.label}
                          </button>
                        ))}
                        <button
                          onClick={() => { setOtherMode(true); if (isTopChip) setTc({ language: "" }); }}
                          className={`px-2.5 py-1 text-xs rounded-md border transition ${chipSelected === "other" ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"}`}
                        >
                          Other
                        </button>
                        {chipSelected === "other" && (
                          <input
                            value={tc.language}
                            onChange={(e) => setTc({ language: e.target.value.toLowerCase().slice(0, 5) })}
                            placeholder="ISO code (e.g. ja, zh, ar)"
                            className="input text-xs w-36"
                            autoFocus
                          />
                        )}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <SectionHeader>Options</SectionHeader>
                      <ToggleButton
                        checked={tc.clean}
                        onClick={() => setTc({ clean: !tc.clean })}
                        title="Removes hallucinated segments using character density filters (< 2 or > 75 chars/s)"
                        description="Drop hallucinated or unnaturally dense segments"
                      >
                        Clean transcript
                      </ToggleButton>
                      {!isCpu && (
                        <>
                          <ToggleButton
                            checked={tc.diarize}
                            onClick={() => setTc({ diarize: !tc.diarize })}
                            title="Label who speaks in each segment using pyannote diarization"
                            description="Label who speaks in each segment"
                          >
                            Speaker identification
                          </ToggleButton>
                          {tc.diarize && !(detected.hf_token || tc.hfToken) && (
                            <div className="pl-6 text-xs text-muted-foreground flex items-center gap-1.5">
                              <span>HuggingFace token needed.</span>
                              <button
                                type="button"
                                onClick={() => navigate({ to: "/settings", search: { tab: "credentials" }, hash: "HF_TOKEN" })}
                                className="inline-flex items-center gap-1 text-primary hover:underline font-medium"
                              >
                                <SettingsIcon className="w-3 h-3" />
                                Set it up in Credentials
                              </button>
                            </div>
                          )}
                        </>
                      )}
                    </div>

                  </>
                );
              })()}
            </div>
          )}
        </div>

        {/* Skip info (when some episodes can't run) */}
        {cantRun > 0 && canRun.length > 0 && (
          <div className="px-5 pb-1 text-xs text-muted-foreground">
            {cantRun} {cantRunReason}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-5 py-3 border-t border-border">
          {/* Back button when in sub-page (manual or custom) */}
          {manualActive && (
            <Button onClick={() => { setManualActive(false); setManualPrompts({}); setManualResults({}); setManualCurrentEp(0); setManualCurrentBatch(0); }} variant="ghost" size="sm">
              Back
            </Button>
          )}
          {selectedSource === "custom" && !manualActive && (
            <Button onClick={() => setSelectedSource(null)} variant="ghost" size="sm">
              Back
            </Button>
          )}
          {selectedSource !== "custom" && !manualActive && (
            <Button onClick={onClose} variant="ghost" size="sm">
              {canRun.length === 0 ? "Close" : "Cancel"}
            </Button>
          )}

          {/* Manual mode: "Next" on step 1, "Generate"/"Apply" on step 2 */}
          {filteredEpisodes.length > 0 && isLLMStep && llm.mode === "manual" && !manualActive && selectedSource !== "custom" && (
            <Button onClick={() => { setManualActive(true); setManualCurrentEp(0); setManualCurrentBatch(0); }} size="sm">
              {stepInfo.label} {filteredEpisodes.length} episode{filteredEpisodes.length !== 1 ? "s" : ""}
            </Button>
          )}
          {filteredEpisodes.length > 0 && isLLMStep && manualActive && (() => {
            const allApplied = filteredEpisodes.every((e) => manualApplied.has(epKey(e)));
            if (allApplied) return <Button onClick={onClose} size="sm">Done</Button>;
            // Fallback: retry any completed-but-not-applied episodes
            const retryEps = filteredEpisodes.filter((e) => isEpDone(e, manualPrompts, manualResults) && !manualApplied.has(epKey(e)));
            if (retryEps.length === 0) return null;
            return (
              <Button
                onClick={async () => {
                  setManualApplying(true);
                  setManualError(null);
                  try {
                    for (const e of retryEps) {
                      await applyEpisode(e, manualResults, manualPrompts);
                      setManualApplied((prev) => new Set(prev).add(epKey(e)));
                    }
                  } catch (e) {
                    setManualError(errorMessage(e));
                  } finally {
                    setManualApplying(false);
                  }
                }}
                disabled={manualApplying}
                size="sm"
              >
                {manualApplying ? "Applying..." : `Retry ${retryEps.length} failed`}
              </Button>
            );
          })()}

          {/* Custom mode: confirm selection and go back to config */}
          {selectedSource === "custom" && !manualActive && (
            <Button onClick={() => setSelectedSource(null)} size="sm">
              <Check className="w-3.5 h-3.5 mr-1" />
              Confirm versions
            </Button>
          )}

          {/* Non-manual run button */}
          {filteredEpisodes.length > 0 && !(isLLMStep && llm.mode === "manual") && !manualActive && selectedSource !== "custom" && (() => {
            const pending = filteredEpisodes.filter((e) => needsWorkIds.has(epKey(e)));
            const runWith = (force?: boolean) => {
              let vids = Object.keys(customVersions).length > 0 ? customVersions : undefined;
              if (!vids && selectedSource && epVersionsMap) {
                const sourceVids: Record<string, string> = {};
                const isVariant = !sourceGroups.some((g) => g.key === selectedSource);
                for (const ep of filteredEpisodes) {
                  const versions = filterVersionsForStep(epVersionsMap[epKey(ep)] || [], step);
                  const match = isVariant
                    ? versions.find((v) => `${v.step}:${versionLabel(v)}` === selectedSource)
                    : versions.find((v) => v.step === selectedSource);
                  if (match) sourceVids[epKey(ep)] = match.id;
                }
                if (Object.keys(sourceVids).length > 0) vids = sourceVids;
              }
              onRun(
                selectedSource || transcribeSource === "subtitles" ? filteredEpisodes : undefined,
                vids,
                step === "transcribe" ? transcribeSource : undefined,
                force,
              );
            };
            return (
              <>
                <Button onClick={() => runWith(true)} variant="ghost" size="sm" title="Reprocess all episodes, replacing existing results">
                  Reprocess all
                </Button>
                {pending.length > 0 ? (
                  <Button onClick={() => runWith()} size="sm">
                    <Play className="w-3.5 h-3.5 mr-1" />
                    {stepInfo.label} {pending.length} episode{pending.length !== 1 ? "s" : ""}
                  </Button>
                ) : (
                  <span className="flex items-center gap-1.5 text-xs text-success">
                    <Check className="w-3.5 h-3.5" /> All up to date
                  </span>
                )}
              </>
            );
          })()}
        </div>
      </div>
    </div>
  );
}
