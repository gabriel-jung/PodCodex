import { useEffect, useMemo, useState, useId } from "react";
import { confirmDiscard, useDirtyEdit } from "@/lib/dirtyEdits";
import { countLabel } from "@/lib/showCounts";
import { useQuery } from "@tanstack/react-query";
import type { Episode } from "@/api/types";
import { getShowMeta, getShowVersions } from "@/api/shows";
import type { VersionEntry } from "@/api/types";
import { queryKeys } from "@/api/queryKeys";
import { getEpisodeStem } from "@/lib/episodeRef";
import { usePipelineConfig } from "@/hooks/usePipelineConfig";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import {
  TRANSCRIBE_PRESETS,
  CPU_MODELS,
  GPU_MODELS,
  CPU_LABELS,
  GPU_LABELS,
  INDEX_PRESETS,
  isCpuTranscribe,
  modeToPreset,
  usePipelineConfigStore,
} from "@/stores/pipelineConfigStore";
import LLMControlsForm from "@/components/common/LLMControlsForm";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Mic, Sparkles, Languages, Database, ChevronDown, Play, Check, Settings as SettingsIcon } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import { languageToISO, selectClass, cn, versionLabel, stepTag, SUB_LANGUAGES } from "@/lib/utils";
import { INPUT_STEPS, filterVersionsForStep, sortVersionsForDefault, type PipelineInputStep } from "@/lib/pipelineInputs";
import { getEpisodeBatchPath } from "@/lib/episodeRef";
import PresetCards from "@/components/common/PresetCards";
import SectionHeader from "@/components/common/SectionHeader";
import HelpLabel from "@/components/common/HelpLabel";
import VersionPicker from "@/components/common/VersionPicker";
import { epKey, useManualBatch } from "./useManualBatch";
import ManualBatchView from "./ManualBatchView";
import { episodeCanRunStep, episodeNeedsStep } from "@/lib/stepStatus";

export type StepKey = PipelineInputStep;

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

/** Find the label for a source key (group or variant). */
function findSourceLabel(groups: SourceGroup[], key: string): string | undefined {
  for (const g of groups) {
    if (g.key === key) return g.label;
    const v = g.variants.find((v) => v.key === key);
    if (v) return v.label;
  }
}

/**
 * Re-key a per-episode map from the local `epKey` space into the batch API's
 * `audio_paths` space.
 *
 * `epKey` falls back to `ep.id` for episodes with no audio, but the batch API
 * identifies those by `<output_dir>.virtual`. Sending the former means the
 * backend's `source_version_ids` lookup misses and it quietly falls back to
 * its own default version, so subtitle-only episodes would ignore the user's
 * pick. See `getEpisodeBatchPath`.
 */
function toBatchKeys(
  byEpKey: Record<string, string>,
  episodes: Episode[],
): Record<string, string> | undefined {
  const out: Record<string, string> = {};
  for (const ep of episodes) {
    const versionId = byEpKey[epKey(ep)];
    const key = getEpisodeBatchPath(ep);
    if (versionId && key) out[key] = versionId;
  }
  return Object.keys(out).length > 0 ? out : undefined;
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
  { key: "index", label: "Index", icon: Database },
  { key: "translate", label: "Translate", icon: Languages },
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

export default function StepConfigEditor({ step, episodes, folder, showLanguage, onRun, onClose }: {
  step: StepKey;
  episodes: Episode[];
  /** Show folder, for the one folder-scoped versions request. */
  folder: string;
  showLanguage: string;
  onRun: (filteredEpisodes?: Episode[], sourceVersionIds?: Record<string, string>, transcribeSource?: TranscribeSource, force?: boolean) => void;
  onClose: () => void;
}) {
  const fieldId = useId();
  const navigate = useNavigate();
  const { tc, setTc, llm, setLLM, targetLang, setTargetLang } = usePipelineConfig();
  const { whisperModels, detectedKeys: detected } = useLLMProviders();
  const { data: showMeta } = useQuery({
    queryKey: queryKeys.showMeta(folder),
    queryFn: () => getShowMeta(folder),
  });
  const transcribePreset = usePipelineConfigStore((s) => s.transcribePreset);
  const applyTranscribePreset = usePipelineConfigStore((s) => s.applyTranscribePreset);
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

  // Sync languages from show metadata
  useEffect(() => {
    if (showLanguage && isLLMStep && llm.sourceLang !== showLanguage) setLLM({ sourceLang: showLanguage });
    const iso = languageToISO(showLanguage);
    if (iso && !tc.language) setTc({ language: iso });
  }, [showLanguage]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectFull = cn(selectClass, "py-1.5 w-full");

  const stepInfo = STEPS.find((s) => s.key === step)!;
  const Icon = stepInfo.icon;

  // Filter to episodes that need this step (prerequisites met + not already done)
  const canRun = useMemo(
    () => episodes.filter((e) => episodeCanRunStep(e, step, transcribeSource)),
    [episodes, step, transcribeSource],
  );
  const needsWork = useMemo(() => canRun.filter((e) => episodeNeedsStep(e, step)), [canRun, step]);
  const needsWorkIds = useMemo(() => new Set(needsWork.map(epKey)), [needsWork]);
  const cantRun = episodes.length - canRun.length;
  const cantRunReason = step === "transcribe" ? "without audio" : "not transcribed";

  // Fetch all versions per episode (for source groups + custom picker)
  const isLLMStep = step === "correct" || step === "translate";
  // One folder-scoped request, not one per selected episode: this used to
  // fan out a GET per episode (300 of them on a 300-episode batch, each
  // hitting the same SQLite file) before the source picker could render,
  // and the key embedded every episode path, so changing the selection
  // re-ran the whole fan-out.
  const { data: versionsByStem } = useQuery({
    queryKey: queryKeys.showVersions(folder),
    queryFn: () => getShowVersions(folder),
    enabled: step !== "transcribe",
    staleTime: 30_000,
  });

  const epVersionsMap = useMemo(() => {
    const map: Record<string, VersionEntry[]> = {};
    if (!versionsByStem) return map;
    for (const ep of canRun) {
      map[epKey(ep)] = versionsByStem[getEpisodeStem(ep)] ?? [];
    }
    return map;
  }, [versionsByStem, canRun]);

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

  // The source version pinned per episode (`epKey`): picked one by one under
  // "Custom", else the group or variant picked under "Input transcripts".
  // Shared by the auto run and the manual flow, so both read the same text.
  const pinnedVersions = useMemo(() => {
    if (Object.keys(customVersions).length > 0) return customVersions;
    if (!selectedSource || !epVersionsMap) return {};
    const out: Record<string, string> = {};
    const isVariant = !sourceGroups.some((g) => g.key === selectedSource);
    for (const ep of filteredEpisodes) {
      const versions = sortVersionsForDefault(
        filterVersionsForStep(epVersionsMap[epKey(ep)] || [], step),
        step,
        ep.verified ?? null,
      );
      const match = isVariant
        ? versions.find((v) => `${v.step}:${versionLabel(v)}` === selectedSource)
        : versions.find((v) => v.step === selectedSource);
      if (match) out[epKey(ep)] = match.id;
    }
    return out;
  }, [customVersions, selectedSource, epVersionsMap, sourceGroups, filteredEpisodes, step]);

  const manual = useManualBatch({
    step: step === "translate" ? "translate" : "correct",
    episodes: filteredEpisodes,
    folder,
    showMeta,
    sourceLang: llm.sourceLang,
    targetLang,
    sourceVersions: pinnedVersions,
  });

  // Pasted replies and validated batches exist only in this dialog until
  // they are applied; Escape, the close button and Cancel all ask first.
  useDirtyEdit(manual.dirty, "pasted LLM batches");
  const requestClose = () => {
    if (!manual.dirty) {
      onClose();
      return;
    }
    void confirmDiscard(["pasted LLM batches"]).then((discard) => {
      if (discard) onClose();
    });
  };

  return (
    // Radix brings the dialog role, focus trap and return, and Escape;
    // every close path still goes through requestClose (pasted batches).
    <Dialog open onOpenChange={(open) => { if (!open) requestClose(); }}>
      <DialogContent showCloseButton={false} aria-describedby={undefined} className="p-0 gap-0 sm:max-w-md">
        {/* Header */}
        <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border">
          <Icon className="w-4 h-4" />
          <DialogTitle className="text-sm font-semibold">{stepInfo.label}</DialogTitle>
          <span className="text-xs text-muted-foreground">
            {filteredEpisodes.length === episodes.length
              ? `${countLabel(episodes.length, "episode")}`
              : `${filteredEpisodes.length} of ${countLabel(episodes.length, "episode")}`}
          </span>
          <div className="flex-1" />
          <button onClick={requestClose} className="text-muted-foreground hover:text-foreground text-lg leading-none" aria-label="Close">&times;</button>
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
                {manual.applied.size > 0
                  ? `Done: ${countLabel(manual.applied.size, "episode")} processed.`
                  : cantRun > 0
                    ? `${episodes.length === 1 ? "The selected episode is" : `All ${countLabel(episodes.length, "selected episode")} are`} ${cantRunReason}.`
                    : "Nothing to process."}
              </p>
            </div>
          )}

          {/* ── Preset cards (transcribe) ── */}
          {canRun.length > 0 && step === "transcribe" && transcribeSource === "audio" && (
            <PresetCards presets={TRANSCRIBE_PRESETS} active={transcribePreset} onSelect={applyTranscribePreset} />
          )}

          {/* ── Input source selector ── */}
          {canRun.length > 0 && sourceGroups.length > 0 && step !== "transcribe" && !manual.active && selectedSource !== "custom" && (
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
          {canRun.length > 0 && selectedSource === "custom" && !manual.active && epVersionsMap && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">Choose which version to use as input for each episode.</p>
              <div className="space-y-1">
                {canRun.map((ep) => {
                  const ek = epKey(ep);
                  const versions = sortVersionsForDefault(
                    filterVersionsForStep(epVersionsMap[ek] || [], step),
                    step,
                    ep.verified ?? null,
                  );
                  const sel = customVersions[ek] || "";
                  return (
                    <div key={ek} className="border border-border/50 rounded px-3 py-2 space-y-1.5">
                      <div className="text-sm font-medium truncate" title={ep.title}>{ep.title}</div>
                      <VersionPicker
                        versions={versions}
                        value={sel || null}
                        onChange={(v) =>
                          setCustomVersions((prev) => ({ ...prev, [ek]: v ?? "" }))
                        }
                        emptyMessage="No versions available"
                        className={cn(selectClass, "text-xs w-full")}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── Manual episode-by-episode workflow ── */}
          {canRun.length > 0 && isLLMStep && manual.active && (
            <ManualBatchView m={manual} episodes={filteredEpisodes} />
          )}

          {/* ── LLM mode + config (correct / translate) ── */}
          {canRun.length > 0 && isLLMStep && !manual.active && selectedSource !== "custom" && (
            <>
              <LLMControlsForm
                config={llm}
                patch={setLLM}
                activePreset={modeToPreset(llm.mode)}
                showBatches={llm.mode !== "manual"}
                languageRows={
                  <>
                    <HelpLabel label="Source language" htmlFor={`${fieldId}-0`} help="Language of the original transcript." />
                    <input id={`${fieldId}-0`} value={llm.sourceLang} onChange={(e) => setLLM({ sourceLang: e.target.value })} className="input" />
                    {step === "translate" && (
                      <>
                        <HelpLabel label="Target language" htmlFor={`${fieldId}-1`} help="Language to translate into." />
                        <input id={`${fieldId}-1`} value={targetLang} onChange={(e) => setTargetLang(e.target.value)} className="input" />
                      </>
                    )}
                  </>
                }
              />
            </>
          )}


          {/* ── Settings (transcribe / index) ── */}
          {canRun.length > 0 && !isLLMStep && !(step === "transcribe" && transcribeSource === "subtitles") && (
            <div className="space-y-4">
              {step === "transcribe" && (() => {
                const isCpu = isCpuTranscribe(transcribePreset, tc.modelSize);
                const entries = Object.keys(whisperModels).length > 0
                  ? Object.entries(whisperModels)
                  : [[tc.modelSize, tc.modelSize] as [string, string]];
                const filtered = isCpu
                  ? entries.filter(([key]) => CPU_MODELS.has(key))
                  : entries.filter(([key]) => GPU_MODELS.has(key));
                return (
                  <>
                    <div className="space-y-1.5">
                      <HelpLabel label="Model" htmlFor={`${fieldId}-2`} help="Speech recognition model. Bigger models make fewer mistakes but are slower and need more GPU memory." />
                      <select id={`${fieldId}-2`} value={tc.modelSize} onChange={(e) => setTc({ modelSize: e.target.value })} className={selectFull}>
                        {filtered.map(([key, label]) => (
                          <option key={key} value={key}>{key}: {isCpu ? CPU_LABELS[key] || label : GPU_LABELS[key] || label}</option>
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
          {manual.active && (
            <Button
              onClick={async () => {
                // Back drops the pasted replies, like closing the dialog.
                if (manual.dirty && !(await confirmDiscard(["pasted LLM batches"]))) return;
                manual.back();
              }}
              variant="ghost"
              size="sm"
            >
              Back
            </Button>
          )}
          {selectedSource === "custom" && !manual.active && (
            <Button onClick={() => setSelectedSource(null)} variant="ghost" size="sm">
              Back
            </Button>
          )}
          {selectedSource !== "custom" && !manual.active && (
            <Button onClick={requestClose} variant="ghost" size="sm">
              {canRun.length === 0 ? "Close" : "Cancel"}
            </Button>
          )}

          {/* Manual mode: "Next" on step 1, "Generate"/"Apply" on step 2 */}
          {filteredEpisodes.length > 0 && isLLMStep && llm.mode === "manual" && !manual.active && selectedSource !== "custom" && (
            <Button onClick={manual.start} size="sm">
              {stepInfo.label} {countLabel(filteredEpisodes.length, "episode")}
            </Button>
          )}
          {filteredEpisodes.length > 0 && isLLMStep && manual.active && (
            manual.allApplied ? (
              <Button onClick={onClose} size="sm">Done</Button>
            ) : manual.pendingApply.length > 0 ? (
              // Replies complete but the apply failed: try those again.
              <Button onClick={() => void manual.retry()} disabled={manual.retrying} size="sm">
                {manual.retrying ? "Applying..." : `Retry ${manual.pendingApply.length} failed`}
              </Button>
            ) : null
          )}

          {/* Custom mode: confirm selection and go back to config */}
          {selectedSource === "custom" && !manual.active && (
            <Button onClick={() => setSelectedSource(null)} size="sm">
              <Check className="w-3.5 h-3.5 mr-1" />
              Confirm versions
            </Button>
          )}

          {/* Non-manual run button */}
          {filteredEpisodes.length > 0 && !(isLLMStep && llm.mode === "manual") && !manual.active && selectedSource !== "custom" && (() => {
            const pending = filteredEpisodes.filter((e) => needsWorkIds.has(epKey(e)));
            const runWith = (force?: boolean) => {
              const vids = toBatchKeys(pinnedVersions, filteredEpisodes);
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
                    {stepInfo.label} {countLabel(pending.length, "episode")}
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
      </DialogContent>
    </Dialog>
  );
}
