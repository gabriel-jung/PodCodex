import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  downloadEpisodes,
  refreshRSS,
  getEpisodes,
  getShowMeta,
  deleteAudioFile,
  startBatch,
  getPipelineConfig,
} from "@/api/client";
import type { Episode } from "@/api/types";
import { useAudioStore, useConfigStore, usePipelineConfigStore, useTaskStore } from "@/stores";
import { Button } from "@/components/ui/button";
import {
  ArrowLeft, RefreshCw, Podcast, Search,
  Mic, Sparkles, Languages, Database,
  SlidersHorizontal, ChevronDown, ChevronUp, Download,
  List, LayoutGrid, ArrowUpDown,
} from "lucide-react";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import { EmptyState } from "@/components/ui/empty-state";
import ShowSettings from "@/components/show/ShowSettings";
import SpeakersPanel from "@/components/show/SpeakersPanel";
import { EpisodeRow } from "@/components/show/EpisodeRow";
import { EpisodeCard } from "@/components/show/EpisodeCard";
import SearchPanel from "@/components/search/SearchPanel";

import { errorMessage, languageToISO } from "@/lib/utils";

/* ── Pipeline step buttons (collapse to dropdown on small screens) ── */

const STEPS = [
  { key: "transcribe", label: "Transcribe", icon: Mic },
  { key: "polish", label: "Polish", icon: Sparkles },
  { key: "translate", label: "Translate", icon: Languages },
  { key: "index", label: "Index", icon: Database },
] as const;

type StepKey = "transcribe" | "polish" | "translate" | "index";

function StepConfigEditor({ step, episodes, showLanguage, onRun, onClose }: { step: StepKey; episodes: Episode[]; showLanguage: string; onRun: () => void; onClose: () => void }) {
  const tc = usePipelineConfigStore((s) => s.transcribe);
  const setTc = usePipelineConfigStore((s) => s.setTranscribe);
  const llm = usePipelineConfigStore((s) => s.llm);
  const setLLM = usePipelineConfigStore((s) => s.setLLM);
  const engine = usePipelineConfigStore((s) => s.engine);
  const setEngine = usePipelineConfigStore((s) => s.setEngine);
  const targetLang = usePipelineConfigStore((s) => s.targetLang);
  const setTargetLang = usePipelineConfigStore((s) => s.setTargetLang);

  const { data: pipelineConfig } = useQuery({
    queryKey: ["pipeline-config"],
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  const whisperModels = pipelineConfig?.whisper_models ?? {};
  const detected = pipelineConfig?.detected_keys ?? {};
  const apiProviders = pipelineConfig
    ? Object.entries(pipelineConfig.llm_providers).filter(([k]) => k !== "ollama")
    : [];

  const selClass = "bg-secondary text-secondary-foreground rounded px-2 py-1.5 border border-border text-sm w-full";
  const inputClass = "input py-1.5 text-sm w-full";

  const stepInfo = STEPS.find((s) => s.key === step)!;
  const Icon = stepInfo.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div className="bg-background border border-border rounded-lg shadow-xl w-full max-w-md mx-4" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center gap-2 px-5 py-4 border-b border-border">
          <Icon className="w-4 h-4" />
          <h3 className="text-base font-semibold">{stepInfo.label}</h3>
          <div className="flex-1" />
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-lg">×</button>
        </div>

        {/* Body */}
        <div className="px-5 py-4 space-y-4 max-h-[60vh] overflow-y-auto">
          {step === "transcribe" && (
            <>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Model</label>
                <select value={tc.modelSize} onChange={(e) => setTc({ modelSize: e.target.value })} className={selClass}>
                  {Object.keys(whisperModels).length > 0
                    ? Object.entries(whisperModels).map(([key, label]) => (
                        <option key={key} value={key}>{key} — {label}</option>
                      ))
                    : <option value={tc.modelSize}>{tc.modelSize}</option>
                  }
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Language</label>
                <div className="flex items-center gap-2 text-sm">
                  <span className="font-mono">{languageToISO(showLanguage) || "auto-detect"}</span>
                  {showLanguage && <span className="text-muted-foreground text-xs">({showLanguage})</span>}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Batch size</label>
                  <input type="number" value={tc.batchSize} onChange={(e) => setTc({ batchSize: Number(e.target.value) })} min={1} className={inputClass} />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Speakers</label>
                  <input type="number" value={tc.numSpeakers} onChange={(e) => setTc({ numSpeakers: e.target.value })} placeholder="auto" min={1} className={inputClass} />
                </div>
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={tc.diarize} onChange={(e) => setTc({ diarize: e.target.checked })} className="accent-primary" />
                <span className="text-sm">Diarize (detect speakers)</span>
              </label>
              {tc.diarize && (
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">HuggingFace token</label>
                  <input type="password" value={tc.hfToken} onChange={(e) => setTc({ hfToken: e.target.value })} placeholder={detected.hf_token || "from env"} className={inputClass} />
                </div>
              )}
            </>
          )}

          {step === "polish" && (
            <>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Engine</label>
                <select value={engine} onChange={(e) => setEngine(e.target.value)} className={selClass}>
                  <option value="Whisper">Whisper</option>
                  <option value="Voxtral">Voxtral</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Source language</label>
                <input value={llm.sourceLang} onChange={(e) => setLLM({ sourceLang: e.target.value })} className={inputClass} />
              </div>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Context</label>
                <textarea value={llm.context} onChange={(e) => setLLM({ context: e.target.value })} placeholder="Describe the podcast, hosts, topics..." className="input py-1.5 text-sm w-full resize-y min-h-[3rem]" />
              </div>
            </>
          )}

          {step === "translate" && (
            <>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Source language</label>
                  <input value={llm.sourceLang} onChange={(e) => setLLM({ sourceLang: e.target.value })} className={inputClass} />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Target language</label>
                  <input value={targetLang} onChange={(e) => setTargetLang(e.target.value)} className={inputClass} />
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Context</label>
                <textarea value={llm.context} onChange={(e) => setLLM({ context: e.target.value })} placeholder="Describe the podcast, hosts, topics..." className="input py-1.5 text-sm w-full resize-y min-h-[3rem]" />
              </div>
            </>
          )}

          {step === "index" && (
            <p className="text-sm text-muted-foreground">No additional configuration needed. Episodes will be indexed using default settings.</p>
          )}

          {/* LLM settings — shared by polish & translate */}
          {(step === "polish" || step === "translate") && (
            <>
              <div className="border-t border-border pt-4 mt-4">
                <h4 className="text-sm font-semibold mb-3">LLM Settings</h4>
              </div>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Mode</label>
                <div className="flex gap-4">
                  {(["ollama", "api"] as const).map((m) => (
                    <label key={m} className="flex items-center gap-1.5 cursor-pointer text-sm">
                      <input type="radio" checked={llm.mode === m} onChange={() => setLLM({ mode: m })} className="accent-primary" />
                      {m}
                    </label>
                  ))}
                </div>
              </div>
              {llm.mode === "api" && (
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Provider</label>
                  <select value={llm.provider} onChange={(e) => setLLM({ provider: e.target.value })} className={selClass}>
                    {apiProviders.length > 0
                      ? apiProviders.map(([key, spec]) => (
                          <option key={key} value={key}>{spec.label}</option>
                        ))
                      : <option value={llm.provider}>{llm.provider}</option>
                    }
                  </select>
                </div>
              )}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Model</label>
                  <input value={llm.model} onChange={(e) => setLLM({ model: e.target.value })} placeholder="default" className={inputClass} />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Batch (min)</label>
                  <input type="number" value={llm.batchMinutes} onChange={(e) => setLLM({ batchMinutes: Number(e.target.value) })} min={1} step={5} className={inputClass} />
                </div>
              </div>
              {llm.mode === "api" && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Endpoint</label>
                    <input value={llm.apiBaseUrl} onChange={(e) => setLLM({ apiBaseUrl: e.target.value })} placeholder="default" className={inputClass} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">API key</label>
                    <input type="password" value={llm.apiKey} onChange={(e) => setLLM({ apiKey: e.target.value })} placeholder={detected[llm.provider] || "from env"} className={inputClass} />
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Episodes summary */}
        <div className="px-5 py-3 border-t border-border">
          <p className="text-xs font-medium text-muted-foreground mb-1.5">{episodes.length} episode{episodes.length !== 1 ? "s" : ""} selected</p>
          <div className="max-h-24 overflow-y-auto space-y-0.5">
            {episodes.map((ep) => (
              <p key={ep.id} className="text-xs text-muted-foreground truncate">{ep.title}</p>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-5 py-4 border-t border-border">
          <Button onClick={onClose} variant="ghost" size="sm">Cancel</Button>
          <Button onClick={onRun} size="sm">{stepInfo.label} {episodes.length} episode{episodes.length !== 1 ? "s" : ""}</Button>
        </div>
      </div>
    </div>
  );
}

function PipelineButtons({
  disabled,
  episodes,
  showLanguage,
  onRun,
}: {
  disabled: boolean;
  episodes: Episode[];
  showLanguage: string;
  onRun: (step: StepKey) => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmStep, setConfirmStep] = useState<StepKey | null>(null);

  const handleClick = (key: StepKey) => {
    setMenuOpen(false);
    setConfirmStep(key);
  };

  const handleConfirm = () => {
    if (confirmStep) onRun(confirmStep);
    setConfirmStep(null);
  };

  return (
    <>
      {/* Wide screens: individual buttons */}
      <div className="hidden md:flex items-center gap-1.5">
        {STEPS.map(({ key, label, icon: Icon }) => (
          <Button
            key={key}
            onClick={() => handleClick(key)}
            disabled={disabled}
            variant="outline"
            size="sm"
            className="text-xs h-7 px-2"
          >
            <Icon className="w-3 h-3 mr-1" /> {label}
          </Button>
        ))}
      </div>

      {/* Small screens: dropdown */}
      <div className="relative md:hidden">
        <Button
          onClick={() => setMenuOpen(!menuOpen)}
          disabled={disabled}
          variant="outline"
          size="sm"
          className="text-xs h-7 px-2"
        >
          Pipeline <ChevronDown className="w-3 h-3 ml-1" />
        </Button>
        {menuOpen && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setMenuOpen(false)} />
            <div className="absolute right-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg py-1 min-w-[140px]">
              {STEPS.map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => handleClick(key)}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-accent transition"
                >
                  <Icon className="w-3 h-3" /> {label}
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Config dialog */}
      {confirmStep && (
        <StepConfigEditor
          step={confirmStep}
          episodes={episodes}
          showLanguage={showLanguage}
          onRun={handleConfirm}
          onClose={() => setConfirmStep(null)}
        />
      )}
    </>
  );
}

/* ── Filter dropdown ── */

function FilterDropdown({
  minDurationMinutes, setMinDurationMinutes,
  maxDurationMinutes, setMaxDurationMinutes,
  titleInclude, setTitleInclude,
  titleExclude, setTitleExclude,
}: {
  minDurationMinutes: number; setMinDurationMinutes: (v: number) => void;
  maxDurationMinutes: number; setMaxDurationMinutes: (v: number) => void;
  titleInclude: string; setTitleInclude: (v: string) => void;
  titleExclude: string; setTitleExclude: (v: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const activeCount = [
    minDurationMinutes > 0,
    maxDurationMinutes > 0,
    titleInclude.length > 0,
    titleExclude.length > 0,
  ].filter(Boolean).length;

  const clearAll = () => {
    setMinDurationMinutes(0);
    setMaxDurationMinutes(0);
    setTitleInclude("");
    setTitleExclude("");
  };

  return (
    <div className="relative">
      <Button
        onClick={() => setOpen(!open)}
        variant={activeCount > 0 ? "secondary" : "ghost"}
        size="sm"
        className="text-xs h-7 px-2 gap-1"
      >
        <SlidersHorizontal className="w-3 h-3" />
        Filters
        {activeCount > 0 && <span className="bg-primary text-primary-foreground rounded-full px-1 text-[10px]">{activeCount}</span>}
      </Button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute left-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg p-3 min-w-[240px] space-y-3">
            <div className="space-y-2">
              <span className="text-xs font-medium">Duration</span>
              <div className="flex items-center gap-2">
                <input
                  type="number" min={0} step={5}
                  value={minDurationMinutes || ""}
                  onChange={(e) => setMinDurationMinutes(Math.max(0, Number(e.target.value)))}
                  placeholder="min"
                  className="input w-16 py-1 text-xs text-center"
                />
                <span className="text-xs text-muted-foreground">to</span>
                <input
                  type="number" min={0} step={5}
                  value={maxDurationMinutes || ""}
                  onChange={(e) => setMaxDurationMinutes(Math.max(0, Number(e.target.value)))}
                  placeholder="max"
                  className="input w-16 py-1 text-xs text-center"
                />
                <span className="text-xs text-muted-foreground">min</span>
              </div>
            </div>
            <div className="space-y-2">
              <span className="text-xs font-medium">Title contains</span>
              <input
                value={titleInclude}
                onChange={(e) => setTitleInclude(e.target.value)}
                placeholder="word or phrase..."
                className="input w-full py-1 text-xs"
              />
            </div>
            <div className="space-y-2">
              <span className="text-xs font-medium">Title excludes</span>
              <input
                value={titleExclude}
                onChange={(e) => setTitleExclude(e.target.value)}
                placeholder="word or phrase..."
                className="input w-full py-1 text-xs"
              />
            </div>
            {activeCount > 0 && (
              <Button onClick={() => { clearAll(); setOpen(false); }} variant="ghost" size="sm" className="text-xs w-full">
                Clear all filters
              </Button>
            )}
          </div>
        </>
      )}
    </div>
  );
}

/* ── Sortable column header ── */

function SortHeader({
  col, label, current, dir, onSort, className = "",
}: {
  col: string;
  label: string;
  current: string;
  dir: "asc" | "desc";
  onSort: (col: any) => void;
  className?: string;
}) {
  const active = current === col;
  return (
    <button
      onClick={() => onSort(col)}
      className={`flex items-center gap-0.5 hover:text-foreground transition group ${active ? "text-foreground" : ""} ${className}`}
    >
      <span>{label}</span>
      {active ? (
        dir === "asc" ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
      ) : (
        <ArrowUpDown className="w-3 h-3 opacity-0 group-hover:opacity-50 transition-opacity" />
      )}
    </button>
  );
}

type ShowTab = "episodes" | "search" | "speakers" | "settings";
type ViewMode = "list" | "card";
type StatusFilter = "all" | "downloaded" | "not_downloaded" | "transcribed" | "not_transcribed" | "polished" | "not_polished" | "translated" | "synthesized" | "indexed";
type SortKey = "date_desc" | "date_asc" | "title_asc" | "title_desc" | "duration_desc" | "duration_asc" | "number_desc" | "number_asc";

export default function ShowPage({ folder }: { folder: string }) {
  const navigate = useNavigate();
  const { seekTo, setAudioMeta, audioPath } = useAudioStore();
  const {
    minDurationMinutes, setMinDurationMinutes,
    maxDurationMinutes, setMaxDurationMinutes,
    titleInclude, setTitleInclude,
    titleExclude, setTitleExclude,
  } = useConfigStore();
  const queryClient = useQueryClient();

  const [tab, setTab] = useState<ShowTab>("episodes");
  const [view, setView] = useState<ViewMode>("list");
  const [cardSize, setCardSize] = useState(3);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [sort, setSort] = useState<SortKey>("date_desc");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const { downloadTaskId, setDownloadTask, batchTaskId, setBatchTask } = useTaskStore();

  // Pipeline config from store (for batch start)
  const tc = usePipelineConfigStore((s) => s.transcribe);
  const llm = usePipelineConfigStore((s) => s.llm);
  const engine = usePipelineConfigStore((s) => s.engine);
  const targetLang = usePipelineConfigStore((s) => s.targetLang);

  const { data: meta } = useQuery({
    queryKey: ["showMeta", folder],
    queryFn: () => getShowMeta(folder),
  });

  const { data: episodes, isLoading: episodesLoading } = useQuery({
    queryKey: ["episodes", folder],
    queryFn: () => getEpisodes(folder),
    refetchInterval: downloadTaskId || batchTaskId ? 5000 : false,
  });

  const refreshMutation = useMutation({
    mutationFn: () => refreshRSS(folder),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["episodes", folder] }),
  });

  const downloadMutation = useMutation({
    mutationFn: (guids: string[]) => downloadEpisodes(folder, guids),
    onSuccess: (data) => { setDownloadTask(data.task_id, folder); },
  });

  const deleteMutation = useMutation({
    mutationFn: (audioPath: string) => deleteAudioFile(audioPath),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["episodes", folder] }),
  });

  const batchMutationEpisodesRef = useRef({ names: [] as string[], step: "" });
  const batchMutation = useMutation({
    mutationFn: (args: Parameters<typeof startBatch>[0]) => {
      batchMutationEpisodesRef.current.names = batchableSelected.map((e) => e.title);
      return startBatch(args);
    },
    onSuccess: (data) => {
      setBatchTask(data.task_id, folder, batchMutationEpisodesRef.current.names, batchMutationEpisodesRef.current.step);
    },
  });

  const all = episodes ?? [];

  const filtered = useMemo(() => {
    let list = all;
    if (minDurationMinutes > 0) {
      const minSec = minDurationMinutes * 60;
      list = list.filter((e) => e.duration >= minSec);
    }
    if (maxDurationMinutes > 0) {
      const maxSec = maxDurationMinutes * 60;
      list = list.filter((e) => e.duration <= maxSec);
    }
    if (titleInclude) {
      const q = titleInclude.toLowerCase();
      list = list.filter((e) => e.title.toLowerCase().includes(q));
    }
    if (titleExclude) {
      const q = titleExclude.toLowerCase();
      list = list.filter((e) => !e.title.toLowerCase().includes(q));
    }
    if (search) {
      const q = search.toLowerCase();
      list = list.filter((e) => e.title.toLowerCase().includes(q));
    }
    if (filter === "downloaded") list = list.filter((e) => e.downloaded);
    if (filter === "not_downloaded") list = list.filter((e) => !e.downloaded);
    if (filter === "transcribed") list = list.filter((e) => e.transcribed);
    if (filter === "not_transcribed") list = list.filter((e) => e.downloaded && !e.transcribed);
    if (filter === "polished") list = list.filter((e) => e.polished);
    if (filter === "not_polished") list = list.filter((e) => e.transcribed && !e.polished);
    if (filter === "translated") list = list.filter((e) => e.translations.length > 0);
    if (filter === "synthesized") list = list.filter((e) => e.synthesized);
    if (filter === "indexed") list = list.filter((e) => e.indexed);
    // Sort
    list = [...list].sort((a, b) => {
      switch (sort) {
        case "date_asc": return new Date(a.pub_date ?? 0).getTime() - new Date(b.pub_date ?? 0).getTime();
        case "date_desc": return new Date(b.pub_date ?? 0).getTime() - new Date(a.pub_date ?? 0).getTime();
        case "title_asc": return a.title.localeCompare(b.title);
        case "title_desc": return b.title.localeCompare(a.title);
        case "duration_asc": return a.duration - b.duration;
        case "duration_desc": return b.duration - a.duration;
        case "number_asc": return (a.episode_number ?? 0) - (b.episode_number ?? 0);
        case "number_desc": return (b.episode_number ?? 0) - (a.episode_number ?? 0);
        default: return 0;
      }
    });
    return list;
  }, [all, search, filter, sort, minDurationMinutes, maxDurationMinutes, titleInclude, titleExclude]);

  const showName = meta?.name || folder.replace(/\/+$/, "").split("/").pop() || "Show";

  const downloadableSelected = filtered.filter((e) => selected.has(e.id) && !e.downloaded && e.audio_url);
  const batchableSelected = filtered.filter((e) => selected.has(e.id) && e.downloaded);

  const selectableEpisodes = filtered.filter((e) => e.downloaded || (!e.downloaded && e.audio_url));
  const allSelectableSelected = selectableEpisodes.length > 0 && selectableEpisodes.every((e) => selected.has(e.id));

  const toggleSelect = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });

  const toggleSelectAll = () => {
    setSelected(allSelectableSelected ? new Set() : new Set(selectableEpisodes.map((e) => e.id)));
  };

  const toggleSort = (col: "date" | "title" | "duration" | "number") => {
    const pairs: Record<string, [SortKey, SortKey]> = {
      date: ["date_desc", "date_asc"],
      title: ["title_asc", "title_desc"],
      duration: ["duration_desc", "duration_asc"],
      number: ["number_desc", "number_asc"],
    };
    const [primary, alt] = pairs[col];
    setSort(sort === primary ? alt : primary);
  };

  const sortCol = sort.replace(/_(?:asc|desc)$/, "");
  const sortDir = sort.endsWith("_asc") ? "asc" : "desc";

  const goEpisode = (stem: string) =>
    navigate({ to: "/show/$folder/episode/$stem", params: { folder: encodeURIComponent(folder), stem: encodeURIComponent(stem) } });

  const runStep = (step: "transcribe" | "polish" | "translate" | "index") => {
    const audioPaths = batchableSelected.map((e) => e.audio_path).filter(Boolean) as string[];
    if (audioPaths.length === 0) return;
    batchMutationEpisodesRef.current.step = step;
    batchMutation.mutate({
      show_folder: folder,
      audio_paths: audioPaths,
      transcribe: step === "transcribe",
      polish: step === "polish",
      translate: step === "translate",
      index: step === "index",
      model_size: tc.modelSize,
      language: languageToISO(meta?.language || ""),
      batch_size: tc.batchSize,
      diarize: tc.diarize,
      hf_token: tc.hfToken || undefined,
      num_speakers: tc.numSpeakers ? Number(tc.numSpeakers) : undefined,
      llm_mode: llm.mode === "api" ? "api" : "ollama",
      llm_provider: llm.mode === "api" ? llm.provider : undefined,
      llm_model: llm.model || undefined,
      llm_api_base_url: llm.apiBaseUrl || undefined,
      llm_api_key: llm.apiKey || undefined,
      context: llm.context,
      source_lang: llm.sourceLang,
      target_lang: targetLang,
      llm_batch_minutes: llm.batchMinutes,
      engine,
      show_name: meta?.name || "",
    });
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Top bar */}
      <div className="px-6 py-4 border-b border-border flex items-center gap-4">
        <Button onClick={() => navigate({ to: "/" })} variant="ghost" size="sm">
          <ArrowLeft /> Shows
        </Button>
        {meta?.artwork_url && (
          <img src={meta.artwork_url} alt="" className="w-10 h-10 rounded-lg shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <h2 className="text-lg font-semibold truncate">{showName}</h2>
          <div className="flex gap-3 text-xs text-muted-foreground mt-0.5">
            {all.length > 0 && <span>{all.length} episode{all.length !== 1 && "s"}</span>}
            {meta?.language && <span>{meta.language}</span>}
            {meta?.speakers && meta.speakers.length > 0 && (
              <span>{meta.speakers.length} speaker{meta.speakers.length !== 1 && "s"}</span>
            )}
          </div>
        </div>
        <Button
          onClick={() => refreshMutation.mutate()}
          disabled={refreshMutation.isPending}
          variant="outline"
          size="sm"
        >
          <RefreshCw className={refreshMutation.isPending ? "animate-spin" : ""} />
          {refreshMutation.isPending ? "Refreshing..." : "Refresh RSS"}
        </Button>
      </div>

      {/* Tabs + view toggle */}
      <div className="px-6 border-b border-border flex items-center">
        <div className="flex gap-1">
          {(["episodes", "search", "speakers", "settings"] as ShowTab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-2 text-sm capitalize transition border-b-2 -mb-px ${
                tab === t
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        {tab === "episodes" && (
          <>
            <div className="flex-1" />
            <div className="flex items-center gap-2">
              {view === "card" && (
                <input type="range" min={1} max={5} value={cardSize} onChange={(e) => setCardSize(Number(e.target.value))} className="w-16 accent-primary" />
              )}
              <div className="flex border border-border rounded overflow-hidden">
                <button onClick={() => setView("list")} className={`px-1.5 py-1 transition ${view === "list" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`} title="List view">
                  <List className="w-3.5 h-3.5" />
                </button>
                <button onClick={() => setView("card")} className={`px-1.5 py-1 transition ${view === "card" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`} title="Card view">
                  <LayoutGrid className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {tab === "episodes" && (<>
      {/* Toolbar: select-all + filters | search */}
      <div className="px-6 py-2 border-b border-border flex items-center gap-2">
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value as StatusFilter)}
          className="bg-secondary text-secondary-foreground text-xs rounded px-2 py-1.5 border border-border"
        >
          <option value="all">All ({all.length})</option>
          <option value="downloaded">Downloaded ({all.filter((e) => e.downloaded).length})</option>
          <option value="not_downloaded">Not downloaded ({all.filter((e) => !e.downloaded).length})</option>
          <option value="transcribed">Transcribed ({all.filter((e) => e.transcribed).length})</option>
          <option value="not_transcribed">Not transcribed ({all.filter((e) => e.downloaded && !e.transcribed).length})</option>
          <option value="polished">Polished ({all.filter((e) => e.polished).length})</option>
          <option value="not_polished">Not polished ({all.filter((e) => e.transcribed && !e.polished).length})</option>
          <option value="translated">Translated ({all.filter((e) => e.translations.length > 0).length})</option>
          <option value="synthesized">Synthesized ({all.filter((e) => e.synthesized).length})</option>
          <option value="indexed">Indexed ({all.filter((e) => e.indexed).length})</option>
        </select>
        <FilterDropdown
          minDurationMinutes={minDurationMinutes} setMinDurationMinutes={setMinDurationMinutes}
          maxDurationMinutes={maxDurationMinutes} setMaxDurationMinutes={setMaxDurationMinutes}
          titleInclude={titleInclude} setTitleInclude={setTitleInclude}
          titleExclude={titleExclude} setTitleExclude={setTitleExclude}
        />
        <div className="flex-1" />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search..."
          className="input w-40 py-1.5 text-xs"
        />
      </div>

      {/* Selection info + pipeline actions (always visible) */}
      <div className="relative px-6 py-2 border-b border-border flex items-center gap-3 text-xs">
        <input
          type="checkbox"
          checked={allSelectableSelected}
          onChange={toggleSelectAll}
          className="accent-primary cursor-pointer"
          title={allSelectableSelected ? "Unselect all" : "Select all"}
        />
        {selected.size > 0 ? (
          <>
            <span className="text-muted-foreground">{selected.size} selected</span>
            <Button onClick={() => setSelected(new Set())} variant="ghost" size="sm" className="text-xs h-6 px-1.5">Clear</Button>
          </>
        ) : (
          <span className="text-muted-foreground">Select episodes to run pipeline</span>
        )}
        <div className="flex-1" />
        <Button
          onClick={() => downloadMutation.mutate(downloadableSelected.map((e) => e.id))}
          disabled={downloadableSelected.length === 0 || downloadMutation.isPending || !!downloadTaskId}
          variant="outline"
          size="sm"
          className="text-xs h-7 px-2"
        >
          <Download className="w-3 h-3 mr-1" /> Download{downloadableSelected.length > 0 ? ` ${downloadableSelected.length}` : ""}
        </Button>
        <PipelineButtons
          disabled={batchableSelected.length === 0 || !!batchTaskId || batchMutation.isPending}
          episodes={batchableSelected}
          showLanguage={meta?.language || ""}
          onRun={runStep}
        />
        {batchMutation.isError && (
          <span className="text-destructive">{errorMessage(batchMutation.error)}</span>
        )}
      </div>

      {/* Episode list */}
      <div className="flex-1 overflow-y-auto">
        {view === "list" ? (
          <div className="divide-y divide-border/50">
            {/* Column headers */}
            <div className="flex items-center gap-3 px-6 py-1.5 text-[10px] uppercase tracking-wider text-muted-foreground select-none border-b border-border">
              <div className="w-10 shrink-0" />
              <SortHeader col="number" label="#" current={sortCol} dir={sortDir} onSort={toggleSort} className="w-8 text-right shrink-0" />
              <SortHeader col="title" label="Title" current={sortCol} dir={sortDir} onSort={toggleSort} className="flex-1 min-w-0" />
              <div className="w-16 shrink-0" />
              <SortHeader col="date" label="Date" current={sortCol} dir={sortDir} onSort={toggleSort} className="w-20 text-right shrink-0" />
              <SortHeader col="duration" label="Duration" current={sortCol} dir={sortDir} onSort={toggleSort} className="w-12 text-right shrink-0" />
              <div className="w-20 shrink-0" />
            </div>
            {filtered.map((ep) => (
              <EpisodeRow
                key={ep.id}
                ep={ep}
                selected={selected.has(ep.id)}
                onToggle={() => toggleSelect(ep.id)}
                onOpen={() => goEpisode(ep.stem || ep.id)}
                onPlay={() => ep.audio_path && (setAudioMeta(ep.audio_path, { title: ep.title, artwork: ep.artwork_url || meta?.artwork_url, showName }), seekTo(ep.audio_path, 0))}
                onDownload={() => downloadMutation.mutate([ep.id])}
                onDelete={() => ep.audio_path && confirmDialog.open({
                  title: "Delete episode audio?",
                  description: `This will remove the downloaded audio for "${ep.title}". You can re-download it later from RSS.`,
                  confirmLabel: "Delete",
                  variant: "destructive",
                  onConfirm: () => deleteMutation.mutate(ep.audio_path!),
                })}
                downloading={downloadMutation.isPending || !!downloadTaskId}
                isPlaying={!!ep.audio_path && ep.audio_path === audioPath}
              />
            ))}
          </div>
        ) : (
          <div
            className="p-6 grid gap-4"
            style={{ gridTemplateColumns: `repeat(${cardSize}, minmax(0, 1fr))` }}
          >
            {filtered.map((ep) => (
              <EpisodeCard
                key={ep.id}
                ep={ep}
                onOpen={() => goEpisode(ep.stem || ep.id)}
                onPlay={() => ep.audio_path && (setAudioMeta(ep.audio_path, { title: ep.title, artwork: ep.artwork_url || meta?.artwork_url, showName }), seekTo(ep.audio_path, 0))}
                onDownload={() => downloadMutation.mutate([ep.id])}
                downloading={downloadMutation.isPending || !!downloadTaskId}
                isPlaying={!!ep.audio_path && ep.audio_path === audioPath}
              />
            ))}
          </div>
        )}

        {episodesLoading && (
          <EmptyState icon={Podcast} title="Loading episodes..." />
        )}
        {!episodesLoading && filtered.length === 0 && (
          all.length === 0 ? (
            <EmptyState
              icon={Podcast}
              title="No episodes yet"
              description="Refresh the RSS feed to fetch episodes, or add audio files manually."
              action={{ label: "Refresh RSS", onClick: () => refreshMutation.mutate() }}
            />
          ) : (
            <EmptyState
              icon={Search}
              title="No episodes match your filters"
              description="Try changing the search term or status filter."
            />
          )
        )}
      </div>

      {refreshMutation.isError && (
        <div className="px-6 py-2 border-t border-border text-destructive text-xs">
          {errorMessage(refreshMutation.error)}
        </div>
      )}
      </>)}

      {tab === "search" && (
        <SearchPanel scope="show" folder={folder} showName={showName} />
      )}

      {tab === "speakers" && meta && (
        <SpeakersPanel folder={folder} meta={meta} />
      )}

      {tab === "settings" && meta && (
        <ShowSettings
          folder={folder}
          meta={meta}
          hasIndex={all.some((e) => e.indexed)}
        />
      )}
    </div>
  );
}
