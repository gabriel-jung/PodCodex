import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useMemo, useRef, useState } from "react";
import {
  downloadEpisodes,
  refreshRSS,
  getEpisodes,
  getShowMeta,
  deleteAudioFile,
  startBatch,
} from "@/api/client";
import type { Episode } from "@/api/types";
import { useAudioStore, useConfigStore, useTaskStore } from "@/stores";
import { usePipelineConfig } from "@/hooks/usePipelineConfig";
import { PIPELINE_PRESETS, usePipelineConfigStore } from "@/stores/pipelineConfigStore";
import { Button } from "@/components/ui/button";
import {
  ArrowLeft, RefreshCw, Podcast, Search,
  Download, List, LayoutGrid,
} from "lucide-react";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import { EmptyState } from "@/components/ui/empty-state";
import ShowSettings from "@/components/show/ShowSettings";
import SpeakersPanel from "@/components/show/SpeakersPanel";
import { EpisodeRow } from "@/components/show/EpisodeRow";
import { EpisodeCard } from "@/components/show/EpisodeCard";
import SearchPanel from "@/components/search/SearchPanel";
import PipelineButtons from "@/components/show/PipelineButtons";
import ProcessDialog from "@/components/show/ProcessDialog";
import FilterDropdown from "@/components/show/FilterDropdown";
import SortHeader from "@/components/show/SortHeader";
import { errorMessage, languageToISO } from "@/lib/utils";

type ShowTab = "episodes" | "search" | "speakers" | "settings";
const TABS: ShowTab[] = ["episodes", "search", "speakers", "settings"];
type ViewMode = "list" | "card";
type StatusFilter = "all" | "downloaded" | "not_downloaded" | "transcribed" | "not_transcribed" | "polished" | "not_polished" | "translated" | "synthesized" | "indexed" | "outdated";
type SortKey = "date_desc" | "date_asc" | "title_asc" | "title_desc" | "duration_desc" | "duration_asc" | "number_desc" | "number_asc";

export default function ShowPage({ folder, initialTab }: { folder: string; initialTab?: string }) {
  const navigate = useNavigate();
  const { seekTo, setAudioMeta, audioPath } = useAudioStore();
  const {
    minDurationMinutes, setMinDurationMinutes,
    maxDurationMinutes, setMaxDurationMinutes,
    titleInclude, setTitleInclude,
    titleExclude, setTitleExclude,
  } = useConfigStore();
  const queryClient = useQueryClient();

  const [processDialogOpen, setProcessDialogOpen] = useState(false);
  const [tab, setTab] = useState<ShowTab>(
    TABS.includes(initialTab as ShowTab) ? (initialTab as ShowTab) : "episodes",
  );
  const [view, setView] = useState<ViewMode>("list");
  const [cardSize, setCardSize] = useState(3);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [sort, setSort] = useState<SortKey>("date_desc");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const { downloadTaskId, setDownloadTask, batchTaskId, setBatchTask } = useTaskStore();

  // Pipeline config from store (for batch start)
  const { tc, llm, engine, targetLang } = usePipelineConfig();

  // Build app-level defaults for step status comparison
  const pipelineDefaults = useMemo(() => ({
    model_size: tc.modelSize,
    diarize: tc.diarize,
    llm_mode: llm.mode === "api" ? "api" : "ollama",
    llm_provider: llm.mode === "api" ? llm.provider : "",
    llm_model: llm.model,
    target_lang: targetLang,
  }), [tc.modelSize, tc.diarize, llm.mode, llm.provider, llm.model, targetLang]);

  const { data: meta } = useQuery({
    queryKey: ["showMeta", folder],
    queryFn: () => getShowMeta(folder),
  });

  const { data: episodes, isLoading: episodesLoading } = useQuery({
    queryKey: ["episodes", folder, pipelineDefaults],
    queryFn: () => getEpisodes(folder, pipelineDefaults),
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
    if (filter === "outdated") list = list.filter((e) => e.transcribe_status === "outdated" || e.polish_status === "outdated" || e.translate_status === "outdated");
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

  const runQuickProcess = (opts: {
    preset: string;
    transcribe: boolean;
    polish: boolean;
    translate: boolean;
    index: boolean;
  }) => {
    const audioPaths = batchableSelected.map((e) => e.audio_path).filter(Boolean) as string[];
    if (audioPaths.length === 0) return;
    const p = PIPELINE_PRESETS[opts.preset] || PIPELINE_PRESETS.medium;
    const steps = [opts.transcribe && "transcribe", opts.polish && "polish", opts.translate && "translate", opts.index && "index"].filter(Boolean).join("+");
    batchMutationEpisodesRef.current.step = steps;
    batchMutation.mutate({
      show_folder: folder,
      audio_paths: audioPaths,
      transcribe: opts.transcribe,
      polish: opts.polish,
      translate: opts.translate,
      index: opts.index,
      model_size: p.whisperModel,
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
      index_model_keys: [p.embedModel],
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
          {TABS.map((t) => (
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
          <option value="outdated">Outdated ({all.filter((e) => e.transcribe_status === "outdated" || e.polish_status === "outdated" || e.translate_status === "outdated").length})</option>
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

      {/* Selection info + pipeline actions */}
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
        <Button
          onClick={() => setProcessDialogOpen(true)}
          disabled={batchableSelected.length === 0 || !!batchTaskId || batchMutation.isPending}
          variant="default"
          size="sm"
          className="text-xs h-7 px-3"
        >
          Quick Process{batchableSelected.length > 0 ? ` ${batchableSelected.length}` : ""}
        </Button>
        <ProcessDialog
          open={processDialogOpen}
          onOpenChange={setProcessDialogOpen}
          onRun={runQuickProcess}
          disabled={!!batchTaskId || batchMutation.isPending}
          episodeCount={batchableSelected.length}
        />
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
        />
      )}
    </div>
  );
}
