import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useMemo, useRef, useState } from "react";
import {
  refreshRSS,
  refreshYouTube,
  getEpisodes,
  getShowMeta,
  deleteAudioFile,
  startBatch,
} from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { artworkUrl } from "@/api/filesystem";
import type { Episode } from "@/api/types";
import { languageToISO, isOutdated } from "@/lib/utils";
import { useAudioStore, useEpisodeStore, useTaskStore } from "@/stores";
import { usePipelineConfig, usePipelineDefaults } from "@/hooks/usePipelineConfig";
import { useShowActions } from "@/hooks/useShowActions";

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
import FilterDropdown from "@/components/show/FilterDropdown";
import SortHeader from "@/components/show/SortHeader";
import { errorMessage } from "@/lib/utils";
import DownloadDropdown from "@/components/common/DownloadDropdown";

type ShowTab = "episodes" | "search" | "speakers" | "settings";
const TABS: ShowTab[] = ["episodes", "search", "speakers", "settings"];
type ViewMode = "list" | "card";
type StatusFilter = "all" | "ready" | "transcribed" | "polished" | "translated" | "indexed" | "outdated";
type SortKey = "date_desc" | "date_asc" | "title_asc" | "title_desc" | "duration_desc" | "duration_asc" | "number_desc" | "number_asc";

export default function ShowPage({ folder, initialTab }: { folder: string; initialTab?: string }) {
  const navigate = useNavigate();
  const { seekTo, setAudioMeta, audioPath } = useAudioStore();
  const { minDurationMinutes, maxDurationMinutes, titleInclude, titleExclude } = useEpisodeStore();
  const queryClient = useQueryClient();


  const [tab, setTab] = useState<ShowTab>(
    TABS.includes(initialTab as ShowTab) ? (initialTab as ShowTab) : "episodes",
  );
  const [view, setView] = useState<ViewMode>("list");
  const [cardSize, setCardSize] = useState(3);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [sort, setSort] = useState<SortKey>("date_desc");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const lastShiftClickIndex = useRef<number | null>(null);
  const { downloadTaskId, batchTaskId, setBatchTask } = useTaskStore();

  // Pipeline config from store (for batch start)
  const { tc, llm, engine, targetLang } = usePipelineConfig();

  const pipelineDefaults = usePipelineDefaults();

  const { data: meta } = useQuery({
    queryKey: queryKeys.showMeta(folder),
    queryFn: () => getShowMeta(folder),
  });

  const { data: episodes, isLoading: episodesLoading } = useQuery({
    queryKey: queryKeys.episodes(folder, pipelineDefaults),
    queryFn: () => getEpisodes(folder, pipelineDefaults),
    refetchInterval: downloadTaskId || batchTaskId ? 5000 : false,
  });

  const { downloadMutation, importSubsMutation, isYouTube } = useShowActions(folder, meta);

  const refreshMutation = useMutation({
    mutationFn: () => (isYouTube ? refreshYouTube(folder) : refreshRSS(folder)),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesForFolder(folder) });
      queryClient.invalidateQueries({ queryKey: queryKeys.showMeta(folder) });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (audioPath: string) => deleteAudioFile(audioPath),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.episodesForFolder(folder) }),
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

  const filterCounts = useMemo(() => ({
    all: all.length,
    ready: all.filter((e) => e.downloaded || (e.transcribed && e.output_dir)).length,
    transcribed: all.filter((e) => e.transcribed).length,
    polished: all.filter((e) => e.polished).length,
    translated: all.filter((e) => e.translations.length > 0).length,
    indexed: all.filter((e) => e.indexed).length,
    outdated: all.filter((e) => isOutdated(e)).length,
  }), [all]);

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
    if (filter === "ready") list = list.filter((e) => e.downloaded || (e.transcribed && e.output_dir));
    if (filter === "transcribed") list = list.filter((e) => e.transcribed);
    if (filter === "polished") list = list.filter((e) => e.polished);
    if (filter === "translated") list = list.filter((e) => e.translations.length > 0);
    if (filter === "indexed") list = list.filter((e) => e.indexed);
    if (filter === "outdated") list = list.filter((e) => isOutdated(e));
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

  const downloadableSelected = filtered.filter((e) => selected.has(e.id) && !e.downloaded);
  const subtitleableSelected = filtered.filter((e) => selected.has(e.id));
  const subsNewCount = subtitleableSelected.filter((e) => !e.transcribed).length;
  const subsExistingCount = subtitleableSelected.length - subsNewCount;
  const batchableSelected = filtered.filter((e) =>
    selected.has(e.id) && (e.downloaded || (e.transcribed && e.output_dir)),
  );

  /** Get a batch-usable path: real audio_path, or synthetic path from output_dir. */
  const batchPath = (e: Episode): string | null =>
    e.audio_path ?? (e.output_dir ? e.output_dir.replace(/\/+$/, "") + ".virtual" : null);

  const selectableEpisodes = filtered;
  const allSelectableSelected = selectableEpisodes.length > 0 && selectableEpisodes.every((e) => selected.has(e.id));

  const toggleSelect = (id: string, idx: number, shiftKey: boolean) => {
    const lastIdx = lastShiftClickIndex.current;
    lastShiftClickIndex.current = idx;
    setSelected((prev) => {
      const next = new Set(prev);
      if (shiftKey && lastIdx != null) {
        const from = Math.min(lastIdx, idx);
        const to = Math.max(lastIdx, idx);
        for (let i = from; i <= to; i++) next.add(filtered[i].id);
      } else {
        if (next.has(id)) next.delete(id); else next.add(id);
      }
      return next;
    });
  };

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
    const audioPaths = batchableSelected.map(batchPath).filter(Boolean) as string[];
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
      clean: tc.clean,
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
          <img src={artworkUrl(folder)} alt={showName} className="w-10 h-10 rounded-lg shrink-0" />
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
          {refreshMutation.isPending ? "Refreshing..." : isYouTube ? "Refresh YouTube" : "Refresh RSS"}
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
          <option value="all">All ({filterCounts.all})</option>
          <option value="ready">Ready ({filterCounts.ready})</option>
          <option value="transcribed">Transcribed ({filterCounts.transcribed})</option>
          <option value="polished">Polished ({filterCounts.polished})</option>
          <option value="translated">Translated ({filterCounts.translated})</option>
          <option value="indexed">Indexed ({filterCounts.indexed})</option>
          <option value="outdated">Outdated ({filterCounts.outdated})</option>
        </select>
        <FilterDropdown />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search episodes..."
          className="input w-40 py-1.5 text-xs"
        />
      </div>

      {/* Selection + actions toolbar */}
      <div className="relative px-6 py-2 border-b border-border flex items-center gap-2 text-xs">
        <input
          type="checkbox"
          checked={allSelectableSelected}
          onChange={toggleSelectAll}
          className="accent-primary cursor-pointer"
          title={allSelectableSelected ? "Unselect all" : "Select all"}
        />
        {selected.size > 0 && (
          <>
            <span className="text-muted-foreground">
              {selected.size} selected
              {batchableSelected.length < selected.size && (
                <span title="Some selected episodes have no audio or subtitles and cannot be processed">
                  {" "}({batchableSelected.length} ready)
                </span>
              )}
            </span>
            <Button onClick={() => setSelected(new Set())} variant="ghost" size="sm" className="text-xs h-6 px-1.5">Clear</Button>
          </>
        )}
        <div className="flex-1" />
        <DownloadDropdown
          isYouTube={isYouTube}
          showLanguage={meta?.language || ""}
          onDownload={() => {
            const allSelected = filtered.filter((e) => selected.has(e.id));
            if (downloadableSelected.length > 0) {
              downloadMutation.mutate({ guids: downloadableSelected.map((e) => e.id) });
            } else if (allSelected.length > 0) {
              const alreadyCount = allSelected.length;
              confirmDialog.open({
                title: "Re-download?",
                description: `${alreadyCount} selected episode${alreadyCount !== 1 ? "s are" : " is"} already downloaded. This will re-download and overwrite the existing files.`,
                confirmLabel: "Re-download",
                onConfirm: () => downloadMutation.mutate({ guids: allSelected.map((e) => e.id), force: true }),
              });
            }
          }}
          onImportSubs={(lang) => {
            const newOnly = subtitleableSelected.filter((e) => !e.transcribed);
            if (subsNewCount > 0) {
              importSubsMutation.mutate({ ids: newOnly.map((e) => e.id), lang });
            } else {
              confirmDialog.open({
                title: "Re-import subtitles?",
                description: `All ${subtitleableSelected.length} selected episode${subtitleableSelected.length !== 1 ? "s" : ""} already ha${subtitleableSelected.length === 1 ? "s" : "ve"} a transcript. This will create a new version.`,
                confirmLabel: "Re-import",
                onConfirm: () => importSubsMutation.mutate({ ids: subtitleableSelected.map((e) => e.id), lang }),
              });
            }
          }}
          subsLabel={subsNewCount > 0 ? `Subtitles (${subsNewCount} new)` : `Re-import subs (${subtitleableSelected.length})`}
          subsEnabled={subtitleableSelected.length > 0}
          audioLabel={`Audio${downloadableSelected.length > 0 ? ` (${downloadableSelected.length})` : ""}`}
          showAudio={true}
          audioEnabled={downloadableSelected.length > 0 || selected.size > 0}
          disabled={!!downloadTaskId || downloadMutation.isPending || importSubsMutation.isPending}
          className="text-xs h-7 px-2"
        />
        <PipelineButtons
          disabled={batchableSelected.length === 0 || !!batchTaskId || batchMutation.isPending}
          episodes={batchableSelected}
          showLanguage={meta?.language || ""}
          onRun={runStep}

        />
        {batchMutation.isError && (
          <span className="text-destructive text-xs">{errorMessage(batchMutation.error)}</span>
        )}
        {downloadMutation.isError && (
          <span className="text-destructive text-xs">{errorMessage(downloadMutation.error)}</span>
        )}
        {importSubsMutation.isError && (
          <span className="text-destructive text-xs">{errorMessage(importSubsMutation.error)}</span>
        )}
      </div>

      {/* Episode list */}
      <div className="flex-1 overflow-y-auto">
        {view === "list" ? (
          <div className="divide-y divide-border/50">
            {/* Column headers */}
            <div className="flex items-center gap-3 px-6 py-1.5 text-2xs uppercase tracking-wider text-muted-foreground select-none border-b border-border">
              <div className="w-10 shrink-0" />
              <SortHeader col="number" label="#" current={sortCol} dir={sortDir} onSort={toggleSort} className="w-8 text-right shrink-0" />
              <SortHeader col="title" label="Title" current={sortCol} dir={sortDir} onSort={toggleSort} className="flex-1 min-w-0" />
              <div className="w-16 shrink-0" />
              <SortHeader col="date" label="Date" current={sortCol} dir={sortDir} onSort={toggleSort} className="w-20 text-right shrink-0" />
              <SortHeader col="duration" label="Duration" current={sortCol} dir={sortDir} onSort={toggleSort} className="w-12 text-right shrink-0" />
              <div className="w-20 shrink-0" />
            </div>
            {filtered.map((ep, i) => (
              <EpisodeRow
                key={ep.id}
                ep={ep}
                selected={selected.has(ep.id)}
                onToggle={(shiftKey) => toggleSelect(ep.id, i, shiftKey)}
                onOpen={() => goEpisode(ep.stem || ep.id)}
                onPlay={() => ep.audio_path && (setAudioMeta(ep.audio_path, { title: ep.title, artwork: ep.artwork_url || meta?.artwork_url, showName }), seekTo(ep.audio_path, 0))}
                onDownload={() => downloadMutation.mutate({ guids: [ep.id] })}
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
                onDownload={() => downloadMutation.mutate({ guids: [ep.id] })}
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
              description={isYouTube ? "Refresh to fetch videos from YouTube." : "Refresh the RSS feed to fetch episodes, or add audio files manually."}
              action={{ label: isYouTube ? "Refresh YouTube" : "Refresh RSS", onClick: () => refreshMutation.mutate() }}
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
