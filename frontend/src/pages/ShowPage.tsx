import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useMemo, useRef, useState } from "react";
import {
  getEpisodes,
  getShowMeta,
  getSpeakerRoster,
  deleteAudioFile,
  startBatch,
} from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { removeQueriesUnderPath } from "@/api/cacheInvalidation";
import { artworkUrl } from "@/api/filesystem";
import type { Episode } from "@/api/types";
import { languageToISO, isOutdated, splitPath } from "@/lib/utils";
import { dateCmp } from "@/lib/episodeSort";
import type { PipelineInputStep } from "@/lib/pipelineInputs";
import { FeedRefreshButton } from "@/components/common/FeedRefreshButton";
import { useFeedRefresh, useFeedRefreshing } from "@/hooks/useFeedRefresh";
import { useAudioStore, useEpisodeStore, useTaskStore, usePipelineConfigStore, useLayoutStore, useSeedPipelineFromShow } from "@/stores";
import { usePipelineConfig, usePipelineDefaults } from "@/hooks/usePipelineConfig";
import { useShowActions } from "@/hooks/useShowActions";
import { episodeCardGridTemplate } from "@/lib/cardGrid";

import AppSidebar, { type SidebarSection } from "@/components/layout/AppSidebar";
import EditorialHeader from "@/components/layout/EditorialHeader";
import { Button } from "@/components/ui/button";
import {
  Podcast, Search, Users, SlidersHorizontal,
  List, LayoutGrid,
} from "lucide-react";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorAlert } from "@/components/ui/error-alert";
import ShowSettings from "@/components/show/ShowSettings";
import SpeakersPanel from "@/components/show/SpeakersPanel";
import { EpisodeRow } from "@/components/show/EpisodeRow";
import { EpisodeCard } from "@/components/show/EpisodeCard";
import CompactToggle from "@/components/show/CompactToggle";
import SearchPanel from "@/components/search/SearchPanel";
import PipelineButtons from "@/components/show/PipelineButtons";
import FilterDropdown from "@/components/show/FilterDropdown";
import SortDropdown, { type SortKey } from "@/components/show/SortDropdown";
import DownloadDropdown from "@/components/common/DownloadDropdown";

type ShowTab = "episodes" | "search" | "speakers" | "settings";
const TABS: ShowTab[] = ["episodes", "search", "speakers", "settings"];
type ViewMode = "list" | "card";
type StatusFilter = "all" | "ready" | "transcribed" | "corrected" | "translated" | "indexed" | "outdated" | "verified";

const SIDEBAR_SECTIONS: SidebarSection[] = [
  {
    items: [
      { key: "episodes", label: "Episodes", icon: Podcast },
      { key: "search", label: "Search", icon: Search },
      { key: "speakers", label: "Speakers", icon: Users },
      { key: "settings", label: "Show settings", icon: SlidersHorizontal },
    ],
  },
];

// `.virtual` suffix signals to the batch API that the episode has no audio on
// disk but does have an output_dir to resume from (subtitle-only imports).
function batchPath(e: Episode): string | null {
  return e.audio_path ?? (e.output_dir ? e.output_dir.replace(/\/+$/, "") + ".virtual" : null);
}

export default function ShowPage({ folder, initialTab }: { folder: string; initialTab?: string }) {
  const navigate = useNavigate();
  const audioPath = useAudioStore((s) => s.audioPath);
  const audioIsPlaying = useAudioStore((s) => s.isPlaying);
  const minDurationMinutes = useEpisodeStore((s) => s.minDurationMinutes);
  const maxDurationMinutes = useEpisodeStore((s) => s.maxDurationMinutes);
  const titleInclude = useEpisodeStore((s) => s.titleInclude);
  const titleExclude = useEpisodeStore((s) => s.titleExclude);
  const queryClient = useQueryClient();


  const [tab, setTab] = useState<ShowTab>(
    TABS.includes(initialTab as ShowTab) ? (initialTab as ShowTab) : "episodes",
  );
  const [view, setView] = useState<ViewMode>("list");
  const [cardSize, setCardSize] = useState(3);
  const compact = useLayoutStore((s) => s.compact);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [sort, setSort] = useState<SortKey>("date_desc");
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const lastShiftClickIndex = useRef<number | null>(null);
  // Narrow selectors so unrelated task store writes don't re-render this page.
  const downloadTaskId = useTaskStore((s) => s.downloadTaskId);
  const batchTaskId = useTaskStore((s) => s.batchTaskId);
  const setBatchTask = useTaskStore((s) => s.setBatchTask);

  // Pipeline config from store (for batch start)
  const { tc, llm, engine, targetLang } = usePipelineConfig();

  const pipelineDefaults = usePipelineDefaults();

  const { data: meta } = useQuery({
    queryKey: queryKeys.showMeta(folder),
    queryFn: () => getShowMeta(folder),
  });

  useSeedPipelineFromShow(folder, meta?.pipeline, !!meta);

  const isPolling = !!(downloadTaskId || batchTaskId);
  const { data: episodes, isLoading: episodesLoading } = useQuery({
    queryKey: queryKeys.episodes(folder, pipelineDefaults),
    queryFn: () => getEpisodes(folder, pipelineDefaults),
    placeholderData: keepPreviousData,
    refetchInterval: isPolling ? 5000 : false,
    // While polling, suppress focus refetch so alt-tabs don't double the cadence.
    refetchOnWindowFocus: isPolling ? false : undefined,
  });

  const { downloadMutation, importSubsMutation, isYouTube } = useShowActions(folder, meta);

  const { mutation: refreshMutation } = useFeedRefresh(folder, isYouTube);
  // Page-level only for the empty-state action; the header button subscribes itself.
  const feedRefreshing = useFeedRefreshing();
  // isYouTube derives from meta and reads false while meta loads; refreshing
  // in that window would hit the RSS endpoint on a YouTube show. Disable
  // refresh until meta is in.
  const metaLoaded = !!meta;
  const refreshLabel = !metaLoaded ? "Refresh" : isYouTube ? "Refresh YouTube" : "Refresh RSS";

  const deleteMutation = useMutation({
    mutationFn: (audioPath: string) => deleteAudioFile(audioPath),
    onSuccess: (_data, audioPath) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesForFolder(folder) });
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
      // Drop every per-audioPath cache entry tied to the now-missing file.
      removeQueriesUnderPath(queryClient, audioPath);
    },
  });

  const batchableSelectedRef = useRef<Episode[]>([]);
  const batchMutation = useMutation({
    mutationFn: (args: Parameters<typeof startBatch>[0]) => startBatch(args),
  });

  const all = useMemo(() => episodes ?? [], [episodes]);

  const filterCounts = useMemo(() => ({
    all: all.length,
    ready: all.filter((e) => e.downloaded || (e.transcribed && e.output_dir)).length,
    transcribed: all.filter((e) => e.transcribed).length,
    corrected: all.filter((e) => e.corrected).length,
    translated: all.filter((e) => e.translations.length > 0).length,
    indexed: all.filter((e) => e.indexed).length,
    outdated: all.filter((e) => isOutdated(e)).length,
    verified: all.filter((e) => !!e.verified).length,
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
    if (filter === "corrected") list = list.filter((e) => e.corrected);
    if (filter === "translated") list = list.filter((e) => e.translations.length > 0);
    if (filter === "indexed") list = list.filter((e) => e.indexed);
    if (filter === "outdated") list = list.filter((e) => isOutdated(e));
    if (filter === "verified") list = list.filter((e) => !!e.verified);
    // Sort. Date sorts fall back to feed_order via the shared comparator
    // (`lib/episodeSort`), so this list and episode prev/next nav agree.
    list = [...list].sort((a, b) => {
      switch (sort) {
        case "date_asc": return dateCmp(a, b, 1);
        case "date_desc": return dateCmp(a, b, -1);
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

  // Speaker names per episode for the list column. One roster fetch (shared,
  // cached with the Speakers tab) inverted to stem -> names ordered by airtime.
  const { data: roster } = useQuery({
    queryKey: queryKeys.speakerRoster(folder),
    queryFn: () => getSpeakerRoster(folder),
    enabled: !!folder,
    staleTime: 5 * 60_000,
  });
  const speakersByStem = useMemo(() => {
    const acc = new Map<string, { name: string; secs: number }[]>();
    for (const sp of roster?.speakers ?? []) {
      for (const ep of sp.episodes) {
        const arr = acc.get(ep.stem) ?? [];
        arr.push({ name: sp.name, secs: ep.total_seconds });
        acc.set(ep.stem, arr);
      }
    }
    const out = new Map<string, string[]>();
    for (const [stem, arr] of acc) {
      out.set(stem, arr.sort((a, b) => b.secs - a.secs).map((x) => x.name));
    }
    return out;
  }, [roster]);

  const showName = meta?.name || splitPath(folder.replace(/[\\/]+$/, "")).basename || "Show";

  // Single pass over `filtered` to derive selection-gated subsets.
  // Replaces five separate .filter() walks.
  const selectionDerived = useMemo(() => {
    const downloadable: Episode[] = [];
    const subtitleable: Episode[] = [];
    const missingSubs: Episode[] = [];
    const batchable: Episode[] = [];
    let allSelected = filtered.length > 0;
    for (const e of filtered) {
      const isSel = selected.has(e.id);
      if (!isSel) { allSelected = false; continue; }
      if (!e.removed) {
        subtitleable.push(e);
        if (!e.has_subtitles) missingSubs.push(e);
        if (!e.downloaded) downloadable.push(e);
      }
      if (e.downloaded || e.has_subtitles || (e.transcribed && e.output_dir)) {
        batchable.push(e);
      }
    }
    return { downloadable, subtitleable, missingSubs, batchable, allSelected };
  }, [filtered, selected]);

  const downloadableSelected = selectionDerived.downloadable;
  const subtitleableSelected = selectionDerived.subtitleable;
  const missingSubsSelected = selectionDerived.missingSubs;
  const batchableSelected = selectionDerived.batchable;
  const allSelectableSelected = selectionDerived.allSelected;
  // eslint-disable-next-line react-hooks/refs
  batchableSelectedRef.current = batchableSelected;

  // React Query returns a new mutation *object* every render but the `mutate`
  // function itself is referentially stable — depending on it (not the parent
  // object) keeps our useCallback identity stable across renders.
  const deleteMutate = deleteMutation.mutate;
  const downloadMutate = downloadMutation.mutate;

  const confirmDeleteAudio = useCallback((ep: Episode) => {
    if (!ep.audio_path) return;
    const description = ep.removed
      ? `This will remove the downloaded audio for "${ep.title}". This episode is no longer in the live feed, so re-downloading won't be possible.`
      : `This will remove the downloaded audio for "${ep.title}". You can re-download it later from ${isYouTube ? "YouTube" : "RSS"}.`;
    confirmDialog.open({
      title: "Delete episode audio?",
      description,
      confirmLabel: "Delete",
      variant: "destructive",
      onConfirm: () => deleteMutate(ep.audio_path!),
    });
  }, [isYouTube, deleteMutate]);

  const filteredRef = useRef(filtered);
  // eslint-disable-next-line react-hooks/refs
  filteredRef.current = filtered;

  const toggleSelect = useCallback((id: string, idx: number, shiftKey: boolean) => {
    const lastIdx = lastShiftClickIndex.current;
    lastShiftClickIndex.current = idx;
    setSelected((prev) => {
      const next = new Set(prev);
      if (shiftKey && lastIdx != null) {
        // Mirror the clicked item's toggle direction across the whole range:
        // clicking an already-selected item → unselect range; otherwise select.
        const shouldSelect = !next.has(id);
        const from = Math.min(lastIdx, idx);
        const to = Math.max(lastIdx, idx);
        const list = filteredRef.current;
        for (let i = from; i <= to; i++) {
          if (shouldSelect) next.add(list[i].id);
          else next.delete(list[i].id);
        }
      } else {
        if (next.has(id)) next.delete(id); else next.add(id);
      }
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelected((prev) => {
      const list = filteredRef.current;
      const all = list.length > 0 && list.every((e) => prev.has(e.id));
      return all ? new Set() : new Set(list.map((e) => e.id));
    });
  }, []);

  const goEpisode = useCallback((stem: string) =>
    navigate({ to: "/show/$folder/episode/$stem", params: { folder: encodeURIComponent(folder), stem: encodeURIComponent(stem) } }),
  [navigate, folder]);

  const indexModel = usePipelineConfigStore((s) => s.indexModel);
  const indexChunker = usePipelineConfigStore((s) => s.indexChunker);

  const batchMutate = batchMutation.mutate;
  const runStep = useCallback((step: PipelineInputStep, filteredEpisodes?: Episode[], sourceVersionIds?: Record<string, string>, transcribeSource?: string, force?: boolean) => {
    const source = filteredEpisodes || batchableSelectedRef.current;
    const audioPaths = source.map(batchPath).filter(Boolean) as string[];
    if (audioPaths.length === 0) return;
    const episodes = source.map((e) => ({ title: e.title, stem: e.stem || e.id }));
    batchMutate({
      show_folder: folder,
      audio_paths: audioPaths,
      transcribe: step === "transcribe",
      correct: step === "correct",
      translate: step === "translate",
      index: step === "index",
      model_size: tc.modelSize,
      language: tc.language || languageToISO(meta?.language || ""),
      batch_size: tc.batchSize ?? undefined,
      diarize: tc.diarize,
      clean: tc.clean,
      hf_token: tc.hfToken || undefined,
      num_speakers: tc.numSpeakers ? Number(tc.numSpeakers) : undefined,
      transcribe_source: transcribeSource || "audio",
      sub_lang: languageToISO(meta?.language || "") || "en",
      llm_mode: llm.mode === "api" ? "api" : "ollama",
      llm_provider_profile: llm.mode === "api" ? (llm.providerProfile || undefined) : undefined,
      llm_key_name: llm.mode === "api" ? (llm.keyName || undefined) : undefined,
      llm_model: llm.model || undefined,
      context: llm.context,
      source_lang: llm.sourceLang,
      target_lang: targetLang,
      llm_batch_minutes: llm.batchMinutes,
      engine,
      show_name: meta?.name || "",
      index_model_keys: step === "index" ? [indexModel] : undefined,
      index_chunkings: step === "index" ? [indexChunker] : undefined,
      source_version_ids: sourceVersionIds && Object.keys(sourceVersionIds).length > 0 ? sourceVersionIds : undefined,
      force,
    }, {
      onSuccess: (data) => setBatchTask(data.task_id, folder, episodes, step),
    });
  }, [batchMutate, folder, tc, llm, engine, targetLang, meta?.name, meta?.language, indexModel, indexChunker, setBatchTask]);

  const playEpisode = useCallback((ep: Episode) => {
    if (!ep.audio_path) return;
    const audio = useAudioStore.getState();
    if (ep.audio_path === audio.audioPath) {
      if (audio.isPlaying) audio.pauseAudio();
      else audio.resumeAudio();
      return;
    }
    audio.playEpisode(ep.audio_path, 0, {
      title: ep.title,
      artwork: ep.artwork_url || meta?.artwork_url,
      showName,
      folder,
      stem: ep.stem || ep.id,
    });
  }, [meta?.artwork_url, showName, folder]);

  const downloadEpisode = useCallback((id: string) => {
    downloadMutate({ guids: [id] });
  }, [downloadMutate]);

  const rowDownloading = downloadMutation.isPending || !!downloadTaskId;


  return (
    <div className="h-full flex flex-col overflow-hidden">
      <EditorialHeader
        title={showName}
        breadcrumbs={[
          { label: "Shows", onClick: () => navigate({ to: "/" }) },
          { label: showName },
        ]}
        artworkUrl={meta?.artwork_url ? artworkUrl(folder) : undefined}
        fallbackIcon={Podcast}
        stats={[
          ...(all.length > 0 ? [{ value: all.length, label: `episode${all.length !== 1 ? "s" : ""}` }] : []),
          ...(meta?.speakers && meta.speakers.length > 0
            ? [{ value: meta.speakers.length, label: `speaker${meta.speakers.length !== 1 ? "s" : ""}` }]
            : []),
          ...(meta?.language ? [{ value: meta.language }] : []),
        ]}
        actions={
          <FeedRefreshButton
            onRefresh={() => refreshMutation.mutate()}
            title={isYouTube ? "Refresh YouTube videos" : "Refresh RSS feed"}
            lastUpdate={meta?.last_feed_update}
            idleLabel={refreshLabel}
            disabled={!metaLoaded}
          />
        }
      />

      <div className="flex-1 flex flex-col overflow-hidden">
      <AppSidebar
        pageSections={SIDEBAR_SECTIONS}
        activeItem={tab}
        onItemClick={(key) => setTab(key as ShowTab)}
      />
      <div className="flex-1 flex flex-col overflow-hidden">

      {tab === "episodes" && (<>
      {/* Single toolbar: filter controls and selection actions both stay mounted (toggle via `hidden`)
          so the search input keeps focus across selection changes. */}
      <div className="px-6 py-2 border-b border-border flex items-center gap-2 text-xs">
        <input
          type="checkbox"
          checked={allSelectableSelected}
          onChange={toggleSelectAll}
          className="accent-primary cursor-pointer shrink-0"
          title={allSelectableSelected ? "Unselect all" : "Select all"}
        />
        <SelectionActions
          hidden={selected.size === 0}
          selected={selected}
          filtered={filtered}
          buckets={{
            batchable: batchableSelected,
            downloadable: downloadableSelected,
            subtitleable: subtitleableSelected,
            missingSubs: missingSubsSelected,
          }}
          isYouTube={isYouTube}
          language={meta?.language || ""}
          busy={{
            downloadTask: !!downloadTaskId,
            download: downloadMutation.isPending,
            importSubs: importSubsMutation.isPending,
            batchTask: !!batchTaskId,
            batch: batchMutation.isPending,
          }}
          onClear={() => setSelected(new Set())}
          onDownload={downloadMutate}
          onImportSubs={(req) => importSubsMutation.mutate(req)}
          onRun={runStep}
        />
        <FilterControls
          hidden={selected.size > 0}
          search={search}
          setSearch={setSearch}
          filter={filter}
          setFilter={setFilter}
          filterCounts={filterCounts}
        />
        <div className="flex-1" />
        {view === "card" && (
          <input type="range" min={2} max={8} value={cardSize} onChange={(e) => setCardSize(Number(e.target.value))} className="w-20 accent-primary shrink-0" />
        )}
        <CompactToggle />
        <SortDropdown sort={sort} setSort={setSort} />
        <div className="flex border border-border rounded overflow-hidden shrink-0">
          <button onClick={() => setView("list")} className={`px-1.5 py-1 transition ${view === "list" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`} title="List view">
            <List className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => setView("card")} className={`px-1.5 py-1 transition ${view === "card" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`} title="Card view">
            <LayoutGrid className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {[batchMutation, downloadMutation, importSubsMutation]
        .filter((m) => m.isError)
        .map((m, i) => (
          <div key={i} className="px-6 py-2 flex items-center gap-2">
            <ErrorAlert error={m.error} onDismiss={() => m.reset()} compact className="flex-1" />
          </div>
        ))}

      {/* Episode list */}
      <div className="flex-1 overflow-y-auto">
        {view === "list" ? (
          <div className="divide-y divide-border/50">
            {filtered.map((ep, i) => {
              const isCurrent = !!ep.audio_path && ep.audio_path === audioPath;
              return (
                <EpisodeRow
                  key={ep.id}
                  ep={ep}
                  index={i}
                  selected={selected.has(ep.id)}
                  onToggle={toggleSelect}
                  onOpen={goEpisode}
                  onPlay={playEpisode}
                  onDownload={downloadEpisode}
                  onDelete={confirmDeleteAudio}
                  downloading={rowDownloading}
                  isPlaying={isCurrent && audioIsPlaying}
                  isCurrent={isCurrent}
                  speakers={ep.stem ? speakersByStem.get(ep.stem) : undefined}
                />
              );
            })}
          </div>
        ) : (
          <div
            className={`p-6 grid ${compact ? "gap-2" : "gap-4"}`}
            style={{ gridTemplateColumns: episodeCardGridTemplate(cardSize) }}
          >
            {filtered.map((ep) => {
              const isCurrent = !!ep.audio_path && ep.audio_path === audioPath;
              return (
                <EpisodeCard
                  key={ep.id}
                  ep={ep}
                  onOpen={goEpisode}
                  onPlay={playEpisode}
                  onDownload={downloadEpisode}
                  onDelete={confirmDeleteAudio}
                  downloading={rowDownloading}
                  isPlaying={isCurrent && audioIsPlaying}
                  isCurrent={isCurrent}
                  speakers={ep.stem ? speakersByStem.get(ep.stem) : undefined}
                />
              );
            })}
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
              description={isYouTube ? "Refresh to fetch videos from YouTube." : "Refresh the RSS feed to fetch episodes, or drop audio files here."}
              steps={[
                { label: isYouTube ? "Refresh to pull the latest YouTube videos" : "Refresh to pull the latest RSS episodes" },
                { label: "Download audio to enable transcription" },
                { label: "Transcribe, correct, and index" },
              ]}
              action={{
                label: refreshLabel,
                onClick: () => refreshMutation.mutate(),
                disabled: feedRefreshing || !metaLoaded,
              }}
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
        <div className="px-6 py-2 border-t border-border">
          <ErrorAlert
            error={refreshMutation.error}
            onRetry={() => refreshMutation.mutate()}
            onDismiss={() => refreshMutation.reset()}
            compact
          />
        </div>
      )}
      </>)}

      {tab === "search" && (
        <SearchPanel
          scope="show"
          showName={showName}
          folder={folder}
          artwork={meta?.artwork_url ? artworkUrl(folder) : undefined}
        />
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
    </div>
    </div>
  );
}

interface FilterControlsProps {
  hidden: boolean;
  search: string;
  setSearch: (s: string) => void;
  filter: StatusFilter;
  setFilter: (f: StatusFilter) => void;
  filterCounts: Record<StatusFilter, number>;
}

function FilterControls({ hidden, search, setSearch, filter, setFilter, filterCounts }: FilterControlsProps) {
  return (
    <div className={`flex items-center gap-2 ${hidden ? "hidden" : ""}`}>
      <input
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search episodes..."
        className="input w-40 py-1.5 text-xs"
      />
      <select
        value={filter}
        onChange={(e) => setFilter(e.target.value as StatusFilter)}
        className="bg-secondary text-secondary-foreground text-xs rounded px-2 py-1.5 border border-border"
      >
        <option value="all">All ({filterCounts.all})</option>
        <option value="ready">Ready ({filterCounts.ready})</option>
        <option value="transcribed">Transcribed ({filterCounts.transcribed})</option>
        <option value="corrected">Corrected ({filterCounts.corrected})</option>
        <option value="translated">Translated ({filterCounts.translated})</option>
        <option value="indexed">Indexed ({filterCounts.indexed})</option>
        <option value="outdated">Outdated ({filterCounts.outdated})</option>
        <option value="verified">Verified ({filterCounts.verified})</option>
      </select>
      <FilterDropdown />
    </div>
  );
}

interface SelectionActionsProps {
  hidden: boolean;
  selected: Set<string>;
  filtered: Episode[];
  buckets: {
    batchable: Episode[];
    downloadable: Episode[];
    subtitleable: Episode[];
    missingSubs: Episode[];
  };
  isYouTube: boolean;
  language: string;
  busy: {
    downloadTask: boolean;
    download: boolean;
    importSubs: boolean;
    batchTask: boolean;
    batch: boolean;
  };
  onClear: () => void;
  onDownload: (req: { guids: string[]; force?: boolean }) => void;
  onImportSubs: (req: { ids: string[]; lang: string }) => void;
  onRun: React.ComponentProps<typeof PipelineButtons>["onRun"];
}

function SelectionActions({
  hidden,
  selected,
  filtered,
  buckets,
  isYouTube,
  language,
  busy,
  onClear,
  onDownload,
  onImportSubs,
  onRun,
}: SelectionActionsProps) {
  const { batchable, downloadable, subtitleable, missingSubs } = buckets;
  return (
    <div className={`flex items-center gap-2 ${hidden ? "hidden" : ""}`}>
      <span className="text-muted-foreground shrink-0">
        {selected.size} selected
        {batchable.length < selected.size && (
          <span title="Some selected episodes have no audio or subtitles and cannot be processed">
            {" "}({batchable.length} ready)
          </span>
        )}
      </span>
      <Button onClick={onClear} variant="ghost" size="sm" className="text-xs h-6 px-1.5 shrink-0">Clear</Button>
      <DownloadDropdown
        isYouTube={isYouTube}
        showLanguage={language}
        onDownload={() => {
          const selectedFiltered = filtered.filter((e) => selected.has(e.id));
          if (downloadable.length > 0) {
            onDownload({ guids: downloadable.map((e) => e.id) });
          } else if (selectedFiltered.length > 0) {
            const alreadyCount = selectedFiltered.length;
            confirmDialog.open({
              title: "Re-download?",
              description: `${alreadyCount} selected episode${alreadyCount !== 1 ? "s are" : " is"} already downloaded. This will re-download and overwrite the existing files.`,
              confirmLabel: "Re-download",
              onConfirm: () => onDownload({ guids: selectedFiltered.map((e) => e.id), force: true }),
            });
          }
        }}
        onImportSubs={(lang) => onImportSubs({ ids: subtitleable.map((e) => e.id), lang })}
        subsLabel={
          missingSubs.length > 0
            ? `Subtitles (${missingSubs.length})`
            : subtitleable.length > 0
              ? `Re-import subtitles (${subtitleable.length})`
              : "Subtitles"
        }
        subsEnabled={subtitleable.length > 0}
        audioLabel={`Audio${downloadable.length > 0 ? ` (${downloadable.length})` : ""}`}
        showAudio={true}
        audioEnabled={downloadable.length > 0 || selected.size > 0}
        disabled={busy.downloadTask || busy.download || busy.importSubs}
        className="text-xs h-7 px-2 shrink-0"
      />
      <PipelineButtons
        disabled={batchable.length === 0 || busy.batchTask || busy.batch}
        episodes={batchable}
        showLanguage={language}
        onRun={onRun}
      />
    </div>
  );
}
