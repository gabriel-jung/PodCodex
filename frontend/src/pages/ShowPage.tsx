import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import {
  downloadEpisodes,
  refreshRSS,
  getEpisodes,
  getShowMeta,
  deleteAudioFile,
} from "@/api/client";
import type { Episode } from "@/api/types";
import ProgressBar from "@/components/editor/ProgressBar";
import { useAudioStore, useConfigStore } from "@/stores";
import { Button } from "@/components/ui/button";
import { ArrowLeft, RefreshCw, Podcast, Search } from "lucide-react";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import { EmptyState } from "@/components/ui/empty-state";
import ShowSettings from "@/components/show/ShowSettings";
import { EpisodeRow } from "@/components/show/EpisodeRow";
import { EpisodeCard } from "@/components/show/EpisodeCard";
import SearchPanel from "@/components/search/SearchPanel";

import { errorMessage } from "@/lib/utils";

type ShowTab = "episodes" | "search" | "settings";
type ViewMode = "list" | "card";
type StatusFilter = "all" | "downloaded" | "not_downloaded" | "transcribed" | "polished" | "translated" | "synthesized" | "indexed";

export default function ShowPage({ folder }: { folder: string }) {
  const navigate = useNavigate();
  const { seekTo, setAudioMeta, audioPath } = useAudioStore();
  const { minDurationMinutes, setMinDurationMinutes } = useConfigStore();
  const queryClient = useQueryClient();

  const [tab, setTab] = useState<ShowTab>("episodes");
  const [view, setView] = useState<ViewMode>("list");
  const [cardSize, setCardSize] = useState(3); // 1-5, maps to grid columns
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [downloadTaskId, setDownloadTaskId] = useState<string | null>(null);

  const { data: meta } = useQuery({
    queryKey: ["showMeta", folder],
    queryFn: () => getShowMeta(folder),
  });

  const { data: episodes, isLoading: episodesLoading } = useQuery({
    queryKey: ["episodes", folder],
    queryFn: () => getEpisodes(folder),
    // Refetch every 5s while a download is in progress
    refetchInterval: downloadTaskId ? 5000 : false,
  });

  const refreshMutation = useMutation({
    mutationFn: () => refreshRSS(folder),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["episodes", folder] });
    },
  });

  const downloadMutation = useMutation({
    mutationFn: (guids: string[]) => downloadEpisodes(folder, guids),
    onSuccess: (data) => {
      setDownloadTaskId(data.task_id);
      setSelected(new Set());
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (audioPath: string) => deleteAudioFile(audioPath),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["episodes", folder] });
    },
  });

  const all = episodes ?? [];

  const filtered = useMemo(() => {
    let list = all;
    if (minDurationMinutes > 0) {
      const minSec = minDurationMinutes * 60;
      list = list.filter((e) => e.duration >= minSec);
    }
    if (search) {
      const q = search.toLowerCase();
      list = list.filter((e) => e.title.toLowerCase().includes(q));
    }
    if (filter === "downloaded") list = list.filter((e) => e.downloaded);
    if (filter === "not_downloaded") list = list.filter((e) => !e.downloaded);
    if (filter === "transcribed") list = list.filter((e) => e.transcribed);
    if (filter === "polished") list = list.filter((e) => e.polished);
    if (filter === "translated") list = list.filter((e) => e.translations.length > 0);
    if (filter === "synthesized") list = list.filter((e) => e.synthesized);
    if (filter === "indexed") list = list.filter((e) => e.indexed);
    return list;
  }, [all, search, filter, minDurationMinutes]);

  const showName = meta?.name || folder.replace(/\/+$/, "").split("/").pop() || "Show";

  const downloadableSelected = filtered.filter(
    (e) => selected.has(e.id) && !e.downloaded && e.audio_url,
  );
  const availableForDownload = filtered.filter((e) => !e.downloaded && e.audio_url);
  const allAvailableSelected =
    availableForDownload.length > 0 &&
    availableForDownload.every((e) => selected.has(e.id));

  const toggleSelect = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });

  const toggleSelectAll = () =>
    setSelected(allAvailableSelected ? new Set() : new Set(availableForDownload.map((e) => e.id)));

  const goEpisode = (stem: string) =>
    navigate({ to: "/show/$folder/episode/$stem", params: { folder: encodeURIComponent(folder), stem: encodeURIComponent(stem) } });

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

      {/* Tabs */}
      <div className="px-6 border-b border-border flex gap-1">
        {(["episodes", "search", "settings"] as const).map((t) => (
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

      {tab === "episodes" && (<>
      {/* Filters */}
      <div className="px-6 py-3 border-b border-border flex items-center gap-3 flex-wrap">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search episodes..."
          className="input w-48 py-1.5"
        />
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value as StatusFilter)}
          className="bg-secondary text-secondary-foreground text-xs rounded px-2 py-1.5 border border-border"
        >
          <option value="all">All ({all.length})</option>
          <option value="downloaded">Downloaded ({all.filter((e) => e.downloaded).length})</option>
          <option value="not_downloaded">Not downloaded ({all.filter((e) => !e.downloaded).length})</option>
          <option value="transcribed">Transcribed ({all.filter((e) => e.transcribed).length})</option>
          <option value="polished">Polished ({all.filter((e) => e.polished).length})</option>
          <option value="translated">Translated ({all.filter((e) => e.translations.length > 0).length})</option>
          <option value="synthesized">Synthesized ({all.filter((e) => e.synthesized).length})</option>
          <option value="indexed">Indexed ({all.filter((e) => e.indexed).length})</option>
        </select>
        <div className="flex items-center gap-1.5">
          <label className="text-xs text-muted-foreground whitespace-nowrap">Min</label>
          <input
            type="number"
            min={0}
            step={5}
            value={minDurationMinutes || ""}
            onChange={(e) => setMinDurationMinutes(Math.max(0, Number(e.target.value)))}
            placeholder="0"
            className="input w-14 py-1.5 text-xs text-center"
          />
          <span className="text-xs text-muted-foreground">min</span>
        </div>
        <div className="flex-1" />
        <span className="text-xs text-muted-foreground">{filtered.length} shown</span>
        <div className="flex items-center gap-2">
          {view === "card" && (
            <input
              type="range"
              min={1}
              max={5}
              value={cardSize}
              onChange={(e) => setCardSize(Number(e.target.value))}
              className="w-20 accent-primary"
            />
          )}
          <div className="flex border border-border rounded overflow-hidden">
            {(["list", "card"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-2 py-1 text-xs transition ${
                  view === v ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {v === "list" ? "List" : "Cards"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Selection actions */}
      {(selected.size > 0 || availableForDownload.length > 0) && (
        <div className="px-6 py-2 border-b border-border flex items-center gap-3 text-xs">
          {availableForDownload.length > 0 && (
            <Button onClick={toggleSelectAll} variant="link" size="sm" className="px-0">
              {allAvailableSelected ? "Unselect all" : `Select all downloadable (${availableForDownload.length})`}
            </Button>
          )}
          {selected.size > 0 && (
            <>
              <span className="text-muted-foreground">{selected.size} selected</span>
              <Button onClick={() => setSelected(new Set())} variant="ghost" size="sm">Clear</Button>
              {downloadableSelected.length > 0 && (
                <Button
                  onClick={() => downloadMutation.mutate(downloadableSelected.map((e) => e.id))}
                  disabled={downloadMutation.isPending || !!downloadTaskId}
                  size="sm"
                >
                  {downloadMutation.isPending ? "Starting..." : `Download ${downloadableSelected.length}`}
                </Button>
              )}
            </>
          )}
        </div>
      )}

      {/* Episode list */}
      <div className="flex-1 overflow-y-auto">
        {view === "list" ? (
          <div className="divide-y divide-border/50">
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

      {/* Download progress */}
      {downloadTaskId && (
        <div className="border-t border-border">
          <ProgressBar
            taskId={downloadTaskId}
            onComplete={() => {
              setDownloadTaskId(null);
              queryClient.refetchQueries({ queryKey: ["episodes", folder] });
            }}
          />
        </div>
      )}
      {refreshMutation.isError && (
        <div className="px-6 py-2 border-t border-border text-destructive text-xs">
          {errorMessage(refreshMutation.error)}
        </div>
      )}
      </>)}

      {tab === "search" && (
        <SearchPanel scope="show" folder={folder} showName={showName} />
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
