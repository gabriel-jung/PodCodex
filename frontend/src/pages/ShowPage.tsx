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
import { useAudioStore } from "@/stores";
import { Button } from "@/components/ui/button";
import { ArrowLeft, RefreshCw, Play, ExternalLink, Download, CheckCircle, Trash2, Podcast, Search } from "lucide-react";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import { EmptyState } from "@/components/ui/empty-state";
import ShowSettings from "@/components/show/ShowSettings";
import SearchPanel from "@/components/search/SearchPanel";

import { errorMessage, formatDuration, formatDate } from "@/lib/utils";

type ShowTab = "episodes" | "search" | "settings";
type ViewMode = "list" | "card";
type StatusFilter = "all" | "downloaded" | "not_downloaded" | "transcribed" | "polished" | "translated" | "synthesized" | "indexed";

export default function ShowPage({ folder }: { folder: string }) {
  const navigate = useNavigate();
  const { seekTo, setAudioMeta, audioPath } = useAudioStore();
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
  }, [all, search, filter]);

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

function StatusDots({ ep }: { ep: Episode }) {
  return (
    <div className="flex gap-1.5 items-center">
      {ep.transcribed && <span className="w-2 h-2 rounded-full bg-blue-500" title="Transcribed" />}
      {ep.polished && <span className="w-2 h-2 rounded-full bg-purple-500" title="Polished" />}
      {ep.translations.length > 0 && <span className="w-2 h-2 rounded-full bg-teal-500" title={`Translated (${ep.translations.join(", ")})`} />}
      {ep.synthesized && <span className="w-2 h-2 rounded-full bg-orange-500" title="Synthesized" />}
      {ep.indexed && <span className="w-2 h-2 rounded-full bg-yellow-500" title="Indexed" />}
    </div>
  );
}

function EpisodeRow({ ep, selected, onToggle, onOpen, onPlay, onDownload, onDelete, downloading, isPlaying }: {
  ep: Episode; selected: boolean; onToggle: () => void;
  onOpen: () => void; onPlay: () => void; onDownload: () => void; onDelete: () => void; downloading: boolean; isPlaying: boolean;
}) {
  const canDownload = !ep.downloaded && !!ep.audio_url;
  return (
    <div className="flex items-center gap-3 px-6 py-3 hover:bg-accent/50 transition group">
      {canDownload ? (
        <input type="checkbox" checked={selected} onChange={onToggle} className="accent-primary shrink-0" />
      ) : ep.downloaded ? (
        <CheckCircle className="w-4 h-4 text-green-500 shrink-0" title="Downloaded" />
      ) : (
        <div className="w-4" />
      )}
      {ep.artwork_url && (
        <img src={ep.artwork_url} alt="" className="w-8 h-8 rounded shrink-0" loading="lazy" />
      )}
      {ep.episode_number != null && (
        <span className="text-xs text-muted-foreground w-8 text-right shrink-0">#{ep.episode_number}</span>
      )}
      <button
        onClick={onOpen}
        className="flex-1 text-left text-sm truncate text-foreground hover:text-primary cursor-pointer"
      >
        {ep.title}
      </button>
      <StatusDots ep={ep} />
      <span className="text-xs text-muted-foreground w-20 text-right shrink-0">{formatDate(ep.pub_date)}</span>
      <span className="text-xs text-muted-foreground w-12 text-right shrink-0">{formatDuration(ep.duration)}</span>
      <div className="w-20 flex justify-end gap-1 shrink-0">
        {ep.audio_path && (
          <button onClick={onPlay} title="Play" className={`opacity-0 group-hover:opacity-100 transition ${isPlaying ? "text-green-400 !opacity-100" : "text-muted-foreground hover:text-foreground"}`}>
            <Play className="w-3.5 h-3.5" />
          </button>
        )}
        {ep.audio_path && (
          <button onClick={onDelete} title="Delete audio" className="text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100 transition">
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
        {canDownload && !selected && (
          <button onClick={onDownload} disabled={downloading} title="Download" className="text-green-400 hover:text-green-300 opacity-0 group-hover:opacity-100 transition">
            <Download className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}

function PipelineBar({ ep }: { ep: Episode }) {
  const steps = [ep.downloaded, ep.transcribed, ep.polished, ep.translations.length > 0, ep.synthesized, ep.indexed];
  const done = steps.filter(Boolean).length;
  const pct = (done / steps.length) * 100;
  if (done === 0) return null;
  const color = done === steps.length ? "bg-green-500" : "bg-primary";
  return (
    <div className="h-1 bg-muted/50 rounded-full overflow-hidden">
      <div className={`h-full ${color} transition-all duration-300`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function EpisodeCard({ ep, onOpen, onPlay, onDownload, downloading, isPlaying }: {
  ep: Episode; onOpen: () => void; onPlay: () => void; onDownload: () => void; downloading: boolean; isPlaying: boolean;
}) {
  const canDownload = !ep.downloaded && !!ep.audio_url;
  return (
    <div
      className="group relative bg-card border border-border rounded-xl overflow-hidden hover:border-muted-foreground/30 transition cursor-pointer"
      onClick={onOpen}
    >
      {/* Artwork / placeholder */}
      <div className="relative aspect-square bg-muted">
        {ep.artwork_url ? (
          <img src={ep.artwork_url} alt="" className="w-full h-full object-cover" loading="lazy" />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-muted-foreground/30">
            <Play className="w-10 h-10" />
          </div>
        )}

        {/* Hover overlay with play/download buttons */}
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-colors flex items-center justify-center gap-3 opacity-0 group-hover:opacity-100">
          {ep.audio_path && (
            <button
              onClick={(e) => { e.stopPropagation(); onPlay(); }}
              className={`w-10 h-10 rounded-full bg-white/90 flex items-center justify-center transition hover:scale-110 ${isPlaying ? "ring-2 ring-green-400" : ""}`}
              title="Play"
            >
              <Play className="w-5 h-5 text-black fill-black ml-0.5" />
            </button>
          )}
          {canDownload && (
            <button
              onClick={(e) => { e.stopPropagation(); onDownload(); }}
              disabled={downloading}
              className="w-10 h-10 rounded-full bg-white/90 flex items-center justify-center transition hover:scale-110"
              title="Download"
            >
              <Download className="w-5 h-5 text-black" />
            </button>
          )}
        </div>

        {/* Top-right: status badges */}
        <div className="absolute top-2 right-2 flex gap-1">
          <StatusDots ep={ep} />
        </div>

        {/* Top-left: episode number */}
        {ep.episode_number != null && (
          <span className="absolute top-2 left-2 text-[10px] bg-black/60 text-white px-1.5 py-0.5 rounded-md font-medium">
            #{ep.episode_number}
          </span>
        )}

        {/* Now playing indicator */}
        {isPlaying && (
          <span className="absolute bottom-2 right-2 text-[10px] bg-green-500 text-white px-1.5 py-0.5 rounded-md font-medium animate-pulse">
            Playing
          </span>
        )}
      </div>

      {/* Text content */}
      <div className="p-3 space-y-1.5">
        <p className="text-sm font-medium line-clamp-2 leading-snug">{ep.title}</p>
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          {ep.pub_date && <span>{formatDate(ep.pub_date)}</span>}
          {ep.duration > 0 && <span>{formatDuration(ep.duration)}</span>}
        </div>
        <PipelineBar ep={ep} />
      </div>
    </div>
  );
}
