import { memo, useRef } from "react";
import type { Episode } from "@/api/types";
import { Play, Pause, Download, Trash2, Captions, CloudOff } from "lucide-react";
import { formatDuration, formatDate } from "@/lib/utils";
import { useLayoutStore } from "@/stores";
import { StatusChips } from "./StatusChips";

export interface EpisodeRowProps {
  ep: Episode;
  index: number;
  selected: boolean;
  onToggle: (id: string, index: number, shiftKey: boolean) => void;
  onOpen: (stem: string) => void;
  onPlay: (ep: Episode) => void;
  onDownload?: (id: string) => void;
  onDelete: (ep: Episode) => void;
  downloading?: boolean;
  /** True only while audio is actively playing (paused → false). */
  isPlaying: boolean;
  /** True when this episode is the loaded track, regardless of play/pause. */
  isCurrent: boolean;
}

function EpisodeRowInner({ ep, index, selected, onToggle, onOpen, onPlay, onDownload, onDelete, downloading, isPlaying, isCurrent }: EpisodeRowProps) {
  const compact = useLayoutStore((s) => s.compact);
  const shiftRef = useRef(false);

  const handleOpen = () => onOpen(ep.stem || ep.id);
  const handlePlay = () => onPlay(ep);
  const handleDownload = onDownload ? () => onDownload(ep.id) : undefined;
  const handleDelete = () => onDelete(ep);

  return (
    <div className="flex items-center gap-3 px-6 py-3 hover:bg-accent/50 transition group">
      <input type="checkbox" checked={selected} onMouseDown={(e) => { shiftRef.current = e.shiftKey; }} onChange={() => onToggle(ep.id, index, shiftRef.current)} className="accent-primary cursor-pointer shrink-0" />
      {ep.artwork_url && (
        <img src={ep.artwork_url} alt={ep.title} className="w-12 h-12 object-cover rounded shrink-0" loading="lazy" />
      )}
      {ep.episode_number != null && (
        <span className="text-xs text-muted-foreground w-8 text-right shrink-0 tabular-nums">#{ep.episode_number}</span>
      )}
      <div className="min-w-0 flex-1 space-y-0.5">
        <button
          onClick={handleOpen}
          className={`text-left text-sm hover:text-primary cursor-pointer flex items-center gap-1.5 max-w-full ${ep.removed ? "text-muted-foreground" : "text-foreground"}`}
        >
          {ep.removed && (
            <span title="No longer in the live feed, kept locally" className="shrink-0">
              <CloudOff className="w-3 h-3 text-muted-foreground" />
            </span>
          )}
          <span className="truncate">{ep.title}</span>
        </button>
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
          {ep.pub_date && <span>{formatDate(ep.pub_date)}</span>}
          {ep.duration > 0 && <span>{formatDuration(ep.duration)}</span>}
          {!compact && <StatusChips ep={ep} />}
        </div>
      </div>
      {/* Fixed-width slots so each action keeps its column whether or not the
          icon is present (subtitles / play vs download / delete all vary). */}
      <div className="flex items-center gap-1.5 shrink-0">
        <span className="w-5 flex justify-center">
          {ep.has_subtitles && (
            <span title="Subtitles cached" aria-label="Subtitles cached" className="text-muted-foreground/70">
              <Captions className="w-3.5 h-3.5" />
            </span>
          )}
        </span>
        <span className="w-5 flex justify-center">
          {ep.audio_path ? (
            <button
              onClick={handlePlay}
              title={isPlaying ? "Pause" : isCurrent ? "Resume" : "Play"}
              aria-label={isPlaying ? "Pause" : "Play"}
              className={`transition ${isPlaying ? "text-success" : isCurrent ? "text-primary" : "text-muted-foreground hover:text-foreground"}`}
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            </button>
          ) : handleDownload ? (
            <button onClick={handleDownload} disabled={downloading} title="Download audio" aria-label="Download audio" className="text-muted-foreground hover:text-foreground transition disabled:opacity-50">
              <Download className="w-3.5 h-3.5" />
            </button>
          ) : null}
        </span>
        <span className="w-5 flex justify-center">
          {ep.audio_path && (
            <button
              onClick={handleDelete}
              title="Delete audio"
              aria-label="Delete audio"
              className="text-muted-foreground hover:text-destructive transition opacity-0 group-hover:opacity-100"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </span>
      </div>
    </div>
  );
}

export const EpisodeRow = memo(EpisodeRowInner);
