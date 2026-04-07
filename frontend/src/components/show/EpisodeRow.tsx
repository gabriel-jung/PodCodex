import { useRef } from "react";
import type { Episode } from "@/api/types";
import { CheckCircle, Play, Trash2, Download } from "lucide-react";
import { formatDuration, formatDate } from "@/lib/utils";
import { StatusChips } from "./StatusChips";

export interface EpisodeRowProps {
  ep: Episode;
  selected: boolean;
  onToggle: (shiftKey: boolean) => void;
  onOpen: () => void;
  onPlay: () => void;
  onDownload: () => void;
  onDelete: () => void;
  downloading: boolean;
  isPlaying: boolean;
}

export function EpisodeRow({ ep, selected, onToggle, onOpen, onPlay, onDownload, onDelete, downloading, isPlaying }: EpisodeRowProps) {
  const shiftRef = useRef(false);
  const needsDownload = !ep.downloaded;
  return (
    <div className="flex items-center gap-3 px-6 py-3 hover:bg-accent/50 transition group">
      {needsDownload || ep.downloaded ? (
        <div className="flex items-center gap-1.5 shrink-0">
          <input type="checkbox" checked={selected} onMouseDown={(e) => { shiftRef.current = e.shiftKey; }} onChange={() => onToggle(shiftRef.current)} className="accent-primary cursor-pointer" />
          {ep.downloaded && <CheckCircle className="w-3.5 h-3.5 text-success" />}
        </div>
      ) : (
        <div className="w-4" />
      )}
      {ep.artwork_url && (
        <img src={ep.artwork_url} alt={ep.title} className="w-8 h-6 object-cover rounded shrink-0" loading="lazy" />
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
      <StatusChips ep={ep} />
      <span className="text-xs text-muted-foreground w-20 text-right shrink-0">{formatDate(ep.pub_date)}</span>
      <span className="text-xs text-muted-foreground w-12 text-right shrink-0">{formatDuration(ep.duration)}</span>
      <div className="w-20 flex justify-end gap-2.5 shrink-0">
        {ep.audio_path && (
          <button onClick={onPlay} title="Play" className={`transition ${isPlaying ? "text-success" : "text-muted-foreground hover:text-foreground"}`}>
            <Play className="w-3.5 h-3.5" />
          </button>
        )}
        {ep.audio_path && (
          <button onClick={onDelete} title="Delete audio" className="text-muted-foreground hover:text-destructive transition">
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
        {needsDownload && !selected && (
          <button onClick={onDownload} disabled={downloading} title="Download" className="text-success hover:text-success/80 transition">
            <Download className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
