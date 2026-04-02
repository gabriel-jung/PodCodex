import type { Episode } from "@/api/types";
import { CheckCircle, Play, Trash2, Download } from "lucide-react";
import { formatDuration, formatDate } from "@/lib/utils";
import { StatusDots } from "./StatusDots";

export interface EpisodeRowProps {
  ep: Episode;
  selected: boolean;
  onToggle: () => void;
  onOpen: () => void;
  onPlay: () => void;
  onDownload: () => void;
  onDelete: () => void;
  downloading: boolean;
  isPlaying: boolean;
}

export function EpisodeRow({ ep, selected, onToggle, onOpen, onPlay, onDownload, onDelete, downloading, isPlaying }: EpisodeRowProps) {
  const canDownload = !ep.downloaded;
  return (
    <div className="flex items-center gap-3 px-6 py-3 hover:bg-accent/50 transition group">
      {canDownload || ep.downloaded ? (
        <div className="flex items-center gap-1.5 shrink-0">
          <input type="checkbox" checked={selected} onChange={onToggle} className="accent-primary" />
          {ep.downloaded && <CheckCircle className="w-3.5 h-3.5 text-green-500" />}
        </div>
      ) : (
        <div className="w-4" />
      )}
      {ep.artwork_url && (
        <img src={ep.artwork_url} alt="" className="w-8 h-6 object-cover rounded shrink-0" loading="lazy" />
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
      <div className="w-20 flex justify-end gap-2.5 shrink-0">
        {ep.audio_path && (
          <button onClick={onPlay} title="Play" className={`transition ${isPlaying ? "text-green-400" : "text-muted-foreground hover:text-foreground"}`}>
            <Play className="w-3.5 h-3.5" />
          </button>
        )}
        {ep.audio_path && (
          <button onClick={onDelete} title="Delete audio" className="text-muted-foreground hover:text-destructive transition">
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
        {canDownload && !selected && (
          <button onClick={onDownload} disabled={downloading} title="Download" className="text-green-400 hover:text-green-300 transition">
            <Download className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
