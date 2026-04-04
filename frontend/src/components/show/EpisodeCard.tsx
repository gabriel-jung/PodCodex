import type { Episode } from "@/api/types";
import { Play, Download } from "lucide-react";
import { formatDuration, formatDate } from "@/lib/utils";
import { StatusChips } from "./StatusChips";
import { PipelineBar } from "./PipelineBar";

export interface EpisodeCardProps {
  ep: Episode;
  onOpen: () => void;
  onPlay: () => void;
  onDownload: () => void;
  downloading: boolean;
  isPlaying: boolean;
}

export function EpisodeCard({ ep, onOpen, onPlay, onDownload, downloading, isPlaying }: EpisodeCardProps) {
  const needsDownload = !ep.downloaded;
  return (
    <div
      className="group relative bg-card border border-border rounded-xl overflow-hidden hover:border-muted-foreground/30 transition cursor-pointer"
      onClick={onOpen}
    >
      {/* Artwork / placeholder */}
      <div className="relative h-36 bg-muted">
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
          {needsDownload && (
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
          <StatusChips ep={ep} compact />
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
