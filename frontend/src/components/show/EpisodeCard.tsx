import { memo, useRef, useState } from "react";
import type { Episode } from "@/api/types";
import { Play, Download, MoreVertical, Captions, CloudOff } from "lucide-react";
import { formatDuration, formatDate } from "@/lib/utils";
import { StatusChips } from "./StatusChips";
import { EpisodeMenu, type PipelineStep } from "./EpisodeMenu";

export interface EpisodeCardProps {
  ep: Episode;
  onOpen: (stem: string) => void;
  onPlay: (ep: Episode) => void;
  onDownload?: (id: string) => void;
  onDelete?: (ep: Episode) => void;
  onProcess?: (step: PipelineStep, ep: Episode) => void;
  downloading?: boolean;
  isPlaying: boolean;
}

function EpisodeCardInner({ ep, onOpen, onPlay, onDownload, onDelete, onProcess, downloading, isPlaying }: EpisodeCardProps) {
  const menuTriggerRef = useRef<HTMLButtonElement>(null);
  const [isWide, setIsWide] = useState(false);

  const onImgLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalWidth: w, naturalHeight: h } = e.currentTarget;
    if (w && h && w / h > 1.2) setIsWide(true);
  };

  const onContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    menuTriggerRef.current?.click();
  };

  const handleOpen = () => onOpen(ep.stem || ep.id);
  const handlePlay = () => onPlay(ep);
  const handleDownload = onDownload ? () => onDownload(ep.id) : undefined;
  const handleDelete = onDelete ? () => onDelete(ep) : undefined;
  const handleProcess = onProcess ? (step: PipelineStep) => onProcess(step, ep) : undefined;

  return (
    <div
      className="group relative bg-card border border-border rounded-lg overflow-hidden hover:border-muted-foreground/30 transition cursor-pointer"
      onClick={handleOpen}
      onContextMenu={onContextMenu}
    >
      {/* Artwork — square container; wide images (YouTube) letterbox over a blurred backdrop of themselves */}
      <div className={`relative bg-muted overflow-hidden ${isWide ? "aspect-video" : "aspect-square"}`}>
        {ep.artwork_url ? (
          <img
            src={ep.artwork_url}
            alt={ep.title}
            loading="lazy"
            onLoad={onImgLoad}
            className="w-full h-full object-cover transition duration-500 group-hover:scale-[1.02]"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-muted-foreground/30">
            <Play className="w-10 h-10" />
          </div>
        )}

        {/* Bottom gradient for overlay legibility */}
        <div className="absolute inset-x-0 bottom-0 h-1/3 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />

        {/* Hover overlay with play/download button */}
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
          {ep.audio_path ? (
            <button
              onClick={(e) => { e.stopPropagation(); handlePlay(); }}
              className={`w-12 h-12 rounded-full bg-white/95 flex items-center justify-center transition hover:scale-110 shadow-lg ${isPlaying ? "ring-2 ring-success ring-offset-2 ring-offset-black/30" : ""}`}
              title="Play"
              aria-label="Play"
            >
              <Play className="w-5 h-5 text-black fill-black ml-0.5" />
            </button>
          ) : handleDownload && (
            <button
              onClick={(e) => { e.stopPropagation(); handleDownload(); }}
              disabled={downloading}
              className="w-12 h-12 rounded-full bg-white/95 flex items-center justify-center transition hover:scale-110 disabled:opacity-50 shadow-lg"
              title="Download audio"
              aria-label="Download audio"
            >
              <Download className="w-5 h-5 text-black" />
            </button>
          )}
        </div>

        {/* Top-left: episode number + subtitle/removed indicators */}
        <div className="absolute top-2 left-2 flex gap-1 items-center">
          {ep.episode_number != null && (
            <span className="text-2xs bg-black/65 backdrop-blur-sm text-white px-1.5 py-0.5 rounded-md font-medium tabular-nums">
              #{ep.episode_number}
            </span>
          )}
          {ep.has_subtitles && (
            <span
              className="w-5 h-5 bg-black/65 backdrop-blur-sm text-white rounded-md flex items-center justify-center"
              title="Subtitles cached"
              aria-label="Subtitles cached"
            >
              <Captions className="w-3 h-3" />
            </span>
          )}
          {ep.removed && (
            <span
              className="w-5 h-5 bg-black/65 backdrop-blur-sm text-white/70 rounded-md flex items-center justify-center"
              title="No longer in the live feed — kept locally"
              aria-label="No longer in feed"
            >
              <CloudOff className="w-3 h-3" />
            </span>
          )}
        </div>

        {/* Top-right: actions menu (visible on hover) */}
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition">
          <EpisodeMenu
            ep={ep}
            onOpen={handleOpen}
            onPlay={ep.audio_path ? handlePlay : undefined}
            onDownload={handleDownload}
            onDelete={handleDelete}
            onProcess={handleProcess}
          >
            <button
              ref={menuTriggerRef}
              onClick={(e) => e.stopPropagation()}
              className="w-7 h-7 rounded-full bg-black/65 backdrop-blur-sm hover:bg-black/85 text-white flex items-center justify-center transition"
              title="More actions"
              aria-label="More actions"
            >
              <MoreVertical className="w-3.5 h-3.5" />
            </button>
          </EpisodeMenu>
        </div>

        {/* Bottom-left: status chips (moved off artwork top to reduce top-right clutter) */}
        <div className="absolute bottom-2 left-2 right-2 flex items-end justify-between gap-2">
          <div className="flex gap-1 items-center flex-wrap opacity-95">
            <StatusChips ep={ep} compact />
          </div>
          {isPlaying && (
            <span className="text-2xs bg-success text-white px-1.5 py-0.5 rounded-md font-medium shadow-sm shrink-0">
              Playing
            </span>
          )}
        </div>
      </div>

      {/* Text content */}
      <div className="p-3 space-y-1">
        <p className={`text-sm font-medium line-clamp-2 leading-snug ${ep.removed ? "text-muted-foreground" : ""}`}>{ep.title}</p>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {ep.pub_date && <span>{formatDate(ep.pub_date)}</span>}
          {ep.pub_date && ep.duration > 0 && <span className="w-0.5 h-0.5 rounded-full bg-muted-foreground/50" />}
          {ep.duration > 0 && <span>{formatDuration(ep.duration)}</span>}
        </div>
      </div>
    </div>
  );
}

export const EpisodeCard = memo(EpisodeCardInner);
