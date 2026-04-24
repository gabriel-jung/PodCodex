import { memo, useRef } from "react";
import type { Episode } from "@/api/types";
import { Play, Download, MoreVertical, Captions, CloudOff } from "lucide-react";
import { formatDuration, formatDate } from "@/lib/utils";
import { StatusChips } from "./StatusChips";
import { EpisodeMenu, type PipelineStep } from "./EpisodeMenu";

export interface EpisodeRowProps {
  ep: Episode;
  index: number;
  selected: boolean;
  onToggle: (id: string, index: number, shiftKey: boolean) => void;
  onOpen: (stem: string) => void;
  onPlay: (ep: Episode) => void;
  onDownload?: (id: string) => void;
  onDelete: (ep: Episode) => void;
  onProcess?: (step: PipelineStep, ep: Episode) => void;
  downloading?: boolean;
  isPlaying: boolean;
}

function EpisodeRowInner({ ep, index, selected, onToggle, onOpen, onPlay, onDownload, onDelete, onProcess, downloading, isPlaying }: EpisodeRowProps) {
  const shiftRef = useRef(false);
  const menuTriggerRef = useRef<HTMLButtonElement>(null);

  const onContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    menuTriggerRef.current?.click();
  };

  const handleOpen = () => onOpen(ep.stem || ep.id);
  const handlePlay = () => onPlay(ep);
  const handleDownload = onDownload ? () => onDownload(ep.id) : undefined;
  const handleDelete = () => onDelete(ep);
  const handleProcess = onProcess ? (step: PipelineStep) => onProcess(step, ep) : undefined;

  return (
    <div onContextMenu={onContextMenu} className="flex items-center gap-3 px-6 py-3 hover:bg-accent/50 transition group">
      <input type="checkbox" checked={selected} onMouseDown={(e) => { shiftRef.current = e.shiftKey; }} onChange={() => onToggle(ep.id, index, shiftRef.current)} className="accent-primary cursor-pointer shrink-0" />
      {ep.artwork_url && (
        <img src={ep.artwork_url} alt={ep.title} className="w-8 h-6 object-cover rounded shrink-0" loading="lazy" />
      )}
      {ep.episode_number != null && (
        <span className="text-xs text-muted-foreground w-8 text-right shrink-0">#{ep.episode_number}</span>
      )}
      <button
        onClick={handleOpen}
        className={`flex-1 text-left text-sm truncate hover:text-primary cursor-pointer flex items-center gap-1.5 ${ep.removed ? "text-muted-foreground" : "text-foreground"}`}
      >
        {ep.removed && (
          <span title="No longer in the live feed — kept locally" className="shrink-0">
            <CloudOff className="w-3 h-3 text-muted-foreground" />
          </span>
        )}
        <span className="truncate">{ep.title}</span>
      </button>
      <StatusChips ep={ep} compact />
      <span className="text-xs text-muted-foreground w-20 text-right shrink-0">{formatDate(ep.pub_date)}</span>
      <span className="text-xs text-muted-foreground w-12 text-right shrink-0">{formatDuration(ep.duration)}</span>
      <div className="w-24 flex justify-end gap-2.5 shrink-0">
        {ep.has_subtitles && (
          <span title="Subtitles cached" aria-label="Subtitles cached" className="text-muted-foreground/70">
            <Captions className="w-3.5 h-3.5" />
          </span>
        )}
        {ep.audio_path ? (
          <button onClick={handlePlay} title="Play" aria-label="Play" className={`transition ${isPlaying ? "text-success" : "text-muted-foreground hover:text-foreground"}`}>
            <Play className="w-3.5 h-3.5" />
          </button>
        ) : handleDownload && (
          <button onClick={handleDownload} disabled={downloading} title="Download audio" aria-label="Download audio" className="text-muted-foreground hover:text-foreground transition disabled:opacity-50">
            <Download className="w-3.5 h-3.5" />
          </button>
        )}
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
            className="text-muted-foreground hover:text-foreground transition opacity-0 group-hover:opacity-100 data-[state=open]:opacity-100"
            title="More actions"
            aria-label="More actions"
          >
            <MoreVertical className="w-3.5 h-3.5" />
          </button>
        </EpisodeMenu>
      </div>
    </div>
  );
}

export const EpisodeRow = memo(EpisodeRowInner);
