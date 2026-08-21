import { memo, useRef } from "react";
import type { Episode } from "@/api/types";
import { Play, Pause, Download, Trash2, Captions, CloudOff } from "lucide-react";
import { formatDuration, formatDate } from "@/lib/utils";
import { useLayoutStore } from "@/stores";
import { StatusChips } from "./StatusChips";
import { SpeakerNames } from "./SpeakerNames";
import { EpisodeMenu } from "./EpisodeMenu";

export interface EpisodeRowProps {
  ep: Episode;
  index: number;
  selected: boolean;
  onToggle: (id: string, index: number, shiftKey: boolean) => void;
  onOpen: (stem: string) => void;
  onPlay: (ep: Episode) => void;
  onDownload?: (id: string) => void;
  onDelete: (ep: Episode) => void;
  /** Delete the whole episode, not just its audio. */
  onDeleteEpisode?: (ep: Episode) => void;
  downloading?: boolean;
  /** True only while audio is actively playing (paused → false). */
  isPlaying: boolean;
  /** True when this episode is the loaded track, regardless of play/pause. */
  isCurrent: boolean;
  /** Speakers of the episode's canonical transcript, most-airtime first. */
  speakers?: string[];
}

function EpisodeRowInner({ ep, index, selected, onToggle, onOpen, onPlay, onDownload, onDelete, onDeleteEpisode, downloading, isPlaying, isCurrent, speakers }: EpisodeRowProps) {
  const compact = useLayoutStore((s) => s.compact);
  const shiftRef = useRef(false);

  const handleOpen = () => onOpen(ep.stem || ep.id);
  const handlePlay = () => onPlay(ep);
  const handleDownload = onDownload ? () => onDownload(ep.id) : undefined;
  const handleDelete = () => onDelete(ep);
  const handleDeleteEpisode = onDeleteEpisode ? () => onDeleteEpisode(ep) : undefined;

  return (
    <div className="@container flex items-center gap-3 px-6 py-2 hover:bg-accent/50 transition group">
      <input type="checkbox" checked={selected} onMouseDown={(e) => { shiftRef.current = e.shiftKey; }} onChange={() => onToggle(ep.id, index, shiftRef.current)} className="accent-primary cursor-pointer shrink-0" />
      <div className="w-10 h-10 shrink-0">
        {ep.artwork_url && (
          <img src={ep.artwork_url} alt={ep.title} className="w-10 h-10 object-cover rounded" loading="lazy" />
        )}
      </div>
      <span className="text-xs text-muted-foreground w-8 text-right shrink-0 tabular-nums">
        {ep.episode_number != null ? `#${ep.episode_number}` : ""}
      </span>
      <div className="min-w-0 flex-[3_1_0%] flex flex-col gap-0.5">
        <button
          onClick={handleOpen}
          className={`text-left text-sm hover:text-primary cursor-pointer flex items-center gap-1.5 max-w-full ${ep.removed ? "text-muted-foreground" : "text-foreground"}`}
        >
          {ep.removed && (
            <span title="No longer in the live feed, kept locally" className="shrink-0">
              <CloudOff className="w-3 h-3 text-muted-foreground" />
            </span>
          )}
          <span className="truncate" title={ep.title}>{ep.title}</span>
        </button>
        {!compact && <StatusChips ep={ep} />}
      </div>
      {/* Metadata cluster pinned right: speakers · date · duration. Date and
          duration are fixed columns; speakers takes a quarter of the leftover
          so it shrinks with the window instead of squeezing the title, and it
          caps at the old 18rem on wide rows (the title absorbs the rest).
          Gated on the *row's* width, not the viewport: the sidebar is
          `pl-14`/`pl-48`, so a viewport breakpoint would show it on rows too
          narrow to afford it. Below 56rem it hides, which keeps the title
          wider than it is at the app's 720px minimum width. */}
      <div
        className="hidden @[56rem]:flex items-center gap-1 flex-[1_1_0%] min-w-0 max-w-72 text-xs text-muted-foreground"
        title={speakers?.join(", ")}
      >
        {speakers && speakers.length > 0 && <SpeakerNames names={speakers} />}
      </div>
      <span className="w-24 shrink-0 text-right text-xs text-muted-foreground tabular-nums">
        {ep.pub_date ? formatDate(ep.pub_date) : ""}
      </span>
      <span className="w-14 shrink-0 text-right text-xs text-muted-foreground tabular-nums">
        {ep.duration > 0 ? formatDuration(ep.duration) : ""}
      </span>
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
        {/* Card view has always had this menu; the row only had the audio
            trash, which left subtitle-only episodes with no actions at all. */}
        <span className="w-5 flex justify-center">
          <EpisodeMenu
            ep={ep}
            onOpen={handleOpen}
            onPlay={handlePlay}
            onDownload={handleDownload}
            onDelete={handleDelete}
            onDeleteEpisode={handleDeleteEpisode}
          />
        </span>
      </div>
    </div>
  );
}

export const EpisodeRow = memo(EpisodeRowInner);
