/** Lightweight read-only transcript viewer with click-to-seek and active segment tracking. */

import { useQuery } from "@tanstack/react-query";
import { useRef, useEffect } from "react";
import type { Segment } from "@/api/types";
import { useAudioStore } from "@/stores";
import { formatTime } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Pencil } from "lucide-react";

interface ReadOnlyTranscriptProps {
  audioPath: string;
  /** Fetch segments to display. Caller decides which version (raw, polished, etc.). */
  loadSegments: () => Promise<Segment[]>;
  /** Query key for TanStack Query caching. */
  queryKey: string[];
  /** Label shown in the toolbar (e.g. "polished", "raw transcript"). */
  sourceLabel?: string;
  /** Called when user clicks "Edit segments". If omitted, the button is hidden. */
  onEdit?: () => void;
}

export default function ReadOnlyTranscript({
  audioPath,
  loadSegments,
  queryKey,
  sourceLabel,
  onEdit,
}: ReadOnlyTranscriptProps) {
  const { seekTo, currentTime, audioPath: playingPath } = useAudioStore();
  const activeRef = useRef<HTMLDivElement>(null);

  const { data: segments } = useQuery({
    queryKey,
    queryFn: loadSegments,
  });

  const isPlaying = playingPath === audioPath;

  const activeIdx = isPlaying && segments
    ? segments.findIndex((s, i) => {
        const next = segments[i + 1];
        return currentTime >= s.start && (next ? currentTime < next.start : currentTime <= s.end);
      })
    : -1;

  useEffect(() => {
    if (activeIdx >= 0 && activeRef.current) {
      activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [activeIdx]);

  if (!segments) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        No transcript available. Transcribe the episode first.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="px-4 py-2 border-b border-border flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {segments.length} segments{sourceLabel ? ` (${sourceLabel})` : ""}
        </span>
        {onEdit && (
          <Button variant="outline" size="sm" className="text-xs h-7" onClick={onEdit}>
            <Pencil className="w-3 h-3 mr-1" /> Edit segments
          </Button>
        )}
      </div>

      {/* Segments */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-1">
          {segments.map((seg, i) => {
            if (seg.speaker === "[BREAK]") {
              return (
                <div key={i} className="py-2 flex items-center gap-2 text-muted-foreground/40">
                  <div className="flex-1 border-t border-border" />
                  <span className="text-[10px] uppercase">break</span>
                  <div className="flex-1 border-t border-border" />
                </div>
              );
            }
            const isActive = i === activeIdx;
            return (
              <div
                key={i}
                ref={isActive ? activeRef : undefined}
                className={`flex gap-3 py-1.5 px-2 rounded text-sm transition-colors ${
                  isActive ? "bg-accent/60" : "hover:bg-accent/30"
                }`}
              >
                <button
                  onClick={() => seekTo(audioPath, seg.start)}
                  className="text-[11px] text-muted-foreground hover:text-primary font-mono shrink-0 pt-0.5"
                  title="Jump to timestamp"
                >
                  {formatTime(seg.start, false)}
                </button>
                <span className="text-xs font-medium text-primary/70 shrink-0 pt-0.5 w-20 truncate">
                  {seg.speaker}
                </span>
                <span className="text-foreground/90">{seg.text}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
