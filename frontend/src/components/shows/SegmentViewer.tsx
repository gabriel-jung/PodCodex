import { useState, useMemo } from "react";
import type { Segment } from "@/api/types";
import { audioClipUrl } from "@/api/client";
import { Button } from "@/components/ui/button";

const PAGE_SIZES = [10, 20, 50];

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

interface SegmentViewerProps {
  segments: Segment[];
  audioPath?: string;
}

export default function SegmentViewer({ segments, audioPath }: SegmentViewerProps) {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(20);
  const [speakerFilter, setSpeakerFilter] = useState<string>("");
  const [playingIdx, setPlayingIdx] = useState<number | null>(null);

  const speakers = useMemo(() => {
    const set = new Set(segments.map((s) => s.speaker).filter(Boolean));
    return Array.from(set).sort();
  }, [segments]);

  const filtered = useMemo(() => {
    if (!speakerFilter) return segments.map((s, i) => ({ ...s, _idx: i }));
    return segments
      .map((s, i) => ({ ...s, _idx: i }))
      .filter((s) => s.speaker === speakerFilter);
  }, [segments, speakerFilter]);

  const totalPages = Math.ceil(filtered.length / pageSize);
  const pageItems = filtered.slice(page * pageSize, (page + 1) * pageSize);

  return (
    <div className="p-6 space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-4 text-sm">
        <span className="text-muted-foreground">
          {segments.length} segments
          {speakerFilter && ` (${filtered.length} shown)`}
        </span>

        {speakers.length > 1 && (
          <select
            value={speakerFilter}
            onChange={(e) => {
              setSpeakerFilter(e.target.value);
              setPage(0);
            }}
            className="bg-secondary text-secondary-foreground text-xs rounded px-2 py-1
                       border border-border"
          >
            <option value="">All speakers</option>
            {speakers.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        )}

        <select
          value={pageSize}
          onChange={(e) => {
            setPageSize(Number(e.target.value));
            setPage(0);
          }}
          className="bg-secondary text-secondary-foreground text-xs rounded px-2 py-1
                     border border-border"
        >
          {PAGE_SIZES.map((n) => (
            <option key={n} value={n}>
              {n} / page
            </option>
          ))}
        </select>
      </div>

      {/* Segments */}
      <div className="space-y-2">
        {pageItems.map((seg) => (
          <SegmentRow
            key={seg._idx}
            segment={seg}
            audioPath={audioPath}
            isPlaying={playingIdx === seg._idx}
            onPlay={() => setPlayingIdx(seg._idx)}
            onStop={() => setPlayingIdx(null)}
          />
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 text-sm">
          <Button
            onClick={() => setPage(Math.max(0, page - 1))}
            disabled={page === 0}
            variant="outline"
            size="sm"
          >
            Prev
          </Button>
          <span className="text-muted-foreground px-3">
            {page + 1} / {totalPages}
          </span>
          <Button
            onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
            disabled={page >= totalPages - 1}
            variant="outline"
            size="sm"
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
}

function SegmentRow({
  segment,
  audioPath,
  isPlaying,
  onPlay,
  onStop,
}: {
  segment: Segment;
  audioPath?: string;
  isPlaying: boolean;
  onPlay: () => void;
  onStop: () => void;
}) {
  const isBreak = segment.speaker === "[BREAK]";

  if (isBreak) {
    return (
      <div className="px-4 py-2 text-xs text-muted-foreground text-center border-t border-b border-border/50">
        — break ({formatTime(segment.start)} → {formatTime(segment.end)}) —
      </div>
    );
  }

  return (
    <div className="bg-secondary/40 rounded-lg px-4 py-3 space-y-1">
      <div className="flex items-center gap-3 text-xs">
        <span className="text-primary font-medium min-w-[80px]">
          {segment.speaker || "Unknown"}
        </span>
        <span className="text-muted-foreground">
          {formatTime(segment.start)} → {formatTime(segment.end)}
        </span>

        {audioPath && (
          <button
            onClick={() => {
              if (isPlaying) {
                document.querySelectorAll("audio.clip-player").forEach((el) => {
                  (el as HTMLAudioElement).pause();
                });
                onStop();
              } else {
                onPlay();
              }
            }}
            className="text-muted-foreground hover:text-foreground transition"
          >
            {isPlaying ? "⏹" : "▶"}
          </button>
        )}
      </div>

      <p className="text-sm text-foreground/80 leading-relaxed">{segment.text}</p>

      {isPlaying && audioPath && (
        <audio
          className="clip-player"
          src={audioClipUrl(audioPath, segment.start, segment.end)}
          autoPlay
          onEnded={onStop}
          onError={onStop}
        />
      )}
    </div>
  );
}
