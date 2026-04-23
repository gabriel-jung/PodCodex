import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Segment } from "@/api/types";
import { getSegments as getTranscribeSegs } from "@/api/transcribe";
import { getCorrectSegments } from "@/api/correct";
import { getTranslateSegments } from "@/api/translate";
import { queryKeys } from "@/api/queryKeys";
import { formatDate, formatDuration, formatTime, errorMessage } from "@/lib/utils";
import { speakerColor } from "@/lib/speakerColor";
import { useAudioStore } from "@/stores";
import { getIndexedEpisode } from "@/api/episodes";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Play, Pause } from "lucide-react";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  audioPath: string;
  /** Indexed source: "transcript", "corrected", or a language code. */
  source: string;
  /** Matched chunk range — highlighted and scrolled into view. Omit to show all without highlighting. */
  start?: number;
  end?: number;
  episodeTitle: string;
  /** Show name + episode stem enable the episode-meta lookup that adds date/
   *  duration/speakers to the dialog subtitle. Model+chunking disambiguate
   *  multi-collection indices. */
  showName?: string;
  episodeStem?: string;
  model?: string;
  chunking?: string;
  onSeek: (time: number) => void;
  /** When provided, shows an "Open editor" button in the header. */
  onOpenEditor?: () => void;
}

function sourceToStep(source: string): "transcribe" | "correct" | "translate" {
  if (source === "transcript") return "transcribe";
  if (source === "corrected") return "correct";
  return "translate";
}

export default function SegmentContextDialog({
  open,
  onOpenChange,
  audioPath,
  source,
  start,
  end,
  episodeTitle,
  showName,
  episodeStem,
  model,
  chunking,
  onSeek,
  onOpenEditor,
}: Props) {
  const step = sourceToStep(source);
  const lang = step === "translate" ? source : "";
  const editorKey = step === "translate" ? `translate-${lang}` : step;

  const { data: segments, isLoading, isError, error } = useQuery({
    queryKey: queryKeys.stepSegments(editorKey, audioPath),
    queryFn: () => {
      if (step === "transcribe") return getTranscribeSegs(audioPath);
      if (step === "correct") return getCorrectSegments(audioPath);
      return getTranslateSegments(audioPath, lang);
    },
    enabled: open && !!audioPath,
    staleTime: 30_000,
  });

  // Require model+chunking on the key so two open dialogs on the same show/stem
  // but different collections don't share a cache entry and surface wrong meta.
  const { data: episodeMeta } = useQuery({
    queryKey: queryKeys.indexedEpisode(showName ?? "", episodeStem ?? "", model ?? "", chunking ?? ""),
    queryFn: () => getIndexedEpisode(showName!, episodeStem!, { model, chunking }),
    enabled: open && !!showName && !!episodeStem && !!model && !!chunking,
    staleTime: 5 * 60_000,
  });
  const metaDate = episodeMeta?.pub_date ? formatDate(episodeMeta.pub_date) : "";
  const metaDuration = episodeMeta?.duration ? formatDuration(episodeMeta.duration) : "";
  const metaNumber = episodeMeta?.episode_number != null ? `#${episodeMeta.episode_number}` : "";
  const metaSpeakers = episodeMeta?.speakers ?? [];

  const { matchIndices, firstMatchIdx } = useMemo(() => {
    const set = new Set<number>();
    let first = -1;
    if (segments && start != null && end != null) {
      segments.forEach((s, i) => {
        const ss = s.start ?? 0;
        const se = s.end ?? 0;
        if (se >= start && ss <= end) {
          set.add(i);
          if (first === -1) first = i;
        }
      });
    }
    return { matchIndices: set, firstMatchIdx: first };
  }, [segments, start, end]);

  const pauseAudio = useAudioStore((s) => s.pauseAudio);
  const isPlaying = useAudioStore((s) => s.isPlaying);
  const storeAudioPath = useAudioStore((s) => s.audioPath);
  const isPlayingThisFile = !!audioPath && storeAudioPath === audioPath;

  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  useEffect(() => {
    if (!isPlayingThisFile || !segments) {
      setActiveIdx(null);
      return;
    }
    const interval = setInterval(() => {
      const t = useAudioStore.getState().currentTime;
      if (!useAudioStore.getState().isPlaying) return;
      for (let i = segments.length - 1; i >= 0; i--) {
        const s = segments[i];
        if ((s.start ?? 0) <= t && t < (s.end ?? 0)) {
          setActiveIdx((prev) => (prev === i ? prev : i));
          return;
        }
      }
      setActiveIdx(null);
    }, 250);
    return () => clearInterval(interval);
  }, [isPlayingThisFile, segments]);

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const firstMatchRef = useRef<HTMLLIElement | null>(null);
  useEffect(() => {
    if (!open || firstMatchIdx === -1) return;
    const id = setTimeout(() => {
      const container = scrollContainerRef.current;
      const el = firstMatchRef.current;
      if (!container || !el) return;
      const containerRect = container.getBoundingClientRect();
      const elRect = el.getBoundingClientRect();
      container.scrollTop += elRect.top - containerRect.top - 8;
    }, 200);
    return () => clearTimeout(id);
  }, [open, firstMatchIdx]);

  const handlePlay = (seg: Segment) => {
    onSeek(seg.start ?? 0);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-3xl max-h-[80vh] flex flex-col gap-3">
        <DialogHeader>
          <div className="flex items-start justify-between gap-3 pr-8">
            <DialogTitle className="truncate leading-tight">{episodeTitle}</DialogTitle>
            {onOpenEditor && (
              <Button
                variant="outline"
                size="sm"
                className="text-xs h-7 shrink-0"
                onClick={() => { onOpenChange(false); onOpenEditor(); }}
              >
                Open editor
              </Button>
            )}
          </div>
          <DialogDescription className="flex items-center gap-2 text-xs flex-wrap">
            <span className="italic">{source}</span>
            {[metaDate, metaDuration, metaNumber]
              .filter(Boolean)
              .map((value, i) => (
                <span key={i} className="flex items-center gap-2">
                  <span className="text-muted-foreground/40">·</span>
                  <span>{value}</span>
                </span>
              ))}
          </DialogDescription>
          {metaSpeakers.length > 0 && (
            <div className="flex flex-wrap gap-1 pt-0.5">
              {metaSpeakers.map((sp) => (
                <span
                  key={sp}
                  className="rounded-sm bg-muted/60 px-1.5 py-0.5 text-2xs"
                  style={{ color: speakerColor(sp) }}
                >
                  {sp}
                </span>
              ))}
            </div>
          )}
        </DialogHeader>

        <div ref={scrollContainerRef} className="flex-1 overflow-y-auto -mx-6 px-6">
          {isLoading && (
            <div className="py-8 text-center text-muted-foreground text-sm">
              Loading transcript…
            </div>
          )}
          {isError && (
            <p className="text-destructive text-xs py-4">{errorMessage(error)}</p>
          )}
          {segments && segments.length === 0 && (
            <div className="py-8 text-center text-muted-foreground text-sm">
              No segments available for this source.
            </div>
          )}
          {segments && segments.length > 0 && (
            <ol className="space-y-1">
              {segments.map((seg, i) => {
                const isMatch = matchIndices.has(i);
                const speaker = seg.speaker || "";
                const isBreak = speaker === "[BREAK]";
                return (
                  <li
                    key={i}
                    ref={i === firstMatchIdx ? firstMatchRef : undefined}
                    className={`group grid grid-cols-[auto_auto_1fr] gap-3 px-2 py-1.5 rounded text-xs items-baseline ${
                      isMatch ? "bg-primary/10 border-l-2 border-primary" : ""
                    }`}
                  >
                    <button
                      onClick={() => (activeIdx === i && isPlaying) ? pauseAudio() : handlePlay(seg)}
                      className="font-mono tabular-nums text-muted-foreground/60 hover:text-foreground transition text-left flex items-center gap-1"
                      title={(activeIdx === i && isPlaying) ? "Pause" : `Play from ${formatTime(seg.start ?? 0, false)}`}
                    >
                      {activeIdx === i && isPlaying
                        ? <Pause className="w-2.5 h-2.5" />
                        : <Play className="w-2.5 h-2.5 opacity-0 group-hover:opacity-100 transition" />}
                      {formatTime(seg.start ?? 0, false)}
                    </button>
                    {isBreak ? (
                      <span className="italic text-muted-foreground/60">break</span>
                    ) : (
                      <span
                        className="font-medium truncate"
                        style={{ color: speakerColor(speaker) }}
                      >
                        {speaker}
                      </span>
                    )}
                    <p
                      className={`leading-relaxed whitespace-pre-wrap ${
                        isBreak ? "text-muted-foreground/40" : ""
                      }`}
                    >
                      {seg.text}
                    </p>
                  </li>
                );
              })}
            </ol>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
