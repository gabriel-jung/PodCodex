import { memo, useState } from "react";
import type { SearchResult } from "@/api/types";
import { useAudioStore } from "@/stores";
import { formatDate, formatTime, mergeDisplayTurns } from "@/lib/utils";
import { speakerColor } from "@/lib/speakerColor";
import { highlightText } from "@/lib/highlight";
import { Play } from "lucide-react";
import SegmentContextDialog from "./SegmentContextDialog";

interface ShowContext {
  name?: string;
  folder?: string;
  artwork?: string;
  model?: string;
  chunking?: string;
}

interface SearchResultCardProps {
  result: SearchResult;
  show?: ShowContext;
  query?: string;
}

function SearchResultCardInner({ result, show, query = "" }: SearchResultCardProps) {
  const playEpisode = useAudioStore((s) => s.playEpisode);
  const [contextOpen, setContextOpen] = useState(false);

  const scoreEmphasis =
    result.score >= 0.8
      ? "text-success"
      : result.score >= 0.5
        ? "text-warning"
        : "text-muted-foreground/60";

  // Badge for /exact hits that aren't literal: accent-folded ("cafe" ≈ "café")
  // or fuzzy near-typo. Signals to the user that this isn't a verbatim hit.
  const matchBadge = result.fuzzy_match
    ? { label: "〜 near-typo", tone: "bg-orange-500/15 text-orange-700 dark:text-orange-300 border-orange-500/30" }
    : result.accent_match
      ? { label: "≈ variant", tone: "bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30" }
      : null;

  const path = result.audio_path;

  const playAt = (time: number) => {
    if (!path) return;
    playEpisode(path, time, {
      title: result.episode,
      artwork: show?.artwork,
      showName: show?.name,
      folder: show?.folder,
      stem: result.episode_stem || undefined,
    });
  };

  const rawTurns =
    result.speakers && result.speakers.length > 0
      ? result.speakers
      : result.speaker
        ? [{ speaker: result.speaker, text: result.text, start: result.start, end: result.end }]
        : [];
  const turns = mergeDisplayTurns(rawTurns);

  return (
    <>
      <div className="group px-4 py-3 rounded-lg bg-secondary border border-border space-y-2">
        <div className="flex items-start gap-2 text-xs">
          {path && (
            <button
              onClick={() => playAt(result.start)}
              className="shrink-0 mt-0.5 w-6 h-6 rounded-full flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-accent transition"
              title={`Play from ${formatTime(result.start, false)}`}
            >
              <Play className="w-3 h-3 fill-current" />
            </button>
          )}
          <div className="flex-1 min-w-0 space-y-0.5">
            <div className="text-sm font-medium truncate flex items-center gap-1.5">
              <span className="truncate">{result.episode}</span>
              {matchBadge && (
                <span
                  className={`shrink-0 inline-flex items-center text-[10px] border rounded-full px-1.5 py-0 font-normal ${matchBadge.tone}`}
                >
                  {matchBadge.label}
                </span>
              )}
            </div>
            <div className="font-mono text-muted-foreground flex items-center gap-2">
              <span>{formatTime(result.start, false)} – {formatTime(result.end, false)}</span>
              {result.pub_date && (
                <>
                  <span className="text-muted-foreground/40">·</span>
                  <span className="font-sans">{formatDate(result.pub_date)}</span>
                </>
              )}
            </div>
          </div>
          <span className={`shrink-0 italic font-mono text-2xs tabular-nums ${scoreEmphasis}`}>
            {Math.round(result.score * 100)}%
          </span>
        </div>
        <button
          onClick={() => path && setContextOpen(true)}
          disabled={!path}
          className="text-left w-full disabled:cursor-default"
          title={path ? "Open in context" : undefined}
        >
          {turns.length > 0 ? (
            <ol className="space-y-1 text-sm">
              {turns.map((turn, i) => (
                <li
                  key={i}
                  className="grid grid-cols-[auto_1fr] gap-x-3 leading-relaxed items-baseline"
                >
                  <span
                    className="font-medium text-xs shrink-0"
                    style={{ color: speakerColor(turn.speaker) }}
                  >
                    {turn.speaker}
                  </span>
                  <span className="whitespace-pre-wrap">{highlightText(turn.text, result.match_text || query)}</span>
                </li>
              ))}
            </ol>
          ) : (
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{highlightText(result.text, result.match_text || query)}</p>
          )}
        </button>
      </div>

      {path && (
        <SegmentContextDialog
          open={contextOpen}
          onOpenChange={setContextOpen}
          audioPath={path}
          source={result.source}
          start={result.start}
          end={result.end}
          episodeTitle={result.episode}
          showName={show?.name}
          episodeStem={result.episode_stem || undefined}
          model={show?.model}
          chunking={show?.chunking}
          onSeek={playAt}
        />
      )}
    </>
  );
}

const SearchResultCard = memo(SearchResultCardInner);
export default SearchResultCard;
