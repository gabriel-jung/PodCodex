import type { SearchResult } from "@/api/types";
import { useAudioStore } from "@/stores";
import { formatTime } from "@/lib/utils";

interface SearchResultCardProps {
  result: SearchResult;
  audioPath?: string;
}

export default function SearchResultCard({ result, audioPath }: SearchResultCardProps) {
  const { seekTo } = useAudioStore();

  const scoreColor =
    result.score >= 0.8
      ? "bg-green-900/40 text-green-400"
      : result.score >= 0.5
        ? "bg-yellow-900/40 text-yellow-400"
        : "bg-muted text-muted-foreground";

  return (
    <div className="px-4 py-3 rounded-lg bg-secondary border border-border space-y-2">
      <div className="flex items-center gap-2 text-xs">
        <span className={`px-2 py-0.5 rounded-full font-mono ${scoreColor}`}>
          {result.score.toFixed(3)}
        </span>
        <span className="text-muted-foreground">{result.episode}</span>
        <span className="text-muted-foreground">
          {formatTime(result.start, false)} - {formatTime(result.end, false)}
        </span>
        {result.speaker && (
          <span className="font-medium">{result.speaker}</span>
        )}
        {audioPath && (
          <button
            onClick={() => seekTo(audioPath, result.start)}
            className="ml-auto text-xs px-2 py-0.5 rounded bg-accent hover:bg-accent/80 transition"
          >
            Seek
          </button>
        )}
      </div>
      <p className="text-sm leading-relaxed whitespace-pre-wrap">
        {result.text}
      </p>
    </div>
  );
}
