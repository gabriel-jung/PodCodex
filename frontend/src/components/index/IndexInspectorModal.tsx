import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Play } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { EmptyState } from "@/components/ui/empty-state";
import {
  getIndexInspect,
  type InspectChunk,
  type InspectChunkSpeakerTurn,
  type InspectSummary,
} from "@/api/search";
import { queryKeys } from "@/api/queryKeys";
import { speakerColor } from "@/lib/speakerColor";
import { formatTime, errorMessage } from "@/lib/utils";
import { useAudioStore } from "@/stores";

interface Props {
  open: boolean;
  onClose: () => void;
  audioPath: string;
  show: string;
  model: string;
  modelLabel?: string;
  chunking: string;
}

export default function IndexInspectorModal({
  open, onClose, audioPath, show, model, modelLabel, chunking,
}: Props) {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: queryKeys.indexInspect(audioPath, show, model, chunking),
    queryFn: () => getIndexInspect(audioPath, show, model, chunking),
    enabled: open,
    staleTime: 30_000,
  });

  const source = data?.chunks[0]?.source;

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="sm:max-w-3xl max-h-[80vh] flex flex-col gap-3">
        <DialogHeader>
          <DialogTitle className="truncate leading-tight">
            {modelLabel ?? model} · {chunking}
          </DialogTitle>
          <DialogDescription className="flex items-center gap-2 text-xs flex-wrap">
            {source && <span className="italic">{source}</span>}
            {data && (
              <>
                {source && <span className="text-muted-foreground/40">·</span>}
                <span>{data.summary.n_chunks} chunks</span>
                <span className="text-muted-foreground/40">·</span>
                <span>dim {data.summary.dim}</span>
              </>
            )}
          </DialogDescription>
          {data && <Warnings summary={data.summary} />}
        </DialogHeader>

        <div className="flex-1 overflow-y-auto -mx-6 px-6">
          {isLoading && (
            <div className="py-8 text-center text-muted-foreground text-sm flex items-center justify-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading chunks…
            </div>
          )}
          {isError && (
            <p className="text-destructive text-xs py-4">{errorMessage(error)}</p>
          )}
          {data && data.chunks.length === 0 && (
            <EmptyState
              title="No chunks indexed"
              description="This combination has no chunks for the current episode."
            />
          )}
          {data && data.chunks.length > 0 && (
            <ol className="space-y-3">
              {data.chunks.map((c) => (
                <ChunkBlock
                  key={c.chunk_index}
                  chunk={c}
                  audioPath={audioPath}
                  zeroThreshold={data.summary.zero_warn_threshold}
                />
              ))}
            </ol>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Warnings({ summary }: { summary: InspectSummary }) {
  const items: { label: string; tone: "destructive" | "warning" }[] = [];
  if (summary.n_dead_chunks > 0)
    items.push({ label: `${summary.n_dead_chunks} dead chunks (‖v‖ ≈ 0)`, tone: "destructive" });
  if (summary.n_collapsed_chunks > 0)
    items.push({ label: `${summary.n_collapsed_chunks} collapsed (>50% zeros)`, tone: "destructive" });
  if (summary.n_with_zeros > 0 && summary.n_collapsed_chunks === 0) {
    const pct = Math.round(summary.zero_warn_threshold * 100);
    items.push({
      label: `${summary.n_with_zeros} chunks with >${pct}% zeros`,
      tone: "warning",
    });
  }
  if (items.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1 pt-1">
      {items.map((it, i) => (
        <span
          key={i}
          className={`rounded-sm px-1.5 py-0.5 text-2xs ${
            it.tone === "destructive"
              ? "bg-destructive/15 text-destructive"
              : "bg-warning/15 text-warning"
          }`}
        >
          ⚠ {it.label}
        </span>
      ))}
    </div>
  );
}

function ChunkBlock({
  chunk, audioPath, zeroThreshold,
}: {
  chunk: InspectChunk;
  audioPath: string;
  zeroThreshold: number;
}) {
  const seek = useAudioStore((s) => s.seekTo);
  const turns = chunk.speakers ?? [];
  const hasZeros = chunk.vector_zero_frac > zeroThreshold;
  const zeroPct = chunk.vector_zero_frac * 100;

  return (
    <li className="rounded border border-border/60 bg-muted/20">
      <div className="flex items-center gap-3 px-3 py-2 border-b border-border/40 text-xs">
        <span className="font-mono text-muted-foreground tabular-nums shrink-0">
          #{chunk.chunk_index}
        </span>
        <span className="font-mono tabular-nums shrink-0">
          {formatTime(chunk.start, false)} → {formatTime(chunk.end, false)}
        </span>
        <div className="flex-1 min-w-0 flex flex-wrap items-baseline font-mono tabular-nums text-muted-foreground">
          <Stat label="‖v‖" value={chunk.vector_norm} first />
          <Stat label="min(v)" value={chunk.vector_min} />
          <Stat label="max(v)" value={chunk.vector_max} />
          <Stat
            label={
              <span className="inline-block leading-none border-t border-current pt-px">v</span>
            }
            value={chunk.vector_mean}
          />
          <Stat label="σ(v)" value={chunk.vector_std} />
        </div>
        {hasZeros && (
          <span
            className={`text-2xs px-1.5 py-0.5 rounded shrink-0 ${
              chunk.vector_zero_frac > 0.5
                ? "bg-destructive/15 text-destructive"
                : "bg-warning/15 text-warning"
            }`}
            title={`${zeroPct.toFixed(1)}% of dimensions are exactly zero`}
          >
            zeros {zeroPct.toFixed(0)}%
          </span>
        )}
        <button
          type="button"
          onClick={() => seek(audioPath, chunk.start)}
          className="shrink-0 text-muted-foreground/60 hover:text-foreground transition"
          aria-label={`Play from ${formatTime(chunk.start, false)}`}
          title={`Play from ${formatTime(chunk.start, false)}`}
        >
          <Play className="w-3.5 h-3.5" aria-hidden="true" />
        </button>
      </div>
      {turns.length > 0 ? (
        <ol className="px-2 py-1.5 space-y-1">
          {turns.map((t, i) => (
            <TurnRow key={i} turn={t} audioPath={audioPath} onSeek={seek} />
          ))}
        </ol>
      ) : (
        <p className="px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap">
          {chunk.text}
        </p>
      )}
    </li>
  );
}

function Stat({ label, value, first }: { label: ReactNode; value: number; first?: boolean }) {
  // Pick precision based on magnitude so "μ = 0.0001" stays readable while
  // "‖v‖ = 12.4" doesn't waste digits.
  const abs = Math.abs(value);
  const text =
    abs === 0 ? "0"
    : abs >= 1 ? value.toFixed(2)
    : abs >= 0.01 ? value.toFixed(3)
    : value.toExponential(1);
  return (
    <span className="inline-flex items-baseline">
      {!first && <span className="px-2 text-muted-foreground/40">·</span>}
      <span>{label}</span>
      <span className="px-1 text-muted-foreground/60">=</span>
      <span className="text-foreground">{text}</span>
    </span>
  );
}

function TurnRow({
  turn, audioPath, onSeek,
}: {
  turn: InspectChunkSpeakerTurn;
  audioPath: string;
  onSeek: (path: string, time: number) => void;
}) {
  return (
    <li className="group grid grid-cols-[auto_auto_1fr] gap-3 px-2 py-1.5 rounded text-xs items-baseline">
      <button
        onClick={() => onSeek(audioPath, turn.start)}
        className="font-mono tabular-nums text-muted-foreground/60 hover:text-foreground transition text-left flex items-center gap-1"
        title={`Play from ${formatTime(turn.start, false)}`}
      >
        <Play className="w-2.5 h-2.5 opacity-0 group-hover:opacity-100 transition" />
        {formatTime(turn.start, false)}
      </button>
      <span
        className="font-medium truncate"
        style={{ color: speakerColor(turn.speaker) }}
      >
        {turn.speaker}
      </span>
      <p className="leading-relaxed whitespace-pre-wrap">{turn.text}</p>
    </li>
  );
}
