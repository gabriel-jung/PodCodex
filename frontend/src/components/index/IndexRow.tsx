import { useState } from "react";
import { Trash2 } from "lucide-react";
import InlineConfirm from "@/components/common/InlineConfirm";

export interface IndexRowProps {
  model: string;
  chunker: string;
  source?: string;
  chunkCount: number;
  onInspect: () => void;
  /** Optional delete affordance — bundling the handler + confirm message
   *  enforces "if you can delete, you must explain what's being deleted". */
  deletion?: {
    onConfirm: () => void;
    message: string;
  };
}

export default function IndexRow({
  model,
  chunker,
  source,
  chunkCount,
  onInspect,
  deletion,
}: IndexRowProps) {
  const [confirming, setConfirming] = useState(false);

  if (confirming && deletion) {
    return (
      <div className="px-4 py-2 border-l-2 border-transparent">
        <InlineConfirm
          message={deletion.message}
          onConfirm={() => {
            setConfirming(false);
            deletion.onConfirm();
          }}
          onCancel={() => setConfirming(false)}
        />
      </div>
    );
  }

  return (
    <div className="px-4 py-2 flex items-center gap-2 group/row hover:bg-accent/40 transition border-l-2 border-transparent">
      <span className="shrink-0 w-1.5 h-1.5 rounded-full bg-info" />
      <button
        type="button"
        onClick={onInspect}
        className="flex-1 truncate text-xs text-left hover:underline cursor-pointer"
        title="Inspect chunks and vectors"
      >
        <span className="text-foreground">
          {model} · {chunker}
        </span>
        {source && (
          <span className="text-muted-foreground"> · from {source}</span>
        )}
      </button>
      <span className="shrink-0 font-mono text-2xs text-muted-foreground/60 tabular-nums">
        {chunkCount} chunks
      </span>
      {deletion && (
        <button
          onClick={() => setConfirming(true)}
          className="shrink-0 text-muted-foreground/40 hover:text-destructive p-0.5 opacity-0 group-hover/row:opacity-100 transition"
          title="Remove from this collection"
        >
          <Trash2 className="w-3 h-3" />
        </button>
      )}
    </div>
  );
}
