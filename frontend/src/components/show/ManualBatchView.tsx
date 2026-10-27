/** Body of the batch dialog's manual flow, one episode at a time: prompts
 *  to copy, a box for each pasted reply, and progress across episodes.
 *  State lives in `useManualBatch`. */

import { Check, ChevronDown, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Episode } from "@/api/types";
import type { ManualBatch } from "./useManualBatch";
import { epKey } from "./useManualBatch";

export default function ManualBatchView({ m, episodes }: { m: ManualBatch; episodes: Episode[] }) {
  const ep = m.ep;
  if (!ep) return null;
  const ek = m.ek;
  const dur = ep.duration ? `${Math.round(ep.duration / 60)} min` : "";
  const hasPrompts = m.prompts.length > 0;
  const batch = m.prompts[m.batchIndex];
  const batchDone = m.batches[m.batchIndex] != null;
  const epDoneCount = m.prompts.filter((_, i) => m.batches[i] != null).length;
  const epAllDone = hasPrompts && epDoneCount === m.prompts.length;
  const totalEpsDone = episodes.filter((e) => m.applied.has(epKey(e))).length;

  return (
    <div className="space-y-3">
      {/* Episode navigation */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => m.goToEpisode(m.epIndex - 1)}
          disabled={m.epIndex === 0}
          className="text-muted-foreground hover:text-foreground disabled:opacity-30 transition"
          aria-label="Previous episode"
        >
          <ChevronDown className="w-3.5 h-3.5 rotate-90" />
        </button>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium truncate" title={ep.title}>{ep.title}</div>
          <div className="text-xs text-muted-foreground">
            Episode {m.epIndex + 1} of {m.episodeCount}
            {dur && ` - ${dur}`}
          </div>
        </div>
        <button
          onClick={() => m.goToEpisode(m.epIndex + 1)}
          disabled={m.epIndex >= m.episodeCount - 1}
          className="text-muted-foreground hover:text-foreground disabled:opacity-30 transition"
          aria-label="Next episode"
        >
          <ChevronDown className="w-3.5 h-3.5 -rotate-90" />
        </button>
      </div>

      {/* Overall progress (when multiple episodes) */}
      {m.episodeCount > 1 && (
        <div className="flex items-center gap-3">
          <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full rounded-full bg-primary transition-all"
              style={{ width: `${(totalEpsDone / m.episodeCount) * 100}%` }}
            />
          </div>
          <span className="text-xs text-muted-foreground shrink-0">{totalEpsDone}/{m.episodeCount} done</span>
        </div>
      )}

      {/* Batch count slider. Shown whenever the episode has a source, not
          only while prompts are loaded: a new count refetches them, and
          unmounting the slider mid-drag would drop the drag and the focus. */}
      {m.hasSource && (
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <label htmlFor={`batches-${ek}`} className="text-sm font-medium">Batches</label>
            <input
              id={`batches-${ek}`}
              type="range"
              min={1}
              max={10}
              value={m.batchCount}
              onChange={(e) => m.setBatchCount(Number(e.target.value))}
              className="flex-1 accent-primary"
            />
            <span className="text-sm tabular-nums w-4 text-center">{m.batchCount}</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Number of prompts to split the episode into. Large models like Claude Opus 4.6 or GPT-4o can handle hours-long episodes in one batch.
          </p>
        </div>
      )}

      {!m.hasSource && (
        <p className="text-xs text-muted-foreground">No transcript available for this episode.</p>
      )}
      {m.promptsLoading && (
        <p className="text-xs text-muted-foreground">Generating prompt...</p>
      )}
      {m.promptsError && <p className="text-destructive text-xs">{m.promptsError}</p>}

      {/* Batch navigation (if multiple) */}
      {m.prompts.length > 1 && (
        <div className="flex items-center gap-1.5">
          {m.prompts.map((_, i) => (
            <button
              key={i}
              onClick={() => m.goToBatch(i)}
              aria-label={`Batch ${i + 1}`}
              aria-current={i === m.batchIndex ? "step" : undefined}
              className={`w-6 h-6 rounded text-2xs font-medium transition ${
                i === m.batchIndex
                  ? "bg-primary text-primary-foreground"
                  : m.batches[i] != null
                    ? "bg-success/20 text-success border border-success/30"
                    : "bg-secondary text-muted-foreground border border-border"
              }`}
            >
              {i + 1}
            </button>
          ))}
          <span className="text-xs text-muted-foreground ml-1">{epDoneCount}/{m.prompts.length} batches</span>
        </div>
      )}

      {/* Prompt display + copy */}
      {batch && (
        <div className="border border-border rounded">
          <div className="flex items-center justify-between px-3 py-1.5 bg-secondary/50 border-b border-border">
            <span className="text-xs text-muted-foreground">
              {m.prompts.length > 1 ? `Batch ${m.batchIndex + 1} - ` : ""}{batch.segment_count} segments
              {batchDone && <span className="text-success ml-2">validated</span>}
            </span>
            <Button
              onClick={() => void m.copy(batch.prompt)}
              variant="ghost"
              size="sm"
              className="h-6 px-2"
              aria-label="Copy prompt"
            >
              {m.copied ? <Check className="w-3 h-3 text-success" /> : <Copy className="w-3 h-3" />}
            </Button>
          </div>
          <pre className="p-3 text-xs max-h-40 overflow-y-auto whitespace-pre-wrap leading-relaxed">
            {batch.prompt}
          </pre>
        </div>
      )}

      {/* Paste + validate */}
      {batch && !batchDone && (
        <div className="space-y-2">
          <textarea
            value={m.pasted}
            onChange={(e) => m.setPasted(e.target.value)}
            placeholder="Paste LLM JSON response here..."
            aria-label="LLM reply"
            className="input text-xs w-full resize-y"
            rows={4}
          />
          {m.parseError && <p className="text-destructive text-xs">{m.parseError}</p>}
          <Button
            onClick={() => void m.validate()}
            disabled={!m.pasted.trim() || m.applyingEp === ek}
            size="sm"
          >
            Validate
          </Button>
        </div>
      )}

      {/* Redo stays offered until the episode is applied: a failed apply
          must be fixable without starting the episode over. */}
      {batch && batchDone && !(epAllDone && m.applied.has(ek)) && m.applyingEp !== ek && (
        <div className="flex items-center gap-2 text-xs">
          <Check className="w-3.5 h-3.5 text-success" />
          <span className="text-success">Validated ({m.batches[m.batchIndex].length} segments)</span>
          <Button onClick={m.redo} variant="ghost" size="sm" className="h-6 text-xs text-muted-foreground">
            Redo
          </Button>
        </div>
      )}

      {m.applyingEp === ek && (
        <p className="text-xs text-muted-foreground">Applying corrections...</p>
      )}
      {m.applied.has(epKey(ep)) && (
        <div className="flex items-center gap-2 text-xs text-success">
          <Check className="w-3.5 h-3.5" />
          Applied
        </div>
      )}

      {m.error && <p className="text-destructive text-xs">{m.error}</p>}
    </div>
  );
}
