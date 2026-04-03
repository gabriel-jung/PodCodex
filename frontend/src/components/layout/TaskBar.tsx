/** Global task progress bar — pinned above the audio bar like a persistent status strip. */

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTaskStore } from "@/stores";
import { cancelTask } from "@/api/client";
import { useProgress, type TaskProgress } from "@/hooks/useProgress";
import { Button } from "@/components/ui/button";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Circle,
  Download,
  Loader2,
  Terminal,
  X,
  XCircle,
} from "lucide-react";
import { deriveEpisodeStatuses, type EpStatus } from "@/lib/batchUtils";

function DownloadStrip() {
  const { downloadTaskId, downloadFolder, setDownloadTask } = useTaskStore();
  const progress = useProgress(downloadTaskId);
  const queryClient = useQueryClient();

  if (!downloadTaskId) return null;

  const pct = progress ? Math.round(progress.progress * 100) : 0;
  const msg = progress?.message || "Starting...";
  const isDone = progress?.status === "completed";
  const isFailed = progress?.status === "failed";
  const isCancelled = progress?.status === "cancelled";
  const isFinished = isDone || isFailed || isCancelled;

  // Derive per-status counts from result (array of {stem, status})
  const results = (isFinished && Array.isArray(progress?.result) ? progress.result : []) as { stem: string; status: string }[];
  const total = results.length || 1;
  const failedCount = results.filter(r => r.status === "failed").length;
  const successCount = results.filter(r => r.status === "downloaded" || r.status === "exists").length;
  const successPct = Math.round((successCount / total) * 100);
  const failedPct = Math.round((failedCount / total) * 100);
  const hasFailures = failedCount > 0;

  const dismiss = () => {
    setDownloadTask(null);
    if (downloadFolder) {
      queryClient.refetchQueries({ queryKey: ["episodes", downloadFolder] });
    }
  };

  if (isFinished && isDone && !hasFailures) {
    setTimeout(dismiss, 2000);
  }

  return (
    <div className="px-4 py-2 flex items-center gap-3 text-xs">
      <Download className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
      <div className="flex-1 min-w-0 flex items-center gap-2">
        <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden flex">
          {isFinished && results.length > 0 ? (
            <>
              <div
                className="h-full bg-green-500 transition-all duration-300"
                style={{ width: `${successPct}%` }}
              />
              {failedPct > 0 && (
                <div
                  className="h-full bg-destructive transition-all duration-300"
                  style={{ width: `${failedPct}%` }}
                />
              )}
            </>
          ) : (
            <div
              className={`h-full transition-all duration-300 ${
                isFailed ? "bg-destructive" : isCancelled ? "bg-yellow-500" : "bg-primary"
              }`}
              style={{ width: `${pct}%` }}
            />
          )}
        </div>
        <span className="text-muted-foreground shrink-0 w-8 text-right">{pct}%</span>
      </div>
      <span className="text-muted-foreground truncate max-w-[300px]">{msg}</span>
      {!isFinished ? (
        <Button
          onClick={() => cancelTask(downloadTaskId).catch(() => dismiss())}
          variant="ghost"
          size="sm"
          className="h-5 w-5 p-0 text-muted-foreground hover:text-foreground"
          title="Cancel download"
        >
          <X className="w-3 h-3" />
        </Button>
      ) : (
        <Button
          onClick={dismiss}
          variant="ghost"
          size="sm"
          className="h-5 w-5 p-0 text-muted-foreground hover:text-foreground"
          title="Dismiss"
        >
          <X className="w-3 h-3" />
        </Button>
      )}
    </div>
  );
}

const STATUS_ICON: Record<EpStatus, typeof Circle> = {
  pending: Circle,
  running: Loader2,
  done: CheckCircle2,
  failed: XCircle,
};

const STATUS_COLOR: Record<EpStatus, string> = {
  pending: "text-muted-foreground",
  running: "text-primary animate-spin",
  done: "text-green-500",
  failed: "text-destructive",
};

/* ── Batch results summary (shown after completion) ── */

interface BatchResult {
  total: number;
  completed: number;
  failed: number;
  skipped: number;
  errors: { episode: string; error: string }[];
}

function BatchResultSummary({ result, onDismiss }: { result: BatchResult; onDismiss: () => void }) {
  const [showErrors, setShowErrors] = useState(false);
  const hasErrors = result.errors.length > 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onDismiss}>
      <div className="bg-background border border-border rounded-lg shadow-xl w-full max-w-sm mx-4" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 px-5 py-4 border-b border-border">
          {hasErrors ? (
            <AlertTriangle className="w-4 h-4 text-destructive" />
          ) : (
            <CheckCircle2 className="w-4 h-4 text-green-500" />
          )}
          <h3 className="text-base font-semibold">Batch Complete</h3>
          <div className="flex-1" />
          <button onClick={onDismiss} className="text-muted-foreground hover:text-foreground text-lg">
            ×
          </button>
        </div>

        <div className="px-5 py-4 space-y-2">
          <div className="grid grid-cols-3 gap-3 text-center">
            <div>
              <p className="text-2xl font-bold text-green-500">{result.completed}</p>
              <p className="text-xs text-muted-foreground">Completed</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-muted-foreground">{result.skipped}</p>
              <p className="text-xs text-muted-foreground">Skipped</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-destructive">{result.failed}</p>
              <p className="text-xs text-muted-foreground">Failed</p>
            </div>
          </div>

          {hasErrors && (
            <div className="pt-2">
              <button
                onClick={() => setShowErrors(!showErrors)}
                className="flex items-center gap-1 text-xs text-destructive hover:underline"
              >
                {showErrors ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                {result.errors.length} error{result.errors.length !== 1 ? "s" : ""}
              </button>
              {showErrors && (
                <div className="mt-2 max-h-40 overflow-y-auto space-y-1.5">
                  {result.errors.map((err, i) => (
                    <div key={i} className="text-xs bg-destructive/10 rounded p-2">
                      <p className="font-medium truncate">{err.episode}</p>
                      <p className="text-muted-foreground mt-0.5 break-words">{err.error}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center justify-end px-5 py-3 border-t border-border">
          <Button onClick={onDismiss} size="sm">Done</Button>
        </div>
      </div>
    </div>
  );
}

/* ── BatchStrip — expandable progress with per-episode detail ── */

function BatchStrip() {
  const { batchTaskId, batchFolder, batchEpisodeNames, batchStep, setBatchTask } = useTaskStore();
  const progress = useProgress(batchTaskId);
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);

  const log = progress?.log ?? [];

  // Auto-scroll log
  useEffect(() => {
    if (showLog) logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [log.length, showLog]);

  if (!batchTaskId) return null;

  const pct = progress ? Math.round(progress.progress * 100) : 0;
  const msg = progress?.message || "Starting...";
  const isDone = progress?.status === "completed";
  const isFailed = progress?.status === "failed";
  const isCancelled = progress?.status === "cancelled";
  const isFinished = isDone || isFailed || isCancelled;

  const dismiss = () => {
    setBatchTask(null);
    if (batchFolder) {
      queryClient.refetchQueries({ queryKey: ["episodes", batchFolder] });
    }
  };

  const handleDismiss = () => {
    if (isFinished && progress?.result && !showResult) {
      setShowResult(true);
    } else {
      dismiss();
    }
  };

  const episodeStatuses = deriveEpisodeStatuses(batchEpisodeNames, progress);
  const stepLabel = batchStep ? batchStep.charAt(0).toUpperCase() + batchStep.slice(1) : "Batch";

  return (
    <>
      <div>
        <div
          className="px-4 py-2 flex items-center gap-3 text-xs cursor-pointer hover:bg-accent/30 transition"
          onClick={() => batchEpisodeNames.length > 0 && setExpanded(!expanded)}
        >
          <Loader2
            className={`w-3.5 h-3.5 shrink-0 ${isFinished ? "text-muted-foreground" : "text-primary animate-spin"}`}
          />
          <span className="text-foreground font-medium shrink-0">{stepLabel}</span>
          <div className="flex-1 min-w-0 flex items-center gap-2">
            <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  isFailed
                    ? "bg-destructive"
                    : isCancelled
                      ? "bg-yellow-500"
                      : isDone
                        ? "bg-green-500"
                        : "bg-primary"
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="text-muted-foreground shrink-0 w-8 text-right">{pct}%</span>
          </div>
          <span className="text-muted-foreground truncate max-w-[200px]">{msg}</span>
          {batchEpisodeNames.length > 0 && (
            expanded ? (
              <ChevronUp className="w-3 h-3 text-muted-foreground shrink-0" />
            ) : (
              <ChevronDown className="w-3 h-3 text-muted-foreground shrink-0" />
            )
          )}
          {!isFinished ? (
            <Button
              onClick={(e) => { e.stopPropagation(); cancelTask(batchTaskId).catch(() => dismiss()); }}
              variant="ghost"
              size="sm"
              className="h-5 w-5 p-0 text-muted-foreground hover:text-foreground"
              title="Cancel batch"
            >
              <X className="w-3 h-3" />
            </Button>
          ) : (
            <Button
              onClick={(e) => { e.stopPropagation(); handleDismiss(); }}
              variant="ghost"
              size="sm"
              className="h-5 w-5 p-0 text-muted-foreground hover:text-foreground"
              title="Dismiss"
            >
              <X className="w-3 h-3" />
            </Button>
          )}
        </div>

        {/* Expanded per-episode list */}
        {expanded && episodeStatuses.length > 0 && (
          <div className="px-4 pb-2 max-h-48 overflow-y-auto">
            {episodeStatuses.map(({ name, status }, i) => {
              const Icon = STATUS_ICON[status];
              return (
                <div key={i} className="flex items-center gap-2 py-0.5 text-xs">
                  <Icon className={`w-3 h-3 shrink-0 ${STATUS_COLOR[status]}`} />
                  <span className={`truncate ${status === "running" ? "text-foreground font-medium" : "text-muted-foreground"}`}>
                    {name}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {/* Collapsible log */}
        {expanded && log.length > 0 && (
          <div className="px-4 pb-2">
            <button
              onClick={(e) => { e.stopPropagation(); setShowLog(!showLog); }}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
            >
              <Terminal className="w-3 h-3" />
              {showLog ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              Logs ({log.length})
            </button>
            {showLog && (
              <pre className="mt-1.5 p-2 bg-black/40 rounded text-[10px] leading-relaxed text-muted-foreground max-h-40 overflow-auto font-mono">
                {log.map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
                <div ref={logEndRef} />
              </pre>
            )}
          </div>
        )}
      </div>

      {/* Results summary dialog */}
      {showResult && progress?.result && (
        <BatchResultSummary
          result={progress.result as BatchResult}
          onDismiss={() => { setShowResult(false); dismiss(); }}
        />
      )}
    </>
  );
}

export default function TaskBar() {
  const { downloadTaskId, batchTaskId } = useTaskStore();
  if (!downloadTaskId && !batchTaskId) return null;

  return (
    <div className="border-t border-border bg-background divide-y divide-border/50">
      <DownloadStrip />
      <BatchStrip />
    </div>
  );
}
