/** Global task progress bar — pinned above the audio bar like a persistent status strip. */

import { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useTaskStore, useBatchHistoryStore } from "@/stores";
import { cancelTask } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { useProgress } from "@/hooks/useProgress";
import { capitalize } from "@/lib/utils";
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

  // Auto-dismiss if task ID is set but neither WS nor API polling returns progress.
  // 12s gives the 5s poll interval time to fire and hydrate before we give up.
  useEffect(() => {
    if (!downloadTaskId || progress) return;
    const timer = setTimeout(() => setDownloadTask(null), 12_000);
    return () => clearTimeout(timer);
  }, [downloadTaskId, progress, setDownloadTask]);

  if (!downloadTaskId) return null;

  const pct = progress ? Math.round(progress.progress * 100) : 0;
  const msg = progress?.message || "Starting...";
  const isDone = progress?.status === "completed";
  const isFailed = progress?.status === "failed";
  const isCancelled = progress?.status === "cancelled";
  const isFinished = isDone || isFailed || isCancelled;

  // Derive per-status counts from result (array of {stem, status} or {results: [...]})
  const rawResult = isFinished ? progress?.result : null;
  const results = (Array.isArray(rawResult) ? rawResult : Array.isArray(rawResult?.results) ? rawResult.results : []) as { stem: string; title?: string; status: string; error?: string }[];
  const total = results.length || 1;
  const failedCount = results.filter(r => r.status === "failed" || r.status === "no_subtitles" || r.status === "error").length;
  const successCount = results.filter(r => r.status === "downloaded" || r.status === "exists" || r.status === "imported").length;
  const successPct = Math.round((successCount / total) * 100);
  const failedPct = Math.round((failedCount / total) * 100);
  const hasFailures = failedCount > 0;

  const dismiss = () => {
    setDownloadTask(null);
  };

  // Invalidate episode list when download completes
  const didInvalidateRef = useRef(false);
  useEffect(() => {
    if (isFinished && downloadFolder && !didInvalidateRef.current) {
      didInvalidateRef.current = true;
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesForFolder(downloadFolder) });
    }
    if (!isFinished) didInvalidateRef.current = false;
  }, [isFinished, downloadFolder, queryClient]);

  // Auto-dismiss clean completions (no failures: 2s, with failures: 30s)
  useEffect(() => {
    if (!isFinished) return;
    const delay = hasFailures ? 30_000 : 2_000;
    const t = setTimeout(dismiss, delay);
    return () => clearTimeout(t);
  }, [isFinished, hasFailures]);

  const failedResults = results.filter(r => r.status === "failed" || r.status === "no_subtitles" || r.status === "error");
  const [showFailed, setShowFailed] = useState(false);

  return (
    <div>
      <div className="px-4 py-2 flex items-center gap-3 text-xs">
        <Download className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
        <div className="flex-1 min-w-0 flex items-center gap-2">
          <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden flex">
            {isFinished && results.length > 0 ? (
              <>
                <div
                  className="h-full bg-success transition-all duration-300"
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
                  isFailed ? "bg-destructive" : isCancelled ? "bg-warning" : "bg-primary"
                }`}
                style={{ width: `${pct}%` }}
              />
            )}
          </div>
          <span className="text-muted-foreground shrink-0 w-8 text-right">{pct}%</span>
        </div>
        <span className="text-muted-foreground truncate max-w-[300px]">{msg}</span>
        {isFinished && failedResults.length > 0 && (
          <button onClick={() => setShowFailed(!showFailed)} className="text-destructive hover:text-destructive/80 transition text-2xs shrink-0">
            {failedResults.length} failed
          </button>
        )}
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
      {showFailed && failedResults.length > 0 && (
        <div className="px-4 pb-2 max-h-32 overflow-y-auto">
          {failedResults.map((r, i) => (
            <div key={i} className="flex items-center gap-2 py-0.5 text-xs text-muted-foreground">
              <AlertTriangle className="w-3 h-3 text-destructive shrink-0" />
              <span className="truncate">{r.title || r.stem}</span>
              <span className="text-2xs text-muted-foreground/60 shrink-0">{r.error || (r.status === "no_subtitles" ? "no subs" : r.status)}</span>
            </div>
          ))}
        </div>
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
  done: "text-success",
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
            <CheckCircle2 className="w-4 h-4 text-success" />
          )}
          <h3 className="text-base font-semibold">Batch Complete</h3>
          <div className="flex-1" />
          <button onClick={onDismiss} className="text-muted-foreground hover:text-foreground text-lg" aria-label="Close">
            ×
          </button>
        </div>

        <div className="px-5 py-4 space-y-2">
          <div className="grid grid-cols-3 gap-3 text-center">
            <div>
              <p className="text-2xl font-bold text-success">{result.completed}</p>
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
  const { batchTaskId, batchFolder, batchEpisodes, batchStep, setBatchTask } = useTaskStore();
  const progress = useProgress(batchTaskId);
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [expanded, setExpanded] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-dismiss if task ID is set but neither WS nor API polling returns progress.
  // 12s gives the 5s poll interval time to fire and hydrate before we give up.
  useEffect(() => {
    if (!batchTaskId || progress) return;
    const timer = setTimeout(() => setBatchTask(null), 12_000);
    return () => clearTimeout(timer);
  }, [batchTaskId, progress, setBatchTask]);

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
      queryClient.refetchQueries({ queryKey: queryKeys.episodesForFolder(batchFolder) });
    }
  };

  const handleDismiss = () => {
    if (isFinished && progress?.result && !showResult) {
      setShowResult(true);
    } else {
      dismiss();
    }
  };

  // Auto-dismiss after completion (30s to give time to review)
  useEffect(() => {
    if (!isFinished) return;
    const t = setTimeout(dismiss, 30_000);
    return () => clearTimeout(t);
  }, [isFinished, batchFolder]); // eslint-disable-line react-hooks/exhaustive-deps

  const episodeStatuses = useMemo(
    () => deriveEpisodeStatuses(batchEpisodes, progress),
    [batchEpisodes, progress?.status, progress?.message, progress?.result],
  );

  const savedRef = useRef<string | null>(null);
  useEffect(() => {
    if (!isFinished || savedRef.current === batchTaskId || !batchTaskId || !batchFolder || !batchStep || batchEpisodes.length === 0) return;
    savedRef.current = batchTaskId;
    const failed = episodeStatuses
      .filter((e) => e.status === "failed")
      .map((e) => ({ title: e.title, stem: e.stem, error: e.error ?? "unknown" }));
    const showMeta = queryClient.getQueryData<{ name?: string }>(queryKeys.showMeta(batchFolder));
    useBatchHistoryStore.getState().add({
      step: batchStep,
      folder: batchFolder,
      showName: showMeta?.name,
      episodes: batchEpisodes,
      failed,
      totalCount: batchEpisodes.length,
      successCount: batchEpisodes.length - failed.length,
      status: isDone ? "completed" : isFailed ? "failed" : "cancelled",
    });
  }, [isFinished, isDone, isFailed, batchTaskId, batchFolder, batchStep, batchEpisodes, episodeStatuses, queryClient]);

  const stepLabel = batchStep ? capitalize(batchStep) : "Batch";

  return (
    <>
      <div>
        <div
          className="px-4 py-2 flex items-center gap-3 text-xs cursor-pointer hover:bg-accent/30 transition"
          onClick={() => batchEpisodes.length > 0 && setExpanded(!expanded)}
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
                      ? "bg-warning"
                      : isDone
                        ? "bg-success"
                        : "bg-primary"
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="text-muted-foreground shrink-0 w-8 text-right">{pct}%</span>
          </div>
          <span className="text-muted-foreground truncate max-w-[200px]">{msg}</span>
          {batchEpisodes.length > 0 && (
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
            {episodeStatuses.map(({ title, stem, status, error }, i) => {
              const Icon = STATUS_ICON[status];
              return (
                <div
                  key={i}
                  className={`flex items-center gap-2 py-0.5 text-xs ${batchFolder ? "cursor-pointer hover:bg-accent/50 -mx-2 px-2 rounded" : ""}`}
                  onClick={(e) => {
                    if (!batchFolder) return;
                    e.stopPropagation();
                    navigate({
                      to: "/show/$folder/episode/$stem",
                      params: { folder: encodeURIComponent(batchFolder), stem: encodeURIComponent(stem) },
                    });
                  }}
                >
                  <Icon className={`w-3 h-3 shrink-0 ${STATUS_COLOR[status]}`} />
                  <span className={`truncate shrink-0 max-w-[40%] ${status === "running" ? "text-foreground font-medium" : "text-muted-foreground"}`}>
                    {title}
                  </span>
                  {error && (
                    <span className="text-2xs text-destructive/80 truncate flex-1 min-w-0" title={error}>
                      {error}
                    </span>
                  )}
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
              <pre className="mt-1.5 p-2 bg-black/40 rounded text-[0.55rem] leading-normal text-muted-foreground max-h-80 overflow-auto font-mono">
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
          result={progress.result as unknown as BatchResult}
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
