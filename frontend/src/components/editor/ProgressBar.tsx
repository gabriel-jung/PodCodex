import { useEffect, useRef, useState } from "react";
import { useProgress } from "@/hooks/useProgress";
import { ChevronDown, ChevronRight, Check, Loader2, AlertCircle, Terminal, RotateCcw, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ProgressBarProps {
  taskId: string | null;
  onComplete?: () => void;
  onRetry?: () => void;
  onDismiss?: () => void;
  onCancel?: () => void;
}

/** Seconds without a WebSocket update before showing the "stuck" hint. */
const STALE_THRESHOLD = 30;

export default function ProgressBar({ taskId, onComplete, onRetry, onDismiss, onCancel }: ProgressBarProps) {
  const progress = useProgress(taskId);
  const completeCalled = useRef(false);
  const [showLog, setShowLog] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);
  const [stale, setStale] = useState(false);
  const lastUpdateRef = useRef(Date.now());

  useEffect(() => {
    if (progress?.status === "completed" && !completeCalled.current) {
      completeCalled.current = true;
      onComplete?.();
    }
  }, [progress?.status, onComplete]);

  useEffect(() => {
    completeCalled.current = false;
    setShowLog(false);
    setStale(false);
    lastUpdateRef.current = Date.now();
  }, [taskId]);

  // Track last update time and detect stale state
  useEffect(() => {
    if (progress) {
      lastUpdateRef.current = Date.now();
      setStale(false);
    }
  }, [progress?.progress, progress?.message, progress?.log?.length]);

  useEffect(() => {
    if (!taskId || progress?.status === "completed" || progress?.status === "failed") return;
    const interval = setInterval(() => {
      if (Date.now() - lastUpdateRef.current > STALE_THRESHOLD * 1000) {
        setStale(true);
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [taskId, progress?.status]);

  // Auto-scroll log when open
  useEffect(() => {
    if (showLog) logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [progress?.log?.length, showLog]);

  if (!taskId) return null;

  const pct = progress ? Math.round(progress.progress * 100) : 0;
  const steps = progress?.steps ?? [];
  const log = progress?.log ?? [];
  const currentMsg = progress?.message || "Starting...";
  const isFailed = progress?.status === "failed";
  const isDone = progress?.status === "completed";
  const isCancelled = progress?.status === "cancelled";
  const isRunning = !isDone && !isFailed && !isCancelled;
  const showRetry = isFailed || stale;

  return (
    <div className="p-4 space-y-3">
      {/* Progress bar */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground truncate mr-2">{currentMsg}</span>
          <div className="flex items-center gap-2 shrink-0">
            {stale && isRunning && (
              <span className="text-xs text-yellow-500">No updates</span>
            )}
            <span className="text-muted-foreground">{pct}%</span>
            {onCancel && isRunning && (
              <Button onClick={onCancel} variant="ghost" size="sm" className="text-xs h-6 px-1.5">
                Cancel
              </Button>
            )}
            {onDismiss && !isRunning && (
              <Button onClick={onDismiss} variant="ghost" size="sm" className="text-xs h-6 px-1.5">
                Dismiss
              </Button>
            )}
          </div>
        </div>
        <div className="h-2 rounded-full bg-muted overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              isFailed ? "bg-destructive" : stale ? "bg-yellow-500" : isDone ? "bg-green-500" : "bg-primary"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Step list */}
      {steps.length > 0 && (
        <div className="space-y-1">
          {steps.map((step, i) => {
            const isLast = i === steps.length - 1;
            const isActive = isLast && !isDone && !isFailed;
            return (
              <div key={i} className="flex items-center gap-2 text-xs">
                {isActive ? (
                  <Loader2 className="w-3 h-3 text-primary animate-spin shrink-0" />
                ) : isFailed && isLast ? (
                  <AlertCircle className="w-3 h-3 text-destructive shrink-0" />
                ) : (
                  <Check className="w-3 h-3 text-green-500 shrink-0" />
                )}
                <span className={isActive ? "text-foreground" : "text-muted-foreground"}>
                  {step}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Error */}
      {isFailed && progress?.error && (
        <p className="text-destructive text-xs">{progress.error}</p>
      )}

      {/* Done */}
      {isDone && (
        <p className="text-green-400 text-xs">Complete</p>
      )}

      {/* Retry / Dismiss actions */}
      {showRetry && (
        <div className="flex items-center gap-2">
          {onRetry && (
            <Button onClick={onRetry} variant="outline" size="sm">
              <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
              Retry
            </Button>
          )}
          {onDismiss && (
            <Button onClick={onDismiss} variant="ghost" size="sm">
              <X className="w-3.5 h-3.5 mr-1.5" />
              Dismiss
            </Button>
          )}
        </div>
      )}

      {/* Debug log — expandable */}
      {log.length > 0 && (
        <div className="border-t border-border/50 pt-2">
          <button
            onClick={() => setShowLog(!showLog)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
          >
            <Terminal className="w-3 h-3" />
            {showLog ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            Logs ({log.length})
          </button>
          {showLog && (
            <pre className="mt-2 p-2 bg-black/40 rounded text-[10px] leading-relaxed text-muted-foreground max-h-48 overflow-auto font-mono">
              {log.map((line, i) => (
                <div key={i}>{line}</div>
              ))}
              <div ref={logEndRef} />
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
