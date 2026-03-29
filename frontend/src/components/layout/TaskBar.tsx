/** Global task progress bar — pinned above the audio bar like a persistent status strip. */

import { useQueryClient } from "@tanstack/react-query";
import { useTaskStore } from "@/stores";
import { cancelTask } from "@/api/client";
import { useProgress } from "@/hooks/useProgress";
import { Button } from "@/components/ui/button";
import { Download, Loader2, X } from "lucide-react";

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

  const dismiss = () => {
    setDownloadTask(null);
    if (downloadFolder) {
      queryClient.refetchQueries({ queryKey: ["episodes", downloadFolder] });
    }
  };

  if (isFinished && isDone) {
    // Auto-dismiss completed downloads after a short delay
    setTimeout(dismiss, 2000);
  }

  return (
    <div className="px-4 py-2 flex items-center gap-3 text-xs">
      <Download className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
      <div className="flex-1 min-w-0 flex items-center gap-2">
        <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              isFailed ? "bg-destructive" : isCancelled ? "bg-yellow-500" : isDone ? "bg-green-500" : "bg-primary"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-muted-foreground shrink-0 w-8 text-right">{pct}%</span>
      </div>
      <span className="text-muted-foreground truncate max-w-[200px]">{msg}</span>
      {!isFinished ? (
        <Button
          onClick={() => cancelTask(downloadTaskId)}
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

function BatchStrip() {
  const { batchTaskId, batchFolder, setBatchTask } = useTaskStore();
  const progress = useProgress(batchTaskId);
  const queryClient = useQueryClient();

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

  return (
    <div className="px-4 py-2 flex items-center gap-3 text-xs">
      <Loader2 className={`w-3.5 h-3.5 shrink-0 ${isFinished ? "text-muted-foreground" : "text-primary animate-spin"}`} />
      <div className="flex-1 min-w-0 flex items-center gap-2">
        <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              isFailed ? "bg-destructive" : isCancelled ? "bg-yellow-500" : isDone ? "bg-green-500" : "bg-primary"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-muted-foreground shrink-0 w-8 text-right">{pct}%</span>
      </div>
      <span className="text-muted-foreground truncate max-w-[200px]">{msg}</span>
      {!isFinished ? (
        <Button
          onClick={() => cancelTask(batchTaskId)}
          variant="ghost"
          size="sm"
          className="h-5 w-5 p-0 text-muted-foreground hover:text-foreground"
          title="Cancel batch"
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
