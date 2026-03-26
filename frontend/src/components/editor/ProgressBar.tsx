import { useEffect, useRef } from "react";
import { useProgress } from "@/hooks/useProgress";

interface ProgressBarProps {
  taskId: string | null;
  onComplete?: () => void;
}

export default function ProgressBar({ taskId, onComplete }: ProgressBarProps) {
  const progress = useProgress(taskId);
  const completeCalled = useRef(false);

  useEffect(() => {
    if (progress?.status === "completed" && !completeCalled.current) {
      completeCalled.current = true;
      onComplete?.();
    }
  }, [progress?.status, onComplete]);

  // Reset when taskId changes
  useEffect(() => {
    completeCalled.current = false;
  }, [taskId]);

  if (!taskId || !progress) return null;

  const pct = Math.round(progress.progress * 100);

  return (
    <div className="space-y-2 p-4">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{progress.message || "Starting..."}</span>
        <span className="text-muted-foreground">{pct}%</span>
      </div>
      <div className="h-2 rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${
            progress.status === "failed" ? "bg-destructive" : "bg-primary"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {progress.status === "failed" && progress.error && (
        <p className="text-destructive text-xs">{progress.error}</p>
      )}
      {progress.status === "completed" && (
        <p className="text-green-400 text-xs">Complete</p>
      )}
    </div>
  );
}
