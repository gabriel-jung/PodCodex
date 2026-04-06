import { useState, useEffect, useRef } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { installExtra, getExtras } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { errorMessage } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Download, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import ProgressBar from "@/components/editor/ProgressBar";

interface MissingDependencyProps {
  /** Which extra to install (e.g. "pipeline") */
  extra: string;
  /** Human-readable name of what's missing */
  label: string;
  /** What this dependency enables */
  description: string;
}

export default function MissingDependency({ extra, label, description }: MissingDependencyProps) {
  const queryClient = useQueryClient();
  const [taskId, setTaskId] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const installMutation = useMutation({
    mutationFn: () => installExtra(extra),
    onSuccess: (data) => setTaskId(data.task_id),
  });

  // Poll capabilities every 5s while installing, as a fallback if WebSocket misses the completion
  useEffect(() => {
    if (!taskId) return;

    pollRef.current = setInterval(async () => {
      try {
        const data = await getExtras();
        const info = data.extras[extra];
        if (info?.installed) {
          handleComplete();
        }
      } catch {
        // ignore fetch errors during install
      }
    }, 5000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [taskId]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleComplete = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setTaskId(null);
    setDone(true);
    queryClient.invalidateQueries({ queryKey: queryKeys.capabilities() });
    queryClient.invalidateQueries({ queryKey: queryKeys.health() });
    queryClient.invalidateQueries({ queryKey: queryKeys.pipelineConfig() });
  };

  if (done) {
    return (
      <div className="flex items-center gap-3 p-4 rounded-md bg-green-500/10 border border-green-500/20">
        <CheckCircle2 className="w-5 h-5 text-green-500 shrink-0" />
        <div>
          <p className="text-sm font-medium">Installed successfully</p>
          <p className="text-xs text-muted-foreground mt-0.5">
            Restart the backend to activate new dependencies, then reload this page.
          </p>
        </div>
      </div>
    );
  }

  if (taskId) {
    return (
      <div className="space-y-2 p-4 rounded-md bg-muted/50 border border-border">
        <p className="text-sm text-muted-foreground">Installing {label}...</p>
        <ProgressBar
          taskId={taskId}
          onComplete={handleComplete}
          onRetry={() => { setTaskId(null); installMutation.mutate(); }}
          onDismiss={() => setTaskId(null)}
        />
        <p className="text-xs text-muted-foreground">This can take a few minutes for large packages.</p>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 p-4 rounded-md bg-muted/50 border border-border">
      <AlertCircle className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
      <div className="flex-1 space-y-2">
        <div>
          <p className="text-sm font-medium">{label} not installed</p>
          <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            size="sm"
            onClick={() => installMutation.mutate()}
            disabled={installMutation.isPending}
          >
            {installMutation.isPending ? (
              <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
            ) : (
              <Download className="w-3.5 h-3.5 mr-1.5" />
            )}
            Install {extra} extra
          </Button>
          <span className="text-xs text-muted-foreground">
            runs <code className="bg-muted px-1 py-0.5 rounded">uv sync --extra {extra}</code>
          </span>
        </div>
        {installMutation.isError && (
          <p className="text-destructive text-xs">{errorMessage(installMutation.error)}</p>
        )}
      </div>
    </div>
  );
}
