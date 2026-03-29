import { useState, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useActiveTask } from "@/hooks/useActiveTask";

/**
 * Shared hook for pipeline panel task management.
 * Replaces the repeated pattern of useActiveTask + taskId state + handlers.
 */
export function usePipelineTask(
  audioPath: string | null | undefined,
  stepKey: string,
  opts?: { onComplete?: () => void },
) {
  const queryClient = useQueryClient();
  const resumedTaskId = useActiveTask(audioPath);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const activeTaskId = taskId || resumedTaskId;

  const refreshQueries = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: [stepKey] });
    queryClient.invalidateQueries({ queryKey: ["episodes"] });
  }, [queryClient, stepKey]);

  const handleComplete = useCallback(() => {
    refreshQueries();
    setTaskId(null);
    setExpanded(false);
    opts?.onComplete?.();
  }, [refreshQueries, opts?.onComplete]);

  const handleRetry = useCallback(() => {
    setTaskId(null);
    setExpanded(true);
  }, []);

  const handleDismiss = useCallback(() => {
    setTaskId(null);
  }, []);

  const startTask = useCallback((taskId: string) => {
    setTaskId(taskId);
  }, []);

  return {
    activeTaskId,
    expanded,
    setExpanded,
    startTask,
    refreshQueries,
    handleComplete,
    handleRetry,
    handleDismiss,
  };
}
