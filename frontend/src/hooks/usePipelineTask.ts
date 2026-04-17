import { useState, useCallback, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useActiveTask } from "@/hooks/useActiveTask";
import { cancelTask } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";

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
  const [resumedTaskId, setResumedTaskId] = useActiveTask(audioPath);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const activeTaskId = taskId || resumedTaskId;

  // Keep a stable ref to onComplete so handleComplete doesn't change identity
  const onCompleteRef = useRef(opts?.onComplete);
  onCompleteRef.current = opts?.onComplete;

  const clearActive = useCallback(() => {
    setTaskId(null);
    setResumedTaskId(null);
  }, [setResumedTaskId]);

  const refreshQueries = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: queryKeys.stepAll(stepKey) });
    queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
    // Unified versions endpoint feeds cross-step input-version selectors
    // (e.g. Translate can read both corrected and transcript versions).
    queryClient.invalidateQueries({ queryKey: queryKeys.allVersions(audioPath) });
  }, [queryClient, stepKey, audioPath]);

  const handleComplete = useCallback(() => {
    refreshQueries();
    clearActive();
    setExpanded(false);
    onCompleteRef.current?.();
  }, [refreshQueries, clearActive]);

  const handleRetry = useCallback(() => {
    clearActive();
    setExpanded(true);
  }, [clearActive]);

  // Hybrid dismiss — also asks backend to cancel so a hung/running task
  // releases its audio-path lock, otherwise the next run hits "already
  // running". Backend cancel is idempotent for finished tasks.
  const handleDismiss = useCallback(() => {
    const id = taskId || resumedTaskId;
    if (id) cancelTask(id).catch(() => {});
    clearActive();
  }, [taskId, resumedTaskId, clearActive]);

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
