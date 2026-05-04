import { useState, useCallback, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useActiveTask } from "@/hooks/useActiveTask";
import { cancelTask } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import type { Episode } from "@/api/types";

/**
 * Shared hook for pipeline panel task management.
 * Replaces the repeated pattern of useActiveTask + taskId state + handlers.
 *
 * `optimisticPatch` runs the moment the task completes and patches the
 * matching episode (by stem) in the ["episodes"] cache before the slow
 * /episodes refetch returns. /episodes walks the folder + LanceDB on big
 * libraries — without the patch, status pills can lag a minute.
 */
interface PipelineTaskOpts {
  onComplete?: () => void;
  targetStem?: string | null;
  optimisticPatch?: (ep: Episode) => Partial<Episode>;
}

export function usePipelineTask(
  audioPath: string | null | undefined,
  stepKey: string,
  opts?: PipelineTaskOpts,
) {
  const queryClient = useQueryClient();
  const [resumedTaskId, setResumedTaskId] = useActiveTask(audioPath, stepKey);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const activeTaskId = taskId || resumedTaskId;

  // Single ref for opts so handleComplete identity stays put across renders.
  const optsRef = useRef<PipelineTaskOpts | undefined>(opts);
  // eslint-disable-next-line react-hooks/refs
  optsRef.current = opts;

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
    // Indexing populates LanceDB — search panel reads its own ["search", …]
    // namespace (stats, indexed-episodes, …), so the ["index", …] sweep above
    // doesn't reach it. Without this, the SearchPanel keeps showing
    // "No indexed episodes yet" until the user manually reloads.
    if (stepKey === "index") {
      queryClient.invalidateQueries({ queryKey: ["search"] });
    }
  }, [queryClient, stepKey, audioPath]);

  const applyOptimisticPatch = useCallback(() => {
    const { targetStem, optimisticPatch } = optsRef.current ?? {};
    if (!targetStem || !optimisticPatch) return;
    queryClient.setQueriesData<Episode[] | undefined>(
      { queryKey: queryKeys.episodesAll() },
      (prev) => prev?.map((e) => (e.stem === targetStem ? { ...e, ...optimisticPatch(e) } : e)),
    );
  }, [queryClient]);

  const handleComplete = useCallback(() => {
    applyOptimisticPatch();
    refreshQueries();
    clearActive();
    setExpanded(false);
    optsRef.current?.onComplete?.();
  }, [applyOptimisticPatch, refreshQueries, clearActive]);

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
