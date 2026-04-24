import { useEffect, useState } from "react";
import { getActiveTask } from "@/api/client";

/**
 * On mount, check if there's already a running task for this audio path.
 *
 * Task IDs are of the form `{step}_{hex8}`. When ``step`` is provided, only
 * tasks for that step are returned — the caller is a step panel and should
 * not render progress for an unrelated step (e.g. indexing progress leaking
 * into the correct panel).
 *
 * Returns [task_id, setter] so consumers can clear it after dismissing.
 */
export function useActiveTask(
  audioPath: string | null | undefined,
  step?: string,
): [string | null, (id: string | null) => void] {
  const [taskId, setTaskId] = useState<string | null>(null);

  useEffect(() => {
    if (!audioPath) return;
    let cancelled = false;
    setTaskId(null);
    getActiveTask(audioPath).then((data) => {
      if (cancelled || !data?.task_id) return;
      if (step && !data.task_id.startsWith(`${step}_`)) return;
      setTaskId(data.task_id);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [audioPath, step]);

  return [taskId, setTaskId];
}
