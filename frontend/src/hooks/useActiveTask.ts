import { useEffect, useState } from "react";
import { getActiveTask } from "@/api/client";

/**
 * On mount, check if there's already a running task for this audio path.
 * Returns the task_id if found, so the panel can reconnect to it.
 */
export function useActiveTask(audioPath: string | null | undefined): string | null {
  const [taskId, setTaskId] = useState<string | null>(null);

  useEffect(() => {
    if (!audioPath) return;
    let cancelled = false;
    getActiveTask(audioPath).then((data) => {
      if (!cancelled && data?.task_id) {
        setTaskId(data.task_id);
      }
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [audioPath]);

  return taskId;
}
