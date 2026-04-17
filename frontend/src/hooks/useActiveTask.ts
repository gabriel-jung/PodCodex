import { useEffect, useState } from "react";
import { getActiveTask } from "@/api/client";

/**
 * On mount, check if there's already a running task for this audio path.
 * Returns [task_id, setter] so consumers can clear it after dismissing.
 */
export function useActiveTask(
  audioPath: string | null | undefined,
): [string | null, (id: string | null) => void] {
  const [taskId, setTaskId] = useState<string | null>(null);

  useEffect(() => {
    if (!audioPath) return;
    let cancelled = false;
    setTaskId(null);
    getActiveTask(audioPath).then((data) => {
      if (!cancelled && data?.task_id) {
        setTaskId(data.task_id);
      }
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [audioPath]);

  return [taskId, setTaskId];
}
