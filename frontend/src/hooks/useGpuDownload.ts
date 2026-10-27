/** Progress of the GPU backend download, and the app-wide watcher that
 *  finishes it whether or not the Settings panel is open. */

import { useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getTaskStatus } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { useGpuDownloadStore } from "@/stores/gpuDownloadStore";
import { restartApp } from "@/lib/restartApp";
import { isSettled } from "@/hooks/useProgress";

const POLL_INTERVAL_MS = 1000;

/** The running download task, polled while there is one. */
export function useGpuDownloadTask() {
  const taskId = useGpuDownloadStore((s) => s.taskId);
  const { data: task } = useQuery({
    queryKey: queryKeys.gpuDownloadTask(taskId),
    queryFn: () => getTaskStatus(taskId!),
    enabled: !!taskId,
    refetchInterval: taskId ? POLL_INTERVAL_MS : false,
  });
  return { taskId, task: taskId ? task : undefined };
}

/** Mounted once, in RootLayout. */
export function useGpuDownloadWatcher(): void {
  const qc = useQueryClient();
  const finish = useGpuDownloadStore((s) => s.finish);
  const { task } = useGpuDownloadTask();

  useEffect(() => {
    if (task === null) {
      // The server no longer knows the task (a backend restart mid-download).
      finish("The download stopped: the backend restarted. Try again.");
      return;
    }
    if (!task) return;
    if (!isSettled(task)) return;
    const error = task.status === "failed" ? task.error ?? "Download failed" : null;
    if (error) console.error("GPU download task failed:", task);
    finish(error);
    qc.invalidateQueries({ queryKey: queryKeys.gpuStatus() });
    // Sidecar selection happens once at app launch, so a download that
    // completes while the backend is already activated (the update flow)
    // replaced the binary under a running sidecar: restart to pick it up.
    // A fresh install is not activated yet; Activate restarts on its own.
    if (task.status === "completed" && useGpuDownloadStore.getState().wasActivated) {
      void restartApp();
    }
  }, [task, qc, finish]);
}
