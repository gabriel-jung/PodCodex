import { useEffect, useRef, useState } from "react";
import { getTaskStatus, withToken } from "@/api/client";
import { isTauri } from "@/platform";

export interface TaskProgress {
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  message: string;
  steps?: string[];
  log?: string[];
  result?: Record<string, unknown>;
  error?: string;
  /** Cancelled, but the work has not stopped yet and the episode is still
   *  locked: a re-run started now is refused. */
  stopping?: boolean;
}

/** Cancelled, with the work still winding down and the episode locked. */
/** Anything with a task status: the live progress, or a `getTaskStatus` read. */
type TaskState = { status: string; stopping?: boolean };

export function isStopping(p: TaskState | null | undefined): boolean {
  return p?.status === "cancelled" && !!p.stopping;
}

/** Ended for good: completed, failed, or cancelled and stopped. The one rule
 *  for when a strip or panel may offer dismiss or a re-run. */
export function isSettled(p: TaskState | null | undefined): boolean {
  if (!p) return false;
  return (p.status === "completed" || p.status === "failed" || p.status === "cancelled") && !p.stopping;
}

type Listener = (data: TaskProgress) => void;

/** Milliseconds of WS silence before falling back to HTTP polling. */
const STALE_THRESHOLD_MS = 5_000;
/** How often to poll when WS is silent. */
const POLL_INTERVAL_MS = 5_000;

class ProgressManager {
  private ws: WebSocket | null = null;
  private listeners = new Map<string, Set<Listener>>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  /** Timestamp of last WS message received (any task). */
  lastWsMessage = 0;

  connect(): void {
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) return;

    const isTauriProd = isTauri() && import.meta.env.PROD;
    const url = isTauriProd
      ? "ws://127.0.0.1:18811/api/ws"
      : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/api/ws`;
    // Browser WebSocket can't send custom headers; the token rides the query
    // string and the server route checks it before accepting.
    this.ws = new WebSocket(withToken(url));

    this.ws.onmessage = (event) => {
      this.lastWsMessage = Date.now();
      try {
        const data = JSON.parse(event.data) as TaskProgress & { task_id: string };
        const callbacks = this.listeners.get(data.task_id);
        if (callbacks) {
          for (const cb of callbacks) cb(data);
        }
      } catch {
        // ignore malformed messages
      }
    };

    this.ws.onclose = () => {
      if (this.listeners.size === 0) return;
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, 10000);
      this.reconnectTimer = setTimeout(() => this.connect(), this.reconnectDelay);
    };

    this.ws.onopen = () => {
      this.reconnectDelay = 1000;
    };
  }

  subscribe(taskId: string, callback: Listener): () => void {
    if (!this.listeners.has(taskId)) {
      this.listeners.set(taskId, new Set());
    }
    this.listeners.get(taskId)!.add(callback);
    this.connect();

    return () => {
      const set = this.listeners.get(taskId);
      if (set) {
        set.delete(callback);
        if (set.size === 0) this.listeners.delete(taskId);
      }
      // Disconnect when no listeners remain
      if (this.listeners.size === 0) this.disconnect();
    };
  }

  disconnect(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.ws) this.ws.close();
    this.ws = null;
  }
}

const manager = new ProgressManager();

export function useProgress(taskId: string | null): TaskProgress | null {
  // Tagged with the task it belongs to: when the id changes from one task to
  // another, the previous task's finished state must not read as the new
  // task's (a caller would act on "completed" before the new run started).
  const [tagged, setTagged] = useState<{ taskId: string; data: TaskProgress | null } | null>(null);
  const lastWsUpdateRef = useRef<number>(0);

  useEffect(() => {
    if (!taskId) return;
    const setState = (data: TaskProgress | null) => setTagged({ taskId, data });

    const unsubscribe = manager.subscribe(taskId, (data) => {
      lastWsUpdateRef.current = Date.now();
      setState(data);
    });

    // Poll the REST API when WS has been silent — covers server restarts,
    // dropped connections, and tasks that ended without broadcasting.
    const interval = setInterval(async () => {
      const sinceLastUpdate = Date.now() - lastWsUpdateRef.current;
      if (sinceLastUpdate < STALE_THRESHOLD_MS) return;

      try {
        const status = await getTaskStatus(taskId);
        if (status) {
          lastWsUpdateRef.current = Date.now();
          setState(status as TaskProgress);
        } else {
          // Task gone from server (cleaned up or never existed).
          // Reset to null so TaskBar dismiss timer can fire.
          setState(null);
        }
      } catch {
        // Transient network error — don't clear state, retry next interval.
      }
    }, POLL_INTERVAL_MS);

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, [taskId]);

  return taskId && tagged?.taskId === taskId ? tagged.data : null;
}
