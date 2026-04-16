import { useEffect, useRef, useState } from "react";
import { getTaskStatus } from "@/api/client";

export interface TaskProgress {
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  message: string;
  steps?: string[];
  log?: string[];
  result?: Record<string, unknown>;
  error?: string;
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

    const isTauriProd = (window as any).__TAURI__ && import.meta.env.PROD;
    const url = isTauriProd
      ? "ws://127.0.0.1:18811/api/ws"
      : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/api/ws`;
    this.ws = new WebSocket(url);

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
  const [state, setState] = useState<TaskProgress | null>(null);
  const lastWsUpdateRef = useRef<number>(0);

  useEffect(() => {
    if (!taskId) {
      setState(null);
      return;
    }

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

  return state;
}
