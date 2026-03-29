import { useEffect, useState } from "react";

export interface TaskProgress {
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  message: string;
  steps?: string[];
  log?: string[];
  result?: Record<string, unknown>;
  error?: string;
}

type Listener = (data: TaskProgress) => void;

class ProgressManager {
  private ws: WebSocket | null = null;
  private listeners = new Map<string, Set<Listener>>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;

  connect(): void {
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) return;

    const isTauriProd = (window as any).__TAURI__ && import.meta.env.PROD;
    const url = isTauriProd
      ? "ws://127.0.0.1:18811/api/ws"
      : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/api/ws`;
    this.ws = new WebSocket(url);

    this.ws.onmessage = (event) => {
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

  useEffect(() => {
    if (!taskId) {
      setState(null);
      return;
    }
    return manager.subscribe(taskId, setState);
  }, [taskId]);

  return state;
}
