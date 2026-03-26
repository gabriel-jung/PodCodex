"""Background task runner with WebSocket progress broadcasting."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    task_id: str
    audio_path: str = ""
    status: str = "pending"  # pending | running | completed | failed
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    finished_at: float | None = None


class TaskManager:
    """Manages background pipeline tasks with WebSocket progress updates."""

    def __init__(self, max_workers: int = 2) -> None:
        self._tasks: dict[str, TaskInfo] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._ws_connections: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_locks: dict[str, str] = {}  # audio_path → task_id

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_event_loop()
        return self._loop

    def _cleanup_stale(self, max_age: float = 600.0) -> None:
        """Remove completed/failed tasks older than *max_age* seconds."""
        now = time.monotonic()
        stale = [
            tid
            for tid, t in self._tasks.items()
            if t.finished_at is not None and now - t.finished_at > max_age
        ]
        for tid in stale:
            t = self._tasks.pop(tid, None)
            if t:
                self._audio_locks.pop(t.audio_path, None)

    def submit(
        self,
        step: str,
        audio_path: str,
        fn: Callable,
        *args: Any,
    ) -> TaskInfo:
        """Submit a pipeline function for background execution.

        ``fn`` receives a progress callback as its first argument:
        ``fn(progress_cb, *args)`` where ``progress_cb(progress, message)``.
        """
        self._cleanup_stale()
        # Check for existing task on this audio_path
        if audio_path in self._audio_locks:
            existing_id = self._audio_locks[audio_path]
            existing = self._tasks.get(existing_id)
            if existing and existing.status in ("pending", "running"):
                raise ValueError(f"Task {existing_id} already running on {audio_path}")

        task_id = f"{step}_{uuid.uuid4().hex[:8]}"
        info = TaskInfo(task_id=task_id, audio_path=audio_path)
        self._tasks[task_id] = info
        self._audio_locks[audio_path] = task_id

        def progress_cb(progress: float, message: str) -> None:
            self.update_progress(task_id, progress, message)

        def run() -> None:
            info.status = "running"
            self._broadcast_sync(task_id)
            try:
                result = fn(progress_cb, *args)
                info.status = "completed"
                info.progress = 1.0
                info.message = "Done"
                info.result = result
            except Exception as exc:
                logger.exception("Task %s failed", task_id)
                info.status = "failed"
                info.error = str(exc)
            finally:
                info.finished_at = time.monotonic()
                self._audio_locks.pop(audio_path, None)
                self._broadcast_sync(task_id)

        self._executor.submit(run)
        return info

    def update_progress(self, task_id: str, progress: float, message: str) -> None:
        info = self._tasks.get(task_id)
        if info:
            info.progress = progress
            info.message = message
            self._broadcast_sync(task_id)

    def get(self, task_id: str) -> TaskInfo | None:
        return self._tasks.get(task_id)

    def _broadcast_sync(self, task_id: str) -> None:
        """Schedule an async broadcast from a sync/thread context."""
        try:
            loop = self._get_loop()
            asyncio.run_coroutine_threadsafe(self._broadcast(task_id), loop)
        except RuntimeError:
            logger.debug("WebSocket broadcast skipped (no event loop)")

    async def _broadcast(self, task_id: str) -> None:
        info = self._tasks.get(task_id)
        if not info:
            return
        msg = {
            "task_id": info.task_id,
            "status": info.status,
            "progress": info.progress,
            "message": info.message,
        }
        if info.result is not None:
            msg["result"] = info.result
        if info.error is not None:
            msg["error"] = info.error

        dead: list[WebSocket] = []
        for ws in self._ws_connections:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_connections.discard(ws)

    def register_ws(self, ws: WebSocket) -> None:
        self._ws_connections.add(ws)

    def unregister_ws(self, ws: WebSocket) -> None:
        self._ws_connections.discard(ws)


# Module-level singleton
task_manager = TaskManager()
