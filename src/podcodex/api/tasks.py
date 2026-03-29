"""Background task runner with WebSocket progress broadcasting."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Max debug log lines kept per task (ring buffer)
_MAX_DEBUG_LINES = 200


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
    steps: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def add_step(self, message: str) -> None:
        self.steps.append(message)

    def add_log(self, message: str) -> None:
        if len(self.log) >= _MAX_DEBUG_LINES:
            self.log = self.log[-_MAX_DEBUG_LINES // 2 :]
        self.log.append(message)


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
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # Fallback — won't work from threads, but avoids crash
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
        # Capture the event loop while we're still in async context
        self._loop = asyncio.get_running_loop()
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
            # Attach a log handler that captures debug output for this task
            log_handler = _TaskLogHandler(info, self)
            root = logging.getLogger()
            root.addHandler(log_handler)
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
                root.removeHandler(log_handler)
                info.finished_at = time.monotonic()
                self._audio_locks.pop(audio_path, None)
                # Invalidate folder scan cache so next listing reflects changes
                try:
                    from podcodex.ingest.folder import invalidate_scan_cache
                    from pathlib import Path

                    p = Path(audio_path)
                    invalidate_scan_cache(p if p.is_dir() else p.parent)
                except Exception:
                    pass
                self._broadcast_sync(task_id)

        self._executor.submit(run)
        return info

    # Patterns that represent progress ticks, not meaningful step transitions
    _TICK_RE = re.compile(r"^(Batch|Segment|Chunk)\s+\d+", re.IGNORECASE)

    def update_progress(self, task_id: str, progress: float, message: str) -> None:
        info = self._tasks.get(task_id)
        if info:
            if message and message != info.message:
                # Steps = milestone messages; ticks go to log only
                if self._TICK_RE.match(message):
                    info.add_log(message)
                else:
                    info.add_step(message)
                    info.add_log(message)
            info.progress = progress
            info.message = message
            self._broadcast_sync(task_id)

    def get(self, task_id: str) -> TaskInfo | None:
        return self._tasks.get(task_id)

    def get_active(self, audio_path: str) -> TaskInfo | None:
        """Return the running task for an audio path, if any."""
        task_id = self._audio_locks.get(audio_path)
        if not task_id:
            return None
        info = self._tasks.get(task_id)
        if info and info.status in ("pending", "running"):
            return info
        return None

    def _broadcast_sync(self, task_id: str) -> None:
        """Schedule an async broadcast from a sync/thread context."""
        try:
            loop = self._get_loop()
            asyncio.run_coroutine_threadsafe(self._broadcast(task_id), loop)
            logger.debug(
                "Broadcast scheduled for %s (loop running=%s, ws_count=%d)",
                task_id,
                loop.is_running(),
                len(self._ws_connections),
            )
        except RuntimeError as exc:
            logger.warning("WebSocket broadcast failed for %s: %s", task_id, exc)

    async def _broadcast(self, task_id: str) -> None:
        info = self._tasks.get(task_id)
        if not info:
            return
        msg: dict[str, Any] = {
            "task_id": info.task_id,
            "status": info.status,
            "progress": info.progress,
            "message": info.message,
        }
        if info.steps:
            msg["steps"] = info.steps
        if info.log:
            msg["log"] = info.log
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

    async def register_ws(self, ws: WebSocket) -> None:
        self._ws_connections.add(ws)
        # Send current state of all active tasks so reconnecting clients catch up
        for info in self._tasks.values():
            if info.status in ("pending", "running"):
                msg: dict[str, Any] = {
                    "task_id": info.task_id,
                    "status": info.status,
                    "progress": info.progress,
                    "message": info.message,
                }
                if info.steps:
                    msg["steps"] = info.steps
                if info.log:
                    msg["log"] = info.log
                try:
                    await ws.send_json(msg)
                except Exception:
                    pass

    def unregister_ws(self, ws: WebSocket) -> None:
        self._ws_connections.discard(ws)


class _TaskLogHandler(logging.Handler):
    """Captures log records from any logger while a task is running."""

    def __init__(
        self, task_info: TaskInfo, manager: "TaskManager", level: int = logging.DEBUG
    ) -> None:
        super().__init__(level)
        self._info = task_info
        self._manager = manager
        self._last_broadcast = 0.0
        self.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._info.add_log(self.format(record))
            # Throttle broadcasts to at most once per second
            now = time.monotonic()
            if now - self._last_broadcast >= 1.0:
                self._last_broadcast = now
                self._manager._broadcast_sync(self._info.task_id)
        except Exception:
            pass


# Module-level singleton
task_manager = TaskManager()
