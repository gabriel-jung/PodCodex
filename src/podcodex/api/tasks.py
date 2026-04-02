"""Background task runner with WebSocket progress broadcasting."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import WebSocket

from loguru import logger

# Max debug log lines kept per task (ring buffer)
_MAX_DEBUG_LINES = 200


@dataclass
class TaskInfo:
    """Mutable state for a single background pipeline task.

    Tracks status, progress percentage, milestone steps, debug log lines,
    and a threading event for cooperative cancellation.

    Attributes:
        task_id: Unique identifier (``{step}_{hex8}``).
        audio_path: Filesystem path this task operates on (used for locking).
        status: One of ``"pending"``, ``"running"``, ``"completed"``,
            ``"failed"``, or ``"cancelled"``.
        progress: Completion fraction in ``[0.0, 1.0]``.
        message: Most recent human-readable progress message.
        result: Return value of the wrapped callable on success.
        error: Stringified exception on failure.
        finished_at: ``time.monotonic()`` timestamp when the task ended.
        steps: Ordered list of milestone messages (no tick-level noise).
        log: Ring-buffered debug log lines (capped at ``_MAX_DEBUG_LINES``).
        cancel_event: Set by ``TaskManager.cancel`` to request cooperative stop.
    """

    task_id: str
    audio_path: str = ""
    status: str = "pending"  # pending | running | completed | failed | cancelled
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    finished_at: float | None = None
    steps: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def add_step(self, message: str) -> None:
        """Append a milestone message to the steps list."""
        self.steps.append(message)

    def add_log(self, message: str) -> None:
        """Append a debug line to the log ring buffer.

        When the buffer exceeds ``_MAX_DEBUG_LINES``, the oldest half is
        discarded to keep memory bounded.
        """
        if len(self.log) >= _MAX_DEBUG_LINES:
            self.log = self.log[-_MAX_DEBUG_LINES // 2 :]
        self.log.append(message)


class TaskManager:
    """Manages background pipeline tasks with WebSocket progress updates.

    Wraps a ``ThreadPoolExecutor`` to run pipeline callables off the async
    event loop, while broadcasting real-time progress to connected WebSocket
    clients.  Per-audio-path locking prevents duplicate concurrent runs on
    the same file.
    """

    def __init__(self, max_workers: int = 2) -> None:
        """Initialise the task manager.

        Args:
            max_workers: Maximum concurrent background threads.
        """
        self._tasks: dict[str, TaskInfo] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._ws_connections: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_locks: dict[str, str] = {}  # audio_path → task_id

    # ── Public lock API ────────────────────────────

    def lock(self, audio_path: str, task_id: str) -> None:
        """Acquire a processing lock on an audio path.

        Args:
            audio_path: Filesystem path to lock.
            task_id: Owning task identifier.
        """
        self._audio_locks[audio_path] = task_id

    def unlock(self, audio_path: str) -> None:
        """Release the processing lock on an audio path.

        Args:
            audio_path: Filesystem path to unlock.
        """
        self._audio_locks.pop(audio_path, None)

    def release_locks_for_task(self, task_id: str) -> None:
        """Release all locks held by a given task.

        Useful for batch-task cleanup where one task may lock multiple
        audio paths.

        Args:
            task_id: Task whose locks should be released.
        """
        stale = [k for k, v in self._audio_locks.items() if v == task_id]
        for k in stale:
            del self._audio_locks[k]

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the cached asyncio event loop, refreshing if stale."""
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

        ``fn`` receives a progress callback as its first positional argument::

            fn(progress_cb, *args)

        where ``progress_cb(progress: float, message: str)`` updates the
        task's progress and broadcasts to WebSocket clients.  The callback
        also carries a ``cancel_event`` attribute for cooperative cancellation.

        Args:
            step: Short label for the pipeline step (e.g. ``"transcribe"``).
            audio_path: Filesystem path being processed (used for locking).
            fn: Callable to run in a background thread.
            *args: Extra positional arguments forwarded to *fn* after the
                progress callback.

        Returns:
            The newly created ``TaskInfo`` instance.

        Raises:
            ValueError: If another task is already running on *audio_path*.
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

        progress_cb.cancel_event = info.cancel_event  # type: ignore[attr-defined]

        def run() -> None:
            info.status = "running"
            self._broadcast_sync(task_id)
            # Attach a log handler that captures debug output for this task
            log_handler = _TaskLogHandler(info, self)
            root = logging.getLogger()
            root.addHandler(log_handler)
            # Also capture loguru output (used by core pipeline modules)
            loguru_sink_id = _add_loguru_sink(info, self)
            try:
                result = fn(progress_cb, *args)
                if not info.cancel_event.is_set():
                    info.status = "completed"
                    info.progress = 1.0
                    info.message = "Done"
                    info.result = result
            except Exception as exc:
                if not info.cancel_event.is_set():
                    logger.exception("Task %s failed", task_id)
                    info.status = "failed"
                    info.error = str(exc)
            finally:
                root.removeHandler(log_handler)
                _remove_loguru_sink(loguru_sink_id)
                if info.finished_at is None:
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
    _TICK_RE = re.compile(
        r"^(Batch|Segment|Chunk|Downloading|\[\d+/\d+\])\s*", re.IGNORECASE
    )

    def update_progress(self, task_id: str, progress: float, message: str) -> None:
        """Update a task's progress and broadcast the change.

        Milestone messages are recorded as steps; tick-level messages
        (matching ``_TICK_RE``) are sent to the debug log only.

        Args:
            task_id: Task to update.
            progress: New completion fraction in ``[0.0, 1.0]``.
            message: Human-readable progress description.
        """
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
        """Look up a task by its identifier.

        Args:
            task_id: Task identifier to look up.

        Returns:
            The ``TaskInfo`` if found, otherwise ``None``.
        """
        return self._tasks.get(task_id)

    def get_active(self, audio_path: str) -> TaskInfo | None:
        """Return the active (pending or running) task for an audio path.

        Args:
            audio_path: Filesystem path to check.

        Returns:
            The ``TaskInfo`` if a task is currently active, otherwise ``None``.
        """
        task_id = self._audio_locks.get(audio_path)
        if not task_id:
            return None
        info = self._tasks.get(task_id)
        if info and info.status in ("pending", "running"):
            return info
        return None

    def cancel(self, task_id: str) -> bool:
        """Request cooperative cancellation of a running task.

        Sets the task's ``cancel_event``, marks it as cancelled, releases
        all locks, and broadcasts the final state.

        Args:
            task_id: Task to cancel.

        Returns:
            ``True`` if the cancellation event was set, ``False`` if the task
            was not found or already finished.
        """
        info = self._tasks.get(task_id)
        if not info or info.status not in ("pending", "running"):
            return False
        info.cancel_event.set()
        info.status = "cancelled"
        info.message = "Cancelled"
        info.finished_at = time.monotonic()
        # Release primary lock + any per-episode locks held by this task
        self._audio_locks.pop(info.audio_path, None)
        self.release_locks_for_task(task_id)
        self._broadcast_sync(task_id)
        return True

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
        """Send a task's current state to all connected WebSocket clients."""
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
        """Register a WebSocket connection and replay active task states.

        Sends the current state of all pending/running tasks so that
        reconnecting clients can catch up immediately.

        Args:
            ws: The WebSocket connection to register.
        """
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
        """Remove a WebSocket connection from the broadcast set.

        Args:
            ws: The WebSocket connection to remove.
        """
        self._ws_connections.discard(ws)


def _add_loguru_sink(info: TaskInfo, manager: "TaskManager") -> int | None:
    """Add a loguru sink that feeds into a task's log buffer."""
    try:
        from loguru import logger as loguru_logger

        _last_broadcast: dict[str, float] = {"t": 0.0}

        def sink(message: Any) -> None:
            text = str(message).rstrip()
            if not text:
                return
            info.add_log(text)
            now = time.monotonic()
            if now - _last_broadcast["t"] >= 1.0:
                _last_broadcast["t"] = now
                manager._broadcast_sync(info.task_id)

        return loguru_logger.add(sink, level="INFO", format="{name}: {message}")
    except ImportError:
        return None


def _remove_loguru_sink(sink_id: int | None) -> None:
    """Remove a previously added loguru sink, ignoring errors."""
    if sink_id is None:
        return
    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove(sink_id)
    except Exception:
        pass


class _TaskLogHandler(logging.Handler):
    """Captures log records from any logger while a task is running."""

    def __init__(
        self, task_info: TaskInfo, manager: "TaskManager", level: int = logging.DEBUG
    ) -> None:
        """Bind this handler to a specific task and manager for broadcasting."""
        super().__init__(level)
        self._info = task_info
        self._manager = manager
        self._last_broadcast = 0.0
        self.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """Write the formatted record to the task log, throttling broadcasts."""
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
