"""One place that fakes a running background task.

``TaskManager`` has no public way to say "pretend a task holds this lock",
and several suites need one: the delete/move guards, and the manager's own
tests. Every test that reached for ``task_manager._tasks`` directly broke on
each internal rename, so the poking lives here and nowhere else.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def active_task(key: str, task_id: str = "t1", *, finished: bool = False):
    """Register a task holding ``key``, and always release it.

    ``finished`` stamps ``finished_at`` so the task reads as done while still
    holding its lock: the state ``cancel()`` leaves behind while the child
    winds down, and the one ``_cleanup_stale`` acts on.

    The teardown matters — a leaked lock makes every later delete in the
    session 409.
    """
    import time

    from podcodex.api.tasks import TaskInfo, task_manager

    info = TaskInfo(task_id=task_id, audio_path=key)
    info.status = "running"
    if finished:
        info.finished_at = time.monotonic()
    task_manager._tasks[task_id] = info
    task_manager.lock(key, task_id)
    try:
        yield info
    finally:
        task_manager.unlock(key)
        task_manager._tasks.pop(task_id, None)


def wait_task(task_id: str, timeout: float = 5.0):
    """Poll ``task_manager`` until *task_id* finishes; return its ``TaskInfo``."""
    import time

    from podcodex.api.tasks import task_manager

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = task_manager.get(task_id)
        if info is not None and info.finished_at is not None:
            return info
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} never finished")
