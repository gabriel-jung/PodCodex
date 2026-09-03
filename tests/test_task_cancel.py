"""A task cancelled before a worker picks it up must not run.

The pool is bounded, so a submitted task can sit queued long enough to be
cancelled first. Starting it anyway ran work the user had called off and
flipped the status back to "running" with nothing left to move it on.
"""

from __future__ import annotations

import threading
import time

from podcodex.api.tasks import TaskManager


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_cancel_while_queued_skips_the_work_and_frees_the_lock():
    mgr = TaskManager(max_workers=1)
    started = threading.Event()
    release = threading.Event()
    ran_second = threading.Event()

    def blocker(progress_cb):
        started.set()
        release.wait(timeout=5.0)

    def never(progress_cb):
        ran_second.set()

    first = mgr.submit("transcribe", "/tmp/a.mp3", blocker)
    assert _wait_until(started.is_set)

    # Queued behind the single worker, then cancelled before it starts.
    second = mgr.submit("transcribe", "/tmp/b.mp3", never)
    assert mgr.cancel(second.task_id)

    release.set()
    assert _wait_until(lambda: second.finished_at is not None)

    assert not ran_second.is_set(), "cancelled task ran its work anyway"
    assert second.status == "cancelled"
    # The lock is gone, so the same episode can be submitted again.
    assert mgr.get_active("/tmp/b.mp3") is None
    assert _wait_until(lambda: first.finished_at is not None)


def test_a_queued_cancel_does_not_block_a_show_delete():
    """get_active_in_show gates on finished_at, so the release has to happen."""
    mgr = TaskManager(max_workers=1)
    started = threading.Event()
    release = threading.Event()

    def blocker(progress_cb):
        started.set()
        release.wait(timeout=5.0)

    first = mgr.submit("batch", "batch:/shows/MyShow", blocker)
    assert _wait_until(started.is_set)
    second = mgr.submit("transcribe", "/shows/MyShow/ep.mp3", lambda cb: None)
    mgr.cancel(second.task_id)

    release.set()
    assert _wait_until(lambda: first.finished_at is not None)
    assert _wait_until(lambda: second.finished_at is not None)
    assert mgr.get_active_in_show("/shows/MyShow") is None
