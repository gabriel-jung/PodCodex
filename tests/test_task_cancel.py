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


def test_a_cancelled_task_keeps_the_summary_it_reported():
    """A download loop reports what it finished before stopping; replacing
    that with a bare "Cancelled" hid it from the user."""
    mgr = TaskManager(max_workers=1)
    started = threading.Event()

    def loop(progress_cb):
        from podcodex.api.tasks import TaskCancelled

        started.set()
        progress_cb.cancel_event.wait(timeout=5.0)
        raise TaskCancelled("2 downloaded")

    info = mgr.submit("download", "/tmp/show", loop)
    assert _wait_until(started.is_set)
    mgr.cancel(info.task_id)
    assert _wait_until(lambda: info.finished_at is not None)

    assert info.status == "cancelled"
    assert info.message == "Cancelled: 2 downloaded"


def test_a_cancelled_task_without_a_summary_reads_cancelled():
    mgr = TaskManager(max_workers=1)
    started = threading.Event()

    def quiet(progress_cb):
        started.set()
        progress_cb.cancel_event.wait(timeout=5.0)

    info = mgr.submit("download", "/tmp/show2", quiet)
    assert _wait_until(started.is_set)
    mgr.cancel(info.task_id)
    assert _wait_until(lambda: info.finished_at is not None)

    assert info.message == "Cancelled"


def test_cancelling_an_extras_install_stops_uv():
    """uv used to run to the end whatever the user clicked."""
    import sys

    import pytest

    from podcodex.api.routes._helpers import TaskCancelled
    from podcodex.api.routes.health import _run_uv

    cancel = threading.Event()

    def progress_cb(_frac, msg):
        if msg == "working":
            cancel.set()  # cancelled while uv is mid-run

    progress_cb.cancel_event = cancel
    slow = [
        sys.executable,
        "-c",
        "import time; print('working', flush=True); time.sleep(30)",
    ]
    import podcodex.api.routes.health as health

    refreshed: list[bool] = []
    real = health._invalidate_capabilities
    health._invalidate_capabilities = lambda: refreshed.append(True)
    try:
        start = time.monotonic()
        with pytest.raises(TaskCancelled):
            _run_uv(slow, progress_cb, "Installing")
    finally:
        health._invalidate_capabilities = real
    assert time.monotonic() - start < 10
    # uv may have changed packages before it stopped.
    assert refreshed
