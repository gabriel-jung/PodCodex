"""TaskManager: submit, per-episode locking, cancel, stale cleanup.

The locks are what keep two runs off one episode's version dirs and
``pipeline.db`` rows, and what a delete or a move checks before touching a
folder. A regression there strands episodes locked until restart, or lets a
re-run overlap a subprocess still writing — and nothing else in the suite
exercises the manager directly.
"""

from __future__ import annotations

import threading
import time

import pytest

from podcodex.api.tasks import TaskManager
from tests.fixtures.tasks import active_task


@pytest.fixture
def tm():
    """A manager with its own worker pool, torn down after the test."""
    manager = TaskManager(max_workers=2)
    yield manager
    manager._executor.shutdown(wait=True)


def _run_to_completion(manager, info, timeout=5.0):
    deadline = time.monotonic() + timeout
    while info.finished_at is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert info.finished_at is not None, f"task {info.task_id} never finished"


def test_submit_runs_the_function_and_releases_the_lock(tm):
    done = threading.Event()

    def work(_progress_cb):
        done.set()
        return {"ok": True}

    info = tm.submit("transcribe", "/ep.mp3", work)
    _run_to_completion(tm, info)

    assert done.is_set()
    assert info.status == "completed"
    # Released, so the next run on the same episode is accepted.
    assert tm.get_active("/ep.mp3") is None
    assert tm.submit("transcribe", "/ep.mp3", work)


def test_a_second_task_on_the_same_path_is_refused(tm):
    started = threading.Event()
    release = threading.Event()

    def work(_progress_cb):
        started.set()
        release.wait(5.0)

    tm.submit("transcribe", "/ep.mp3", work)
    assert started.wait(5.0)
    try:
        with pytest.raises(ValueError):
            tm.submit("correct", "/ep.mp3", work)
        # A different episode is unaffected.
        assert tm.submit("correct", "/other.mp3", lambda _cb: None)
    finally:
        release.set()


def test_a_failing_task_still_releases_its_lock(tm):
    def boom(_progress_cb):
        raise RuntimeError("nope")

    info = tm.submit("index", "/ep.mp3", boom)
    _run_to_completion(tm, info)

    assert info.status == "failed"
    assert tm.get_active("/ep.mp3") is None


def test_cancel_keeps_the_lock_until_the_worker_finishes(tm):
    """The child keeps running for up to the grace + terminate window, so
    releasing in `cancel()` would let a re-run start alongside a subprocess
    still writing this episode's versions."""
    started = threading.Event()
    release = threading.Event()

    def work(progress_cb):
        started.set()
        release.wait(5.0)
        assert progress_cb.cancel_event.is_set()

    info = tm.submit("transcribe", "/ep.mp3", work)
    assert started.wait(5.0)

    assert tm.cancel(info.task_id) is True
    assert info.status == "cancelled"
    assert info.cancel_event.is_set()
    # Still locked, and still not stamped as finished.
    assert info.finished_at is None
    assert tm.get_active("/ep.mp3") is info

    release.set()
    _run_to_completion(tm, info)
    assert tm.get_active("/ep.mp3") is None


def test_cancel_is_idempotent_and_unknown_ids_are_false(tm):
    assert tm.cancel("nope_00000000") is False
    info = tm.submit("index", "/ep.mp3", lambda _cb: None)
    _run_to_completion(tm, info)
    assert tm.cancel(info.task_id) is True  # already finished, no-op


def test_cleanup_stale_does_not_steal_a_lock_held_by_a_newer_task(tm):
    """`run()` already releases a task's own lock, so by the time a task is
    stale the lock on that path belongs to whoever is running now. Popping
    by path evicted it, and the next submit on that path was accepted while
    a subprocess was still writing."""
    old = tm.submit("transcribe", "/ep.mp3", lambda _cb: None)
    _run_to_completion(tm, old)

    started = threading.Event()
    release = threading.Event()

    def work(_progress_cb):
        started.set()
        release.wait(5.0)

    new = tm.submit("correct", "/ep.mp3", work)
    assert started.wait(5.0)
    try:
        # Age the finished task past the window.
        old.finished_at = time.monotonic() - 10_000
        tm._cleanup_stale(max_age=600.0)

        assert tm.get(old.task_id) is None  # the stale record is gone
        assert tm.get_active("/ep.mp3") is new  # the live lock is not
        with pytest.raises(ValueError):
            tm.submit("index", "/ep.mp3", work)
    finally:
        release.set()


def test_cleanup_stale_releases_a_lock_no_one_else_took(tm):
    info = tm.submit("index", "/ep.mp3", lambda _cb: None)
    _run_to_completion(tm, info)
    tm.lock("/ep.mp3", info.task_id)  # simulate a run() that never released

    info.finished_at = time.monotonic() - 10_000
    tm._cleanup_stale(max_age=600.0)

    assert tm.get_active("/ep.mp3") is None


# ``active_task`` registers on the process-wide ``task_manager`` (that is the
# instance the routes hold), so the query tests below read from it too.


def test_get_active_ignores_a_finished_task_that_still_holds_the_lock():
    from podcodex.api.tasks import task_manager

    with active_task("/ep.mp3", finished=True):
        assert task_manager.get_active("/ep.mp3") is None


def test_get_active_in_show_matches_every_lock_shape():
    """Each submit path mints its own key; a delete or move has to see all
    of them or it runs `rmtree` underneath a live job."""
    from podcodex.api.tasks import task_manager

    for key in (
        "/library/show",
        "batch:/library/show",
        "download:/library/show",
        "/library/show/ep1.mp3",
        "/library/show/ep1.virtual",
    ):
        with active_task(key):
            assert task_manager.get_active_in_show("/library/show") is not None, key
    assert task_manager.get_active_in_show("/library/show") is None


def test_get_active_in_show_ignores_a_sibling_folder():
    from podcodex.api.tasks import task_manager

    with active_task("/library/other/ep1.mp3"):
        assert task_manager.get_active_in_show("/library/show") is None
