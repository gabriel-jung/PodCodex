"""TaskManager: submit, per-episode locking, cancel, stale cleanup.

The locks are what keep two runs off one episode's version dirs and
``pipeline.db`` rows, and what a delete or a move checks before touching a
folder. A regression there strands episodes locked until restart, or lets a
re-run overlap a subprocess still writing — and nothing else in the suite
exercises the manager directly.
"""

from __future__ import annotations

import threading
from pathlib import Path
import time

import pytest

from podcodex.api.tasks import TaskManager
from tests.fixtures.tasks import active_task


def _manager(**kwargs) -> TaskManager:
    """A manager with no WebSocket clients to tell: outside the app there is
    no event loop to broadcast on, and every progress tick would log so."""
    manager = TaskManager(**kwargs)
    manager._broadcast_sync = lambda _task_id: None
    return manager


@pytest.fixture
def tm():
    """A manager with its own worker pool, torn down after the test."""
    manager = _manager(max_workers=2)
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
    # Clients see the lock as `stopping`, so they wait before a re-run.
    assert info.snapshot()["stopping"] is True

    release.set()
    _run_to_completion(tm, info)
    assert tm.get_active("/ep.mp3") is None
    assert info.snapshot()["stopping"] is False


def test_cancel_is_idempotent_and_unknown_ids_are_false(tm):
    assert tm.cancel("nope_00000000") is False
    info = tm.submit("index", "/ep.mp3", lambda _cb: None)
    _run_to_completion(tm, info)
    assert tm.cancel(info.task_id) is True  # already finished, no-op


def test_cleanup_stale_does_not_steal_a_lock_held_by_a_newer_task(tm):
    """`run()` already releases a task's own lock, so by the time a task is
    stale the lock on that path belongs to whoever is running now. Popping
    by path would evict it, and the next submit on that path would be
    accepted while a subprocess is still writing."""
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


def test_concurrent_submits_on_one_path_start_one_task(tm):
    """A double-clicked Start runs two submits on the threadpool at once."""
    release = threading.Event()
    barrier = threading.Barrier(8)
    accepted: list = []
    refused: list = []

    def work(_progress_cb):
        release.wait(5.0)

    def racer():
        barrier.wait()
        try:
            accepted.append(tm.submit("transcribe", "/ep.mp3", work))
        except ValueError:
            refused.append(1)

    threads = [threading.Thread(target=racer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    release.set()

    assert len(accepted) == 1
    assert len(refused) == 7
    _run_to_completion(tm, accepted[0])


def test_lock_refuses_a_live_holder(tm):
    release = threading.Event()
    info = tm.submit("transcribe", "/ep.mp3", lambda _cb: release.wait(5.0))
    try:
        assert tm.lock("/ep.mp3", "batch_x") is False
        assert tm.get_active("/ep.mp3") is info
    finally:
        release.set()
        _run_to_completion(tm, info)
    assert tm.lock("/ep.mp3", "batch_x") is True


def test_cancel_all_stops_running_and_drops_queued_tasks():
    manager = _manager(max_workers=1)
    started = threading.Event()

    def work(progress_cb):
        started.set()
        progress_cb.cancel_event.wait(5.0)

    running = manager.submit("transcribe", "/a.mp3", work)
    started.wait(5.0)
    queued = manager.submit("transcribe", "/b.mp3", work)

    assert manager.cancel_all() == 2
    _run_to_completion(manager, running)
    assert queued.finished_at is not None
    assert manager.get_active("/b.mp3") is None
    # The pool survives: the app can enter its lifespan again.
    again = manager.submit("transcribe", "/b.mp3", lambda _cb: None)
    _run_to_completion(manager, again)
    manager._executor.shutdown(wait=True)


@pytest.fixture
def solo():
    """One worker, so a second submit sits queued behind the first."""
    manager = _manager(max_workers=1)
    yield manager
    manager._executor.shutdown(wait=True)


def test_cancel_while_queued_skips_the_work_and_frees_the_lock(solo):
    """The pool is bounded, so a task can be cancelled before a worker picks
    it up. Starting it anyway would run work the user called off, and a
    lock left behind would block the show's delete (get_active_in_show gates
    on finished_at)."""
    started = threading.Event()
    release = threading.Event()
    ran_second = threading.Event()

    def blocker(_progress_cb):
        started.set()
        release.wait(timeout=5.0)

    first = solo.submit("batch", "batch:/shows/A", blocker)
    assert started.wait(5.0)
    second = solo.submit("transcribe", "/shows/B/ep.mp3", lambda _cb: ran_second.set())
    assert solo.cancel(second.task_id)
    # Settled at once, while the worker is still busy: a client waiting on
    # `stopping` must not wait for an unrelated task to finish.
    assert second.snapshot()["stopping"] is False
    assert solo.get_active("/shows/B/ep.mp3") is None

    release.set()
    _run_to_completion(solo, second)
    _run_to_completion(solo, first)

    assert not ran_second.is_set(), "cancelled task ran its work anyway"
    assert second.status == "cancelled"
    assert solo.get_active("/shows/B/ep.mp3") is None
    assert solo.get_active_in_show("/shows/B") is None


@pytest.mark.parametrize(
    "summary, message",
    [("2 downloaded", "Cancelled: 2 downloaded"), (None, "Cancelled")],
)
def test_a_cancelled_task_keeps_the_summary_it_reported(solo, summary, message):
    """A download loop reports what it finished before stopping; replacing
    that with a bare "Cancelled" hid it from the user."""
    from podcodex.api.tasks import TaskCancelled

    started = threading.Event()

    def loop(progress_cb):
        started.set()
        progress_cb.cancel_event.wait(timeout=5.0)
        if summary:
            raise TaskCancelled(summary)

    info = solo.submit("download", "/tmp/show", loop)
    assert started.wait(5.0)
    solo.cancel(info.task_id)
    _run_to_completion(solo, info)

    assert info.status == "cancelled"
    assert info.message == message


def test_submit_task_refuses_a_different_step_holding_the_episode(monkeypatch):
    """Returning the other step's task made the panel report it as its own."""
    from fastapi import HTTPException

    from podcodex.api import tasks as tasks_mod
    from podcodex.api.routes._helpers import submit_task

    manager = _manager(max_workers=1)
    monkeypatch.setattr(tasks_mod, "task_manager", manager)
    release = threading.Event()
    running = manager.submit("transcribe", "/ep.mp3", lambda _cb: release.wait(5.0))
    try:
        # Same step reconnects to the running task.
        assert submit_task("transcribe", "/ep.mp3", lambda _cb: None).task_id == (
            running.task_id
        )
        with pytest.raises(HTTPException) as exc:
            submit_task("correct", "/ep.mp3", lambda _cb: None)
        assert exc.value.status_code == 409
        assert "transcribe" in exc.value.detail
    finally:
        release.set()
        _run_to_completion(manager, running)
        manager._executor.shutdown(wait=True)


def test_downloads_do_not_take_the_compute_slots():
    """Two long downloads must not leave a transcribe pending."""
    manager = _manager(max_workers=1, io_workers=2)
    release = threading.Event()
    started = threading.Event()
    downloads = [
        manager.submit("download", f"download:/show{i}", lambda _cb: release.wait(5.0))
        for i in range(2)
    ]
    compute = manager.submit("transcribe", "/ep.mp3", lambda _cb: started.set())
    try:
        assert started.wait(2.0), "transcribe queued behind the downloads"
    finally:
        release.set()
        for info in [*downloads, compute]:
            _run_to_completion(manager, info)
        manager._executor.shutdown(wait=True)
        manager._io_executor.shutdown(wait=True)


def test_submit_task_does_not_hand_one_language_to_another(monkeypatch):
    """A Spanish translate must not follow the running French one."""
    from fastapi import HTTPException

    from podcodex.api import tasks as tasks_mod
    from podcodex.api.routes._helpers import submit_task

    manager = _manager(max_workers=1)
    monkeypatch.setattr(tasks_mod, "task_manager", manager)
    release = threading.Event()
    french = submit_task(
        "translate", "/ep.mp3", lambda _cb: release.wait(5.0), subject="french"
    )
    try:
        again = submit_task("translate", "/ep.mp3", lambda _cb: None, subject="french")
        assert again.task_id == french.task_id
        with pytest.raises(HTTPException) as exc:
            submit_task("translate", "/ep.mp3", lambda _cb: None, subject="spanish")
        assert exc.value.status_code == 409
        assert "french translate" in exc.value.detail
    finally:
        release.set()
        _run_to_completion(manager, manager.get(french.task_id))
        manager._executor.shutdown(wait=True)


def test_a_result_can_close_with_its_own_summary(tm):
    """An LLM run whose batches were all rejected used to end as "Done"."""
    plain = tm.submit("index", "/a.mp3", lambda _cb: {"count": 1})
    summary = tm.submit(
        "translate",
        "/b.mp3",
        lambda _cb: {"count": 1, "message": "Done, 1 of 1 rejected"},
    )
    _run_to_completion(tm, plain)
    _run_to_completion(tm, summary)
    assert plain.message == "Done"
    assert summary.message == "Done, 1 of 1 rejected"


def test_download_ticks_stay_out_of_the_steps(tm):
    """A download reports once per MB; as steps that grew the polled task
    payload by thousands of lines. Ticks go to the bounded log only."""

    def work(progress_cb):
        progress_cb(0.0, "Fetching manifest…")
        for mb in range(1, 301):
            progress_cb.tick(mb / 300, f"server core: {mb} / 300 MB")
        progress_cb(0.9, "Extracting server core…")
        progress_cb.tick(0.95, "Extracting server core… 50%")
        progress_cb(0.97, "Model downloaded: 512 MB")
        progress_cb(1.0, "Installed.")
        return {}

    info = tm.submit("gpu_download", "gpu_backend", work)
    _run_to_completion(tm, info)

    assert info.steps == [
        "Fetching manifest…",
        "Extracting server core…",
        "Model downloaded: 512 MB",
        "Installed.",
    ]
    assert len(info.log) <= 200


@pytest.fixture
def stdlib_intercept():
    """The app's stdlib-to-loguru routing, undone afterwards: it rewires the
    root logger and several named ones for the whole process."""
    import logging

    from podcodex import bootstrap

    names = (
        "",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "mcp",
        "FastMCP",
        "mcp.server.streamable_http_manager",
        "mcp.server.lowlevel",
        "httpx",
        "httpcore",
    )
    saved = {
        n: (
            logging.getLogger(n).handlers[:],
            logging.getLogger(n).propagate,
            logging.getLogger(n).level,
        )
        for n in names
    }
    bootstrap._install_stdlib_intercept()
    yield
    for n, (handlers, propagate, level) in saved.items():
        lg = logging.getLogger(n)
        lg.handlers[:] = handlers
        lg.propagate = propagate
        lg.setLevel(level)


@pytest.mark.usefixtures("stdlib_intercept")
def test_a_stdlib_log_line_reaches_the_task_log_once(tm):
    """The app forwards stdlib logging into loguru, and a task captures both:
    httpx's request lines used to show up twice."""
    import logging

    info = tm.submit(
        "correct",
        "/ep.mp3",
        lambda _cb: logging.getLogger("podcodex.x").info("HTTP Request: GET /api/tags"),
    )
    _run_to_completion(tm, info)

    lines = [line for line in info.log if "HTTP Request" in line]
    assert lines == ["podcodex.x: HTTP Request: GET /api/tags"]


@pytest.mark.usefixtures("stdlib_intercept")
def test_a_non_propagating_logger_still_reaches_the_task_log(tm):
    """uvicorn and mcp loggers skip root, so loguru is their only way in."""
    import logging

    info = tm.submit(
        "correct", "/ep.mp3", lambda _cb: logging.getLogger("mcp").info("mcp says hi")
    )
    _run_to_completion(tm, info)

    assert sum("mcp says hi" in line for line in info.log) == 1


@pytest.mark.usefixtures("stdlib_intercept")
def test_httpx_request_lines_are_not_logged():
    """httpx logs full request URLs at INFO, a private feed's token included."""
    import logging

    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
    assert not logging.getLogger("httpcore").isEnabledFor(logging.INFO)


def test_get_active_in_show_reads_keys_by_the_path_they_name(tmp_path):
    """Move and delete use it: a run locked under another spelling of the
    show folder (trailing slash, ``..``) must still be seen."""
    from podcodex.api.tasks import task_manager

    show = tmp_path / "Show"
    show.mkdir()
    with active_task(f"download:{show}/"):
        assert task_manager.get_active_in_show(str(show)) is not None
    with active_task(f"{tmp_path}/x/../Show/ep1.mp3"):
        assert task_manager.get_active_in_show(str(show)) is not None
    with active_task(str(tmp_path / "Other" / "ep1.mp3")):
        assert task_manager.get_active_in_show(str(show)) is None


def test_a_symlinked_show_folder_still_matches_its_episode_locks(tmp_path):
    """Registered through a link, locked through the same link: the show
    root and the keys under it must normalize alike."""
    from podcodex.api.tasks import task_manager

    real = tmp_path / "ext" / "Show"
    real.mkdir(parents=True)
    link = tmp_path / "Show"
    link.symlink_to(real)
    with active_task(str(link / "ep1.mp3")):
        assert task_manager.get_active_in_show(str(link)) is not None
        assert task_manager.get_active_in_show(str(real)) is not None


def test_lock_path_collapses_dots_and_leaves_non_paths_alone(tmp_path, monkeypatch):
    from podcodex.api.tasks import lock_path

    show = tmp_path / "Show"
    (show / "ep1").mkdir(parents=True)
    assert lock_path(f"{show}/ep1/..") == lock_path(show)
    # A key naming no folder is never resolved against the cwd, which could
    # put it inside a show and make that show read as busy.
    monkeypatch.chdir(show)
    assert lock_path("gpu_backend") == Path("gpu_backend")


def test_a_symlinked_episode_folder_stays_inside_its_show(tmp_path):
    """A key naming a symlinked output dir must not follow the link out of
    the show, or a show delete would rmtree under the live job."""
    from podcodex.api.tasks import task_manager

    show = tmp_path / "Show"
    show.mkdir()
    elsewhere = tmp_path / "ext" / "ep1"
    elsewhere.mkdir(parents=True)
    (show / "ep1").symlink_to(elsewhere)
    with active_task(str(show / "ep1")):
        assert task_manager.get_active_in_show(str(show)) is not None
