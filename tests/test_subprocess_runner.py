"""Tests for the subprocess-based pipeline runner."""

from __future__ import annotations

import threading
import time

import pytest

from podcodex.api.subprocess_runner import run_in_subprocess


def test_cancel_event_stops_child():
    cancel_ev = threading.Event()
    cancelled_at: list[float] = []

    def on_progress(_frac, _msg):
        # Cancel once the child runs: a fixed delay raced its start-up, which
        # on a loaded machine outlasted the cancel grace and got it killed.
        if not cancel_ev.is_set():
            cancelled_at.append(time.monotonic())
            cancel_ev.set()

    result = run_in_subprocess(
        "tests.fixtures.subprocess_jobs:slow",
        {},
        on_progress=on_progress,
        cancel_event=cancel_ev,
    )
    assert result == "cancelled"
    # The child polls cancelled() every 50 ms.
    assert time.monotonic() - cancelled_at[0] < 3.0


@pytest.mark.parametrize("entry", ["stubborn", "raises_on_cancel"])
def test_a_cancelled_child_raises_task_cancelled_not_an_error(entry):
    """A batch counts any other exception as a failed episode."""
    from podcodex.api.tasks import TaskCancelled

    cancel_ev = threading.Event()
    threading.Timer(0.3, cancel_ev.set).start()

    with pytest.raises(TaskCancelled):
        run_in_subprocess(
            f"tests.fixtures.subprocess_jobs:{entry}", {}, cancel_event=cancel_ev
        )


def test_a_real_failure_during_a_cancel_is_still_an_error():
    """A crash in the grace window must not pass for the user's cancel."""
    cancel_ev = threading.Event()
    threading.Timer(0.3, cancel_ev.set).start()
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run_in_subprocess(
            "tests.fixtures.subprocess_jobs:crashes_on_cancel",
            {},
            cancel_event=cancel_ev,
        )


def test_model_jobs_run_one_at_a_time():
    """Two at once overflow a consumer GPU; the second waits its turn."""
    spans: list[tuple[float, float]] = []
    runs = [
        threading.Thread(
            target=lambda: spans.append(
                run_in_subprocess("tests.fixtures.subprocess_jobs:timed", {})
            )
        )
        for _ in range(2)
    ]
    for t in runs:
        t.start()
    for t in runs:
        t.join(timeout=60)

    first, second = sorted(spans)
    assert second[0] >= first[1]


def test_a_job_waiting_for_the_model_slot_can_be_cancelled():
    from podcodex.api import subprocess_runner
    from podcodex.api.tasks import TaskCancelled

    cancel_ev = threading.Event()
    messages: list[str] = []
    threading.Timer(0.3, cancel_ev.set).start()
    with subprocess_runner._MODEL_SLOT:
        with pytest.raises(TaskCancelled):
            run_in_subprocess(
                "tests.fixtures.subprocess_jobs:slow",
                {},
                on_progress=lambda _f, m: messages.append(m),
                cancel_event=cancel_ev,
            )
    assert messages == ["Waiting for another job to finish..."]


def test_bad_entry_path_returns_error():
    with pytest.raises(RuntimeError) as exc_info:
        run_in_subprocess("podcodex.does_not_exist:nope", {})
    # Either import error or attribute error — both surface via the err channel.
    msg = str(exc_info.value)
    assert "ModuleNotFoundError" in msg or "AttributeError" in msg


def _run_bounded(entry: str, kwargs: dict, timeout: float = 60.0, on_progress=None):
    """Run in a thread so a deadlock fails the test instead of hanging it."""
    box: dict = {}

    def target():
        try:
            box["value"] = run_in_subprocess(entry, kwargs, on_progress=on_progress)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "run_in_subprocess deadlocked on a large payload"
    return box


def test_kwargs_progress_and_a_result_larger_than_the_pipe_buffer_come_back():
    """The child cannot exit until its result is flushed; reading only after
    the exit would deadlock on any payload over the pipe buffer."""
    events: list[tuple[float, str]] = []
    box = _run_bounded(
        "tests.fixtures.subprocess_jobs:big_result",
        {"size": 1_000_000},
        on_progress=lambda f, m: events.append((f, m)),
    )
    assert box.get("value") == "x" * 1_000_000, box.get("error")
    assert (0.5, "halfway") in events


def test_a_long_error_is_reported_and_truncated():
    box = _run_bounded("tests.fixtures.subprocess_jobs:big_boom", {"size": 200_000})
    err = box.get("error")
    assert isinstance(err, RuntimeError)
    assert "explode" in str(err)
    assert len(str(err)) < 40_000


def test_a_child_does_not_inherit_the_api_thread_cap(monkeypatch):
    """The API defaults OMP_NUM_THREADS=1 for its own threads; a step child
    running on one core makes diarization and TTS several times slower."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("_PODCODEX_OMP_DEFAULTED", "1")
    assert run_in_subprocess("tests.fixtures.subprocess_jobs:omp_threads", {}) is None


def test_only_the_api_default_thread_cap_is_dropped():
    """A cap the user set themselves reaches the child untouched."""
    from podcodex.api.subprocess_runner import _drop_inherited_thread_cap

    defaulted = {"OMP_NUM_THREADS": "1", "_PODCODEX_OMP_DEFAULTED": "1"}
    _drop_inherited_thread_cap(defaulted)
    user_set = {"OMP_NUM_THREADS": "3"}
    _drop_inherited_thread_cap(user_set)

    assert defaulted == {}
    assert user_set == {"OMP_NUM_THREADS": "3"}


def test_a_child_cancel_keeps_its_summary():
    from podcodex.api.tasks import TaskCancelled

    cancel_ev = threading.Event()
    threading.Timer(0.3, cancel_ev.set).start()
    with pytest.raises(TaskCancelled) as exc:
        run_in_subprocess(
            "tests.fixtures.subprocess_jobs:raises_on_cancel",
            {},
            cancel_event=cancel_ev,
        )
    assert exc.value.summary == "2 indexed"
