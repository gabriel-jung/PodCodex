"""Tests for the subprocess-based pipeline runner."""

from __future__ import annotations

import threading
import time

import pytest

from podcodex.api.subprocess_runner import run_in_subprocess


def test_basic_run_and_progress():
    events: list[tuple[float, str]] = []
    result = run_in_subprocess(
        "tests.fixtures.subprocess_jobs:add",
        {"a": 7, "b": 3},
        on_progress=lambda f, m: events.append((f, m)),
    )
    assert result == 10
    assert events, "expected at least one progress event"
    assert any(m == "done" for _, m in events)


def test_error_is_propagated():
    with pytest.raises(RuntimeError) as exc_info:
        run_in_subprocess("tests.fixtures.subprocess_jobs:boom", {})
    assert "explode" in str(exc_info.value)


def test_cancel_event_stops_child():
    cancel_ev = threading.Event()

    def fire_cancel():
        time.sleep(0.3)
        cancel_ev.set()

    t = threading.Thread(target=fire_cancel, daemon=True)
    t.start()
    start = time.monotonic()
    # Child polls cancelled() every 50ms; should exit quickly after signal.
    result = run_in_subprocess(
        "tests.fixtures.subprocess_jobs:slow",
        {},
        cancel_event=cancel_ev,
    )
    elapsed = time.monotonic() - start
    assert result == "cancelled"
    # Floor is the child's bootstrap time (~1–2 s on Linux CI) before it
    # reaches the cancel poll loop; bound generously to stay non-flaky.
    assert elapsed < 6.0, f"took too long to honor cancel: {elapsed:.2f}s"
    t.join(timeout=1)


def test_bad_entry_path_returns_error():
    with pytest.raises(RuntimeError) as exc_info:
        run_in_subprocess("podcodex.does_not_exist:nope", {})
    # Either import error or attribute error — both surface via the err channel.
    msg = str(exc_info.value)
    assert "ModuleNotFoundError" in msg or "AttributeError" in msg


def _run_bounded(entry: str, kwargs: dict, timeout: float = 60.0):
    """Run in a thread so a deadlock fails the test instead of hanging it."""
    box: dict = {}

    def target():
        try:
            box["value"] = run_in_subprocess(entry, kwargs)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "run_in_subprocess deadlocked on a large payload"
    return box


def test_a_result_larger_than_the_pipe_buffer_is_returned():
    """The child cannot exit until its result is flushed; reading only after
    the exit deadlocked any payload over the pipe buffer."""
    box = _run_bounded("tests.fixtures.subprocess_jobs:big_result", {"size": 1_000_000})
    assert box.get("value") == "x" * 1_000_000, box.get("error")


def test_a_long_error_is_reported_and_truncated():
    box = _run_bounded("tests.fixtures.subprocess_jobs:big_boom", {"size": 200_000})
    err = box.get("error")
    assert isinstance(err, RuntimeError)
    assert "explode" in str(err)
    assert len(str(err)) < 40_000


def test_a_child_does_not_inherit_the_api_thread_cap(monkeypatch):
    """The API defaults OMP_NUM_THREADS=1 for its own threads; a step child
    running on one core made diarization and TTS several times slower."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("_PODCODEX_OMP_DEFAULTED", "1")
    assert run_in_subprocess("tests.fixtures.subprocess_jobs:omp_threads", {}) is None


def test_a_user_set_thread_cap_reaches_the_child(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("_PODCODEX_OMP_DEFAULTED", raising=False)
    assert run_in_subprocess("tests.fixtures.subprocess_jobs:omp_threads", {}) == "3"
