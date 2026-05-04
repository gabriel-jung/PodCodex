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
