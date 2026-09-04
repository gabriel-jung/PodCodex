"""`_run_batch`: locking, skipping, cancellation and the reported tally.

The loop itself owns four things nothing else does — it takes a per-episode
lock under the *batch* task's id, skips an episode another task holds, checks
the cancel event between steps, and counts what happened for the UI. Each has
regressed before, and every step helper below it is slow enough that the only
practical way to pin the loop is to stub them out and record the calls.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

import podcodex.api.routes.batch as batch_mod  # noqa: E402
from tests.fixtures.tasks import active_task  # noqa: E402


@pytest.fixture
def calls(monkeypatch):
    """Stub every step helper; returns the recorded (step, audio_path) list."""
    recorded: list[tuple[str, str]] = []

    def _record(step, *, works=True):
        def fn(audio_path, *_a, **_kw):
            recorded.append((step, audio_path))
            return works

        return fn

    monkeypatch.setattr(batch_mod, "_batch_transcribe", _record("transcribe"))
    monkeypatch.setattr(
        batch_mod, "_batch_transcribe_from_subs", _record("transcribe_subs")
    )
    monkeypatch.setattr(batch_mod, "_batch_index", _record("index"))

    def llm(audio_path, _p, _req, *_a, step="correct", **_kw):
        recorded.append((step, audio_path))
        return True

    monkeypatch.setattr(batch_mod, "_batch_llm_step", llm)
    monkeypatch.setattr(
        batch_mod, "invalidate_scan_cache", lambda _p: None, raising=False
    )
    return recorded


class _Progress:
    """The callback tasks.py hands the runner, plus its cancel_event."""

    def __init__(self, cancel_event=None):
        self.messages: list[tuple[float, str]] = []
        self.cancel_event = cancel_event

    def __call__(self, frac, message):
        self.messages.append((frac, message))


def _req(tmp_path, paths, **over):
    from podcodex.api.routes.batch import BatchRequest

    fields = {
        "show_folder": str(tmp_path),
        "audio_paths": [str(p) for p in paths],
        "transcribe": False,
        "correct": True,
        "translate": False,
        "index": False,
        "show_name": "Show",
        **over,
    }
    return BatchRequest(**fields)


def _episode(tmp_path, stem):
    path = tmp_path / f"{stem}.mp3"
    path.write_bytes(b"x")
    return path


def test_every_episode_runs_and_the_tally_matches(tmp_path, calls):
    eps = [_episode(tmp_path, "a"), _episode(tmp_path, "b")]
    out = batch_mod._run_batch(_Progress(), _req(tmp_path, eps))

    assert [c[0] for c in calls] == ["correct", "correct"]
    assert out == {
        "total": 2,
        "completed": 2,
        "failed": 0,
        "skipped": 0,
        "errors": [],
    }


def test_an_episode_locked_by_another_task_is_skipped_not_run(tmp_path, calls):
    """Running it anyway would put two writers on one stem's version dirs."""
    eps = [_episode(tmp_path, "a"), _episode(tmp_path, "b")]

    with active_task(str(eps[0])):
        out = batch_mod._run_batch(_Progress(), _req(tmp_path, eps))

    assert [c[1] for c in calls] == [str(eps[1])]
    assert out["skipped"] == 1
    assert out["completed"] == 1


def test_the_lock_is_released_even_when_a_step_raises(tmp_path, calls, monkeypatch):
    from podcodex.api.tasks import task_manager

    eps = [_episode(tmp_path, "a"), _episode(tmp_path, "b")]

    def boom(audio_path, *_a, **_kw):
        if audio_path == str(eps[0]):
            raise RuntimeError("step exploded")
        return True

    monkeypatch.setattr(batch_mod, "_batch_llm_step", boom)

    out = batch_mod._run_batch(_Progress(), _req(tmp_path, eps))

    assert out["failed"] == 1
    assert out["completed"] == 1
    assert out["errors"] == [{"episode": "a", "error": "step exploded"}]
    # A failed episode must not strand its lock; the next run would 409.
    assert task_manager.get_active(str(eps[0])) is None
    assert task_manager.get_active(str(eps[1])) is None


def test_an_episode_with_nothing_to_do_counts_as_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(
        batch_mod, "_batch_llm_step", lambda *_a, step="correct", **_kw: False
    )
    eps = [_episode(tmp_path, "a")]

    out = batch_mod._run_batch(_Progress(), _req(tmp_path, eps))

    assert out["skipped"] == 1
    assert out["completed"] == 0


def test_cancelling_stops_the_loop_before_the_next_episode(tmp_path, monkeypatch):
    import threading

    eps = [_episode(tmp_path, "a"), _episode(tmp_path, "b"), _episode(tmp_path, "c")]
    cancel = threading.Event()
    seen: list[str] = []

    def llm(audio_path, *_a, step="correct", **_kw):
        seen.append(audio_path)
        cancel.set()  # cancelled while the first episode is in flight
        return True

    monkeypatch.setattr(batch_mod, "_batch_llm_step", llm)
    progress = _Progress(cancel_event=cancel)

    out = batch_mod._run_batch(progress, _req(tmp_path, eps))

    assert seen == [str(eps[0])]
    assert out["total"] == 3
    assert out["completed"] == 1
    assert any("Cancel" in m for _f, m in progress.messages)


def test_disabled_steps_are_not_run(tmp_path, calls):
    eps = [_episode(tmp_path, "a")]

    batch_mod._run_batch(_Progress(), _req(tmp_path, eps, correct=False, index=True))

    assert [c[0] for c in calls] == ["index"]


def test_transcribe_is_skipped_for_an_episode_with_no_audio(tmp_path, calls):
    """A subtitle-only episode has no file to feed WhisperX; the later steps
    still run off its output dir."""
    missing = tmp_path / "gone.mp3"

    batch_mod._run_batch(
        _Progress(), _req(tmp_path, [missing], transcribe=True, correct=True)
    )

    assert [c[0] for c in calls] == ["correct"]
