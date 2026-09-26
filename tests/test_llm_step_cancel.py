"""Cancelling an AI correct or translate run stops it and saves nothing.

Both run in the task worker thread, not a subprocess, so nothing but the
batch callback can stop them: before it checked the cancel event, Cancel
only relabelled the task while every remaining batch ran (and billed) and
the result was saved as a new version.
"""

from __future__ import annotations

import time

import pytest

from podcodex.core.source import SourceVersion
from tests.fixtures.api_client import make_client
from tests.fixtures.llm import stub_llm_resolver

SEGS = [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hi"}]


@pytest.fixture
def client(tmp_path, monkeypatch):
    stub_llm_resolver(monkeypatch)
    return make_client(tmp_path, monkeypatch)


def _wait(task_id, timeout=5.0):
    from podcodex.api.tasks import task_manager

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = task_manager.get(task_id)
        if info is not None and info.finished_at is not None:
            return info
        time.sleep(0.02)
    raise AssertionError("task never finished")


@pytest.mark.parametrize("step", ["correct", "translate"])
def test_cancel_stops_the_run_before_it_saves(client, tmp_path, monkeypatch, step):
    from podcodex.api.routes import correct as correct_route
    from podcodex.api.routes import translate as translate_route
    from podcodex.api.tasks import task_manager
    import podcodex.core.correct as core_correct
    import podcodex.core.translate as core_translate

    audio = tmp_path / "show" / "ep.mp3"
    audio.parent.mkdir()
    audio.write_bytes(b"")
    source = SourceVersion(segments=SEGS, step="transcript", version_id="v1")
    route = correct_route if step == "correct" else translate_route
    monkeypatch.setattr(route, "load_source", lambda *_a, **_k: source)
    if step == "correct":
        monkeypatch.setattr(
            correct_route,
            "enrich_correct_kwargs",
            lambda *_a, **_k: {
                "source_lang": "English",
                "engine": "",
                "engine_model": "",
            },
        )

    batches_run: list[int] = []
    saved: list = []

    def fake_run(segments, *, on_batch, **_k):
        # The user cancels during the first batch.
        info = next(t for t in task_manager._tasks.values() if t.finished_at is None)
        task_manager.cancel(info.task_id)
        for n in (1, 2, 3):
            batches_run.append(n)
            on_batch(n, 3)
        return segments

    if step == "correct":
        monkeypatch.setattr(core_correct, "correct_segments", fake_run)
        monkeypatch.setattr(
            core_correct, "save_corrected", lambda *a, **k: saved.append(a)
        )
        body = {"audio_path": str(audio), "mode": "ollama"}
    else:
        monkeypatch.setattr(core_translate, "translate_segments", fake_run)
        monkeypatch.setattr(
            core_translate, "save_translation", lambda *a, **k: saved.append(a)
        )
        body = {"audio_path": str(audio), "mode": "ollama", "target_lang": "French"}

    r = client.post(f"/api/{step}/start", json=body)
    assert r.status_code == 200, r.text
    info = _wait(r.json()["task_id"])

    assert info.status == "cancelled"
    assert batches_run == [1]
    assert saved == []
    assert task_manager.get_active(str(audio)) is None


def test_a_cancel_during_the_last_batch_keeps_the_paid_for_run(
    client, tmp_path, monkeypatch
):
    from podcodex.api.routes import correct as correct_route
    from podcodex.api.tasks import task_manager
    import podcodex.core.correct as core_correct

    audio = tmp_path / "show" / "ep.mp3"
    audio.parent.mkdir()
    audio.write_bytes(b"")
    source = SourceVersion(segments=SEGS, step="transcript", version_id="v1")
    monkeypatch.setattr(correct_route, "load_source", lambda *_a, **_k: source)
    saved: list = []

    def fake_run(segments, *, on_batch, **_k):
        on_batch(1, 2)
        info = next(t for t in task_manager._tasks.values() if t.finished_at is None)
        task_manager.cancel(info.task_id)  # lands while the last batch runs
        on_batch(2, 2)
        return segments

    monkeypatch.setattr(core_correct, "correct_segments", fake_run)
    monkeypatch.setattr(
        core_correct, "save_corrected", lambda *a, **k: saved.append(a) or "vid"
    )
    r = client.post(
        "/api/correct/start", json={"audio_path": str(audio), "mode": "ollama"}
    )
    _wait(r.json()["task_id"])
    assert len(saved) == 1
