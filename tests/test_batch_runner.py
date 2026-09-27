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
from podcodex.core.versions import save_version  # noqa: E402
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

    from podcodex.api.tasks import TaskCancelled

    with pytest.raises(TaskCancelled) as stopped:
        batch_mod._run_batch(progress, _req(tmp_path, eps))

    assert seen == [str(eps[0])]
    # What finished before the stop becomes the task's final message.
    assert stopped.value.summary == "1 completed"
    assert any("Cancel" in m for _f, m in progress.messages)


def test_each_episode_runs_on_the_version_picked_for_it(tmp_path, monkeypatch):
    """``source_version_ids`` is keyed like ``audio_paths``, virtual keys
    included; a loop that drops it silently runs every episode on a version
    the user did not pick."""
    from podcodex.core._utils import virtual_audio_path

    seen: dict[tuple[str, str], str | None] = {}

    def llm(audio_path, *_a, step="correct", version_id=None, **_kw):
        seen[(step, audio_path)] = version_id
        return True

    def index(audio_path, *_a, version_id=None, **_kw):
        seen[("index", audio_path)] = version_id
        return True

    monkeypatch.setattr(batch_mod, "_batch_llm_step", llm)
    monkeypatch.setattr(batch_mod, "_batch_index", index)
    audio = str(_episode(tmp_path, "a"))
    virtual = virtual_audio_path(tmp_path / "b")
    req = _req(tmp_path, [audio, virtual], index=True)
    req.source_version_ids = {audio: "v-a", virtual: "v-b"}

    batch_mod._run_batch(_Progress(), req)

    assert seen == {
        ("correct", audio): "v-a",
        ("index", audio): "v-a",
        ("correct", virtual): "v-b",
        ("index", virtual): "v-b",
    }


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


def test_an_up_to_date_episode_spawns_no_child(monkeypatch):
    """The skip check runs in the parent; a spawn costs seconds of imports."""
    from types import SimpleNamespace

    import podcodex.api.subprocess_runner as runner
    import podcodex.core.transcribe_job as tjob
    import podcodex.rag.index_job as ijob
    from podcodex.api.routes import batch

    monkeypatch.setattr(
        runner, "run_in_subprocess", lambda **_k: pytest.fail("spawned a child")
    )
    monkeypatch.setattr(tjob, "transcript_is_current", lambda *_a: True)
    monkeypatch.setattr(ijob, "already_indexed", lambda *_a: True)
    req = SimpleNamespace(
        force=False,
        model_size="large-v3",
        language="en",
        diarize=True,
        show_name="S",
        index_model_keys=["bge-m3"],
        index_chunkings=["semantic"],
    )
    args = ("/s/ep.mp3", "ep", None, req, lambda: False, lambda *_a: None, 0, 0.0)
    assert batch._batch_transcribe(*args) is False
    assert batch._batch_index(*args) is False


@pytest.mark.parametrize("model, reruns", [("qwen3:4b", False), ("other:1b", True)])
def test_batch_correct_skips_a_transcript_language_match(
    tmp_path, monkeypatch, model, reruns
):
    """Correct's provenance records the transcript-derived source language, so
    the already-done check has to compare against that, not the request value.
    Otherwise every episode whose transcribe language differs from the LLM
    source-language setting is corrected again on every batch run.

    The LLM resolver is stubbed: against a live Ollama with the model not
    pulled, resolution would fail first and the test would pass without ever
    reaching the check. The twin (another model) proves the check is what decides."""
    import podcodex.core.correct as core_correct
    from podcodex.api.routes.batch import BatchRequest, _batch_llm_step
    from podcodex.core._utils import AudioPaths
    from tests.fixtures.llm import stub_llm_resolver

    stub_llm_resolver(monkeypatch)
    ran: list[int] = []

    def fake_correct(segments, **_k):
        ran.append(1)
        return segments

    monkeypatch.setattr(core_correct, "correct_segments", fake_correct)

    show = tmp_path / "show"
    (show / "ep").mkdir(parents=True)
    audio = show / "ep.mp3"
    audio.touch()
    base = show / "ep" / "ep"

    save_version(
        base,
        "transcript",
        [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "bonjour"}],
        {"step": "transcript", "type": "raw", "params": {"language": "fr"}},
    )
    save_version(
        base,
        "corrected",
        [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "Bonjour"}],
        {
            "step": "corrected",
            "type": "raw",
            "model": "qwen3:4b",
            # What the save path writes: iso_to_language("fr").
            "params": {
                "llm_mode": "ollama",
                "llm_provider_profile": None,
                "source_lang": "French",
            },
        },
    )

    # The request still carries the default source language.
    req = BatchRequest(
        show_folder=str(show),
        audio_paths=[str(audio)],
        llm_mode="ollama",
        llm_model=model,
        source_lang="English",
    )
    p = AudioPaths.from_audio(str(audio))
    did_work = _batch_llm_step(
        str(audio),
        p,
        req,
        lambda: False,
        lambda *a, **k: None,
        0,
        0.0,
        step="correct",
    )
    assert did_work is reruns
    assert bool(ran) is reruns


def test_batch_cancel_stops_inside_the_episode(tmp_path, monkeypatch):
    """Cancel during a batch LLM step stops after the current LLM batch and
    saves nothing, instead of running the episode to the end."""
    import podcodex.core.correct as core_correct
    from podcodex.api.routes._helpers import TaskCancelled
    from podcodex.api.routes.batch import BatchRequest, _batch_llm_step
    from podcodex.core._utils import AudioPaths
    from tests.fixtures.llm import stub_llm_resolver
    from podcodex.core.versions import list_versions

    stub_llm_resolver(monkeypatch)
    ran: list[int] = []

    def fake_correct(segments, *, on_batch, **_k):
        for n in (1, 2, 3):
            ran.append(n)
            on_batch(n, 3)
        return segments

    monkeypatch.setattr(core_correct, "correct_segments", fake_correct)
    show = tmp_path / "show"
    (show / "ep").mkdir(parents=True)
    audio = show / "ep.mp3"
    audio.touch()
    base = show / "ep" / "ep"
    save_version(
        base,
        "transcript",
        [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hi"}],
        {"step": "transcript", "type": "raw", "params": {"language": "en"}},
    )
    req = BatchRequest(
        show_folder=str(show),
        audio_paths=[str(audio)],
        llm_mode="ollama",
        llm_model="m",
    )
    with pytest.raises(TaskCancelled):
        _batch_llm_step(
            str(audio),
            AudioPaths.from_audio(str(audio)),
            req,
            lambda: True,
            lambda *a, **k: None,
            0,
            0.0,
            step="correct",
        )
    assert ran == [1]
    assert list_versions(base, "corrected") == []
