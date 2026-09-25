"""The transcription step worker and the whisperx-facing functions.

Neither runs in CI for real (the pipeline extra is skipped), so both are
driven in-process: `run_for_batch` with the ML steps replaced by recorders
over real version rows, and `diarize_file` / `transcribe_file` with a fake
``whisperx`` module.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from podcodex.core import transcribe
from podcodex.core.transcribe_job import run_for_batch
from podcodex.core.versions import load_latest, save_version
from tests.fixtures.episodes import make_episode, prov


@pytest.fixture
def episode(tmp_path):
    return make_episode(tmp_path)


def _segments_version(base, model: str, text: str) -> str:
    return save_version(
        base,
        "segments",
        [{"start": 0.0, "end": 1.0, "text": text}],
        prov("segments", model, language="en"),
    )


@pytest.fixture
def recorder(monkeypatch):
    """Replace the four ML sub-steps with recorders."""
    calls: list[str] = []
    for name in ("transcribe_file", "diarize_file", "assign_speakers"):
        monkeypatch.setattr(transcribe, name, lambda *a, _n=name, **k: calls.append(_n))
    monkeypatch.setattr(
        transcribe,
        "export_transcript",
        lambda *a, **k: calls.append(f"export(diarized={k['diarized']})"),
    )
    return calls


def _run(audio, *, diarize=False, cancelled=lambda: False, force=False):
    return run_for_batch(
        progress_cb=lambda *_: None,
        cancelled=cancelled,
        audio_path=str(audio),
        stem="ep",
        show_name="show",
        model_size="turbo",
        language="en",
        batch_size=4,
        diarize=diarize,
        hf_token=None,
        num_speakers=None,
        clean=False,
        force=force,
    )


def test_a_matching_current_transcript_skips_everything(episode, recorder):
    base, audio = episode
    save_version(
        base,
        "transcript",
        [{"speaker": "A", "text": "x", "start": 0.0, "end": 1.0}],
        prov("transcript", "turbo", diarize=False, language="en"),
    )

    assert _run(audio) == {"did_work": False}
    assert recorder == []


def test_an_older_match_is_not_reused(episode, recorder):
    """turbo, then medium, then turbo again: the newest segments are
    medium's, and every later step loads the newest, so the matching turbo
    version from the first run must not count as done."""
    base, audio = episode
    _segments_version(base, "turbo", "from turbo")
    _segments_version(base, "medium", "from medium")

    _run(audio)

    assert recorder[0] == "transcribe_file"


def test_current_segments_are_reused_and_the_rest_runs(episode, recorder):
    base, audio = episode
    _segments_version(base, "turbo", "from turbo")

    assert _run(audio, diarize=True) == {"did_work": True}
    assert recorder == [
        "diarize_file",
        "assign_speakers",
        "export(diarized=False)",
        "export(diarized=True)",
    ]


def test_a_cancel_stops_before_the_next_sub_step(episode, recorder):
    _base, audio = episode
    flags = iter([False, True])

    result = _run(audio, diarize=True, cancelled=lambda: next(flags, True))

    assert result == {"did_work": True}
    assert recorder == ["transcribe_file"]


def test_segments_version_without_its_file_is_not_a_match(episode, recorder):
    from podcodex.core.versions import version_path

    base, audio = episode
    vid = _segments_version(base, "turbo", "from turbo")
    version_path(base, "segments", vid).unlink()

    _run(audio)

    assert recorder[0] == "transcribe_file"


# ── whisperx-facing functions, with a fake whisperx ──────────────────────


class _FakePipeline:
    frame: pd.DataFrame

    def __init__(self, **kwargs):
        _FakePipeline.kwargs = kwargs

    def __call__(self, audio, num_speakers=None):
        return _FakePipeline.frame


@pytest.fixture
def fake_whisperx(monkeypatch, tmp_path):
    monkeypatch.setenv("PODCODEX_CACHE_DIR", str(tmp_path / "models"))
    whisperx = types.ModuleType("whisperx")
    whisperx.load_audio = lambda path: [0.0] * 10
    diarize_mod = types.ModuleType("whisperx.diarize")
    diarize_mod.DiarizationPipeline = _FakePipeline
    whisperx.diarize = diarize_mod
    monkeypatch.setitem(sys.modules, "whisperx", whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", diarize_mod)
    monkeypatch.setattr(transcribe, "free_vram", lambda: None)
    monkeypatch.setattr("podcodex.core.device.resolve_device", lambda: ("cpu", "int8"))
    return whisperx


def test_a_missing_hf_token_fails_before_loading_anything(
    episode, fake_whisperx, monkeypatch
):
    _base, audio = episode
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _FakePipeline.kwargs = None

    with pytest.raises(ValueError, match="HF_TOKEN not found"):
        transcribe.diarize_file(audio)
    assert _FakePipeline.kwargs is None


class _Seg:
    def __init__(self, start, end):
        self.start, self.end = start, end


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(
            {
                "segment": [_Seg(0.0, 1.0), _Seg(1.0, 2.0)],
                "speaker": ["SPEAKER_00", "SPEAKER_01"],
            }
        ),
        pd.DataFrame(
            {
                "start": [0.0, 1.0],
                "end": [1.0, 2.0],
                "speaker": ["SPEAKER_00", "SPEAKER_01"],
            }
        ),
    ],
    ids=["segment-objects", "flat-columns"],
)
def test_both_pyannote_output_shapes_give_the_same_speakers(
    episode, fake_whisperx, frame
):
    """pyannote versions return either a `segment` column of objects or flat
    start/end columns; both must save the same diarization."""
    from podcodex.core.constants import DIARIZATION_MODEL

    base, audio = episode
    _FakePipeline.frame = frame

    out = transcribe.diarize_file(audio, hf_token="hf_x")

    assert out["speakers_found"] == ["SPEAKER_00", "SPEAKER_01"]
    assert load_latest(base, "diarization") == [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
    ]
    assert _FakePipeline.kwargs["model_name"] == DIARIZATION_MODEL


def test_a_token_without_the_hf_prefix_is_refused_up_front(episode, fake_whisperx):
    """pyannote drops such a token silently and the load then fails as a
    gated-repo error, pointing at the wrong fix."""
    _base, audio = episode
    with pytest.raises(ValueError, match="start with 'hf_'"):
        transcribe.diarize_file(audio, hf_token="not-a-token")


def test_a_gated_model_names_the_page_to_accept(episode, fake_whisperx, monkeypatch):
    from huggingface_hub.errors import GatedRepoError

    _base, audio = episode

    def gated(**_kw):
        raise GatedRepoError("403 gated")

    monkeypatch.setattr(sys.modules["whisperx.diarize"], "DiarizationPipeline", gated)
    with pytest.raises(
        RuntimeError, match="accept its conditions at https://huggingface.co/"
    ):
        transcribe.diarize_file(audio, hf_token="hf_x")
