"""Tests for podcodex.core.transcribe — pure functions only (no GPU, no models)."""

from pathlib import Path

import pytest

from podcodex.core._utils import merge_consecutive_segments
from podcodex.core.transcribe import (
    load_transcript,
    load_transcript_full,
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

_SEGMENTS = [
    {"speaker": "Alice", "start": 0.0, "end": 5.0, "text": "Hello world"},
    {"speaker": "Bob", "start": 5.0, "end": 10.0, "text": "Hi there how are you"},
]

_NEW_FORMAT = {
    "meta": {
        "show": "My Show",
        "episode": "Episode 1",
        "speakers": ["Alice", "Bob"],
        "duration": 10.0,
        "word_count": 7,
    },
    "segments": _SEGMENTS,
}


def _save_parquet_version(
    base: Path, step: str, records: list[dict], **extra_params
) -> None:
    """Save a list of dicts as a versioned parquet file (test helper)."""
    from podcodex.core.versions import save_version

    provenance = {"step": step, "type": "raw", "params": extra_params}
    save_version(base, step, records, provenance)


# ──────────────────────────────────────────────
# merge_consecutive_segments
# ──────────────────────────────────────────────


def test_simplify_empty():
    assert merge_consecutive_segments([]) == []


def test_simplify_single_segment():
    seg = [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"}]
    result = merge_consecutive_segments(seg)
    assert result == [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"}]


def test_simplify_merges_consecutive_same_speaker():
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"},
        {"speaker": "Alice", "start": 2.0, "end": 4.0, "text": "world"},
        {"speaker": "Bob", "start": 4.0, "end": 6.0, "text": "Hi"},
    ]
    result = merge_consecutive_segments(segments)
    assert len(result) == 2
    assert result[0]["speaker"] == "Alice"
    assert result[0]["text"] == "Hello world"
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 4.0
    assert result[1]["speaker"] == "Bob"


def test_simplify_does_not_merge_alternating_speakers():
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"},
        {"speaker": "Bob", "start": 2.0, "end": 4.0, "text": "Hi"},
        {"speaker": "Alice", "start": 4.0, "end": 6.0, "text": "How are you"},
    ]
    result = merge_consecutive_segments(segments)
    assert len(result) == 3


def test_simplify_falls_back_to_unknown_speaker():
    segments = [{"start": 0.0, "end": 2.0, "text": "No speaker key"}]
    result = merge_consecutive_segments(segments)
    assert result[0]["speaker"] == "UNKNOWN"


def test_simplify_strips_whitespace_from_text():
    segments = [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "  Hello  "}]
    result = merge_consecutive_segments(segments)
    assert result[0]["text"] == "Hello"


# ──────────────────────────────────────────────
# load_transcript / load_transcript_full
# ──────────────────────────────────────────────


def _save_transcript_version(
    base: Path, segments: list[dict], meta: dict | None = None
) -> None:
    from podcodex.core.versions import save_version

    provenance = {"step": "transcript", "type": "raw", "params": {"meta": meta or {}}}
    save_version(base, "transcript", segments, provenance)


def test_load_transcript_returns_segments(tmp_path):
    audio = tmp_path / "ep.mp3"
    audio.touch()
    base = tmp_path / "ep" / "ep"
    base.parent.mkdir()
    _save_transcript_version(base, _SEGMENTS)

    result = load_transcript(audio)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["speaker"] == "Alice"


def test_load_transcript_full_includes_meta(tmp_path):
    audio = tmp_path / "ep.mp3"
    audio.touch()
    base = tmp_path / "ep" / "ep"
    base.parent.mkdir()
    _save_transcript_version(base, _SEGMENTS, meta=_NEW_FORMAT["meta"])

    result = load_transcript_full(audio)

    assert "meta" in result
    assert "segments" in result
    assert result["meta"]["show"] == "My Show"
    assert result["meta"]["episode"] == "Episode 1"
    assert result["meta"]["speakers"] == ["Alice", "Bob"]
    assert result["meta"]["duration"] == 10.0
    assert result["meta"]["word_count"] == 7
    assert len(result["segments"]) == 2


def test_load_transcript_full_empty_when_missing(tmp_path):
    audio = tmp_path / "ep.mp3"
    audio.touch()
    (tmp_path / "ep").mkdir()
    result = load_transcript_full(audio)
    assert result == {"meta": {}, "segments": []}


# ──────────────────────────────────────────────
# export_transcript meta fields
# ──────────────────────────────────────────────


def test_export_transcript_meta(tmp_path):
    """export_transcript writes correct meta derived from segments."""
    from podcodex.core.transcribe import export_transcript, save_speaker_map

    audio = tmp_path / "ep.mp3"

    # Write the minimal files export_transcript() depends on
    diarized = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello world"},
        {
            "start": 5.5,
            "end": 12.0,
            "speaker": "SPEAKER_01",
            "text": "Hi there how are you",
        },
    ]
    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    base = ep_dir / "ep"
    _save_parquet_version(base, "diarized_segments", diarized)
    save_speaker_map(audio, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})

    segments = export_transcript(audio, show="My Show", episode="Episode 1")

    # Return value is still a plain list
    assert isinstance(segments, list)
    assert len(segments) == 2

    # File is in new format
    full = load_transcript_full(audio)
    assert full["meta"]["show"] == "My Show"
    assert full["meta"]["diarized"] is True
    assert full["meta"]["episode"] == "Episode 1"
    assert full["meta"]["speakers"] == ["Alice", "Bob"]
    assert full["meta"]["duration"] == pytest.approx(12.0, abs=0.01)
    assert full["meta"]["word_count"] == 7


# ──────────────────────────────────────────────
# Skip diarization mode
# ──────────────────────────────────────────────


def test_export_transcript_no_diarization(tmp_path):
    """export_transcript(diarized=False) reads raw segments, assigns Narrator."""
    from podcodex.core._utils import NARRATOR_SPEAKER
    from podcodex.core.transcribe import export_transcript

    audio = tmp_path / "ep.mp3"
    raw_segs = [
        {"start": 0.0, "end": 5.0, "text": "Hello world"},
        {"start": 5.5, "end": 12.0, "text": "Hi there"},
    ]
    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    base = ep_dir / "ep"
    _save_parquet_version(base, "segments", raw_segs, language="en", duration=12.0)

    segments = export_transcript(audio, diarized=False, show="S", episode="E")

    # All segments should have Narrator as speaker
    speakers = {s["speaker"] for s in segments if s["speaker"] != "[BREAK]"}
    assert speakers == {NARRATOR_SPEAKER}

    # Version saved to DB
    from podcodex.core.versions import has_version

    assert has_version(ep_dir / "ep", "transcript")

    # Meta includes diarized=False
    full = load_transcript_full(audio)
    assert full["meta"]["diarized"] is False
    assert full["meta"]["show"] == "S"


# ──────────────────────────────────────────────
# clean_transcript integration with export
# ──────────────────────────────────────────────


@pytest.mark.parametrize("clean, speakers", [(True, ["Alice"]), (False, None)])
def test_export_transcript_clean_drops_flagged_segments(tmp_path, clean, speakers):
    """clean=True removes unknown speakers and low-density segments; clean=False
    keeps every segment."""
    from podcodex.core.transcribe import export_transcript, save_speaker_map

    audio = tmp_path / "ep.mp3"
    diarized = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "SPEAKER_00",
            "text": "Hello world this is a good segment",
        },
        # Low density: 2 chars over 6.5s, under the threshold of 2 chars/s.
        {"start": 5.5, "end": 12.0, "speaker": "SPEAKER_01", "text": "Hi"},
        {
            "start": 13.0,
            "end": 18.0,
            "speaker": "UNKNOWN",
            "text": "Unknown speaker segment here",
        },
    ]
    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    _save_parquet_version(ep_dir / "ep", "diarized_segments", diarized)
    save_speaker_map(audio, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})

    segments = export_transcript(audio, clean=clean)

    if speakers is None:
        assert len(segments) == 3
    else:
        assert [s["speaker"] for s in segments] == speakers


def test_export_transcript_clean_nodiarize_only_density(tmp_path):
    """clean=True + diarized=False only filters by density, not speaker.

    With diarize=False all segments share the Narrator speaker and get merged,
    so we need a >10s gap to prevent merging and test density filtering.
    """
    from podcodex.core.transcribe import export_transcript

    audio = tmp_path / "ep.mp3"
    raw_segs = [
        {"start": 0.0, "end": 5.0, "text": "Hello world this is a good segment"},
        # >10s gap prevents merge; low density (2 chars / 5s = 0.4 < 2)
        {"start": 50.0, "end": 55.0, "text": "Hi"},
    ]
    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    base = ep_dir / "ep"
    _save_parquet_version(base, "segments", raw_segs, language="en", duration=55.0)

    segments = export_transcript(audio, diarized=False, clean=True)
    # Low-density segment removed; good segment + BREAK remain
    texts = [s["text"] for s in segments if s["speaker"] != "[BREAK]"]
    assert texts == ["Hello world this is a good segment"]


def test_cached_whisper_and_diarization_models_load_from_their_folders(
    tmp_path, monkeypatch
):
    """Given a name, faster-whisper and pyannote ask the Hub before using the
    cache: offline, pyannote spends about two minutes on retries."""
    import sys
    import types

    import numpy as np

    import podcodex.core.cache as cache
    from podcodex.core import transcribe

    class Loaded(Exception):
        pass

    def record(source, *_a, **kw):
        raise Loaded(kw.get("model_name", source), kw.get("language"))

    whisperx = types.ModuleType("whisperx")
    whisperx.load_model = record
    diarize = types.ModuleType("whisperx.diarize")
    diarize.DiarizationPipeline = lambda **kw: record(None, **kw)
    monkeypatch.setitem(sys.modules, "whisperx", whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", diarize)
    monkeypatch.setattr(transcribe, "_decoded_audio", lambda _p: np.zeros(16000))
    monkeypatch.setattr(cache, "cached_snapshot", lambda repo, **_kw: f"/snap/{repo}")
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"")

    with pytest.raises(Loaded) as whisper:
        transcribe.transcribe_file(audio, model_size="base.en", language="", force=True)
    # The folder path no longer ends in ".en", so English is pinned explicitly.
    assert whisper.value.args == ("/snap/Systran/faster-whisper-base.en", "en")

    with pytest.raises(Exception) as pyannote:
        transcribe.diarize_file(audio, hf_token="hf_x", force=True)
    assert "/snap/pyannote/speaker-diarization-community-1" in repr(
        pyannote.value.__cause__ or pyannote.value
    )


@pytest.mark.parametrize(
    "language, expected",
    [("ja", "/snap/jonatasgrosman/wav2vec2-large-xlsr-53-japanese"), ("fr", None)],
)
def test_a_hub_alignment_model_loads_from_its_folder(
    tmp_path, monkeypatch, language, expected
):
    """Languages torchaudio doesn't cover align with a Hub model; torchaudio
    ones keep whisperx's own default."""
    import sys
    import types

    import numpy as np

    import podcodex.core.cache as cache
    from podcodex.core import transcribe

    class Loaded(Exception):
        pass

    model = types.SimpleNamespace(
        transcribe=lambda *_a, **_kw: {"segments": [], "language": language}
    )
    whisperx = types.ModuleType("whisperx")
    whisperx.load_model = lambda *_a, **_kw: model

    def load_align_model(**kw):
        raise Loaded(kw.get("model_name"))

    whisperx.load_align_model = load_align_model
    alignment = types.ModuleType("whisperx.alignment")
    alignment.DEFAULT_ALIGN_MODELS_TORCH = {"fr": "VOXPOPULI_ASR_BASE_10K_FR"}
    alignment.DEFAULT_ALIGN_MODELS_HF = {
        "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
    }
    monkeypatch.setitem(sys.modules, "whisperx", whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.alignment", alignment)
    monkeypatch.setattr(transcribe, "_decoded_audio", lambda _p: np.zeros(16000))
    monkeypatch.setattr(cache, "cached_snapshot", lambda repo, **_kw: f"/snap/{repo}")
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"")

    with pytest.raises(Loaded) as align:
        transcribe.transcribe_file(audio, language=language, force=True)
    assert align.value.args == (expected,)
