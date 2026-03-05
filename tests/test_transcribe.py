"""Tests for podcodex.core.transcribe — pure functions only (no GPU, no models)."""

import json
import tempfile
from pathlib import Path

import pytest

from podcodex.core.transcribe import (
    load_transcript,
    load_transcript_full,
    simplify_transcript,
    transcript_to_text,
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


def _write_transcript(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ──────────────────────────────────────────────
# simplify_transcript
# ──────────────────────────────────────────────


def test_simplify_empty():
    assert simplify_transcript([]) == []


def test_simplify_single_segment():
    seg = [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"}]
    result = simplify_transcript(seg)
    assert result == [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"}]


def test_simplify_merges_consecutive_same_speaker():
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"},
        {"speaker": "Alice", "start": 2.0, "end": 4.0, "text": "world"},
        {"speaker": "Bob", "start": 4.0, "end": 6.0, "text": "Hi"},
    ]
    result = simplify_transcript(segments)
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
    result = simplify_transcript(segments)
    assert len(result) == 3


def test_simplify_falls_back_to_unknown_speaker():
    segments = [{"start": 0.0, "end": 2.0, "text": "No speaker key"}]
    result = simplify_transcript(segments)
    assert result[0]["speaker"] == "UNKNOWN"


def test_simplify_prefers_speaker_name_over_id():
    segments = [
        {
            "speaker": "SPEAKER_00",
            "speaker_name": "Alice",
            "start": 0.0,
            "end": 2.0,
            "text": "Hi",
        }
    ]
    result = simplify_transcript(segments)
    assert result[0]["speaker"] == "Alice"


def test_simplify_strips_whitespace_from_text():
    segments = [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "  Hello  "}]
    result = simplify_transcript(segments)
    assert result[0]["text"] == "Hello"


# ──────────────────────────────────────────────
# transcript_to_text
# ──────────────────────────────────────────────


def test_transcript_to_text_contains_speaker_and_text():
    segments = [{"speaker": "Alice", "start": 1.0, "end": 3.0, "text": "Hello"}]
    out = transcript_to_text(segments)
    assert "Alice" in out
    assert "Hello" in out


def test_transcript_to_text_contains_timestamps():
    segments = [{"speaker": "Alice", "start": 1.0, "end": 3.5, "text": "Hi"}]
    out = transcript_to_text(segments)
    assert "1.000s" in out
    assert "3.500s" in out


def test_transcript_to_text_multiple_segments_separated():
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"},
        {"speaker": "Bob", "start": 2.0, "end": 4.0, "text": "Hi"},
    ]
    out = transcript_to_text(segments)
    assert out.index("Alice") < out.index("Bob")


def test_transcript_to_text_empty():
    assert transcript_to_text([]) == ""


# ──────────────────────────────────────────────
# load_transcript — backward compat
# ──────────────────────────────────────────────


def test_load_transcript_new_format_returns_segments():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "ep.transcript.json"
        _write_transcript(f, _NEW_FORMAT)
        # We need a fake audio path whose stem matches the transcript stem
        audio = Path(tmp) / "ep.mp3"
        result = load_transcript(audio, output_dir=tmp)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["speaker"] == "Alice"


def test_load_transcript_old_format_returns_list():
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "ep.mp3"
        f = Path(tmp) / "ep.transcript.json"
        _write_transcript(f, _SEGMENTS)
        result = load_transcript(audio, output_dir=tmp)
    assert isinstance(result, list)
    assert result[0]["speaker"] == "Alice"


# ──────────────────────────────────────────────
# load_transcript_full
# ──────────────────────────────────────────────


def test_load_transcript_full_new_format():
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "ep.mp3"
        f = Path(tmp) / "ep.transcript.json"
        _write_transcript(f, _NEW_FORMAT)
        result = load_transcript_full(audio, output_dir=tmp)
    assert "meta" in result
    assert "segments" in result
    assert result["meta"]["show"] == "My Show"
    assert result["meta"]["episode"] == "Episode 1"
    assert result["meta"]["speakers"] == ["Alice", "Bob"]
    assert result["meta"]["duration"] == 10.0
    assert result["meta"]["word_count"] == 7
    assert len(result["segments"]) == 2


def test_load_transcript_full_old_format_wraps_with_empty_meta():
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "ep.mp3"
        f = Path(tmp) / "ep.transcript.json"
        _write_transcript(f, _SEGMENTS)
        result = load_transcript_full(audio, output_dir=tmp)
    assert result["meta"] == {}
    assert len(result["segments"]) == 2


# ──────────────────────────────────────────────
# export_transcript meta fields
# ──────────────────────────────────────────────


def test_export_transcript_meta(tmp_path):
    """export_transcript writes correct meta derived from segments."""
    import pandas as pd
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
    pd.DataFrame(diarized).to_parquet(
        tmp_path / "ep.diarized_segments.parquet", index=False
    )
    save_speaker_map(audio, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})

    segments = export_transcript(audio, show="My Show", episode="Episode 1")

    # Return value is still a plain list
    assert isinstance(segments, list)
    assert len(segments) == 2

    # File is in new format
    full = load_transcript_full(audio)
    assert full["meta"]["show"] == "My Show"
    assert full["meta"]["episode"] == "Episode 1"
    assert full["meta"]["speakers"] == ["Alice", "Bob"]
    assert full["meta"]["duration"] == pytest.approx(12.0, abs=0.01)
    assert full["meta"]["word_count"] == 7


def test_export_transcript_defaults_empty_show_episode(tmp_path):
    import pandas as pd
    from podcodex.core.transcribe import export_transcript, save_speaker_map

    audio = tmp_path / "ep.mp3"
    diarized = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "Hello"}]
    pd.DataFrame(diarized).to_parquet(
        tmp_path / "ep.diarized_segments.parquet", index=False
    )
    save_speaker_map(audio, {"SPEAKER_00": "Alice"})

    export_transcript(audio)

    full = load_transcript_full(audio)
    assert full["meta"]["show"] == ""
    assert full["meta"]["episode"] == ""
