"""Tests for podcodex.core.transcribe — pure functions only (no GPU, no models)."""

import json
import tempfile
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


def _write_transcript(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


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
    result = merge_consecutive_segments(segments)
    assert result[0]["speaker"] == "Alice"


def test_simplify_strips_whitespace_from_text():
    segments = [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "  Hello  "}]
    result = merge_consecutive_segments(segments)
    assert result[0]["text"] == "Hello"


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
    assert full["meta"]["episode"] == "Episode 1"
    assert full["meta"]["speakers"] == ["Alice", "Bob"]
    assert full["meta"]["duration"] == pytest.approx(12.0, abs=0.01)
    assert full["meta"]["word_count"] == 7


def test_export_transcript_defaults_empty_show_episode(tmp_path):
    from podcodex.core.transcribe import export_transcript, save_speaker_map

    audio = tmp_path / "ep.mp3"
    diarized = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "Hello"}]
    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    base = ep_dir / "ep"
    _save_parquet_version(base, "diarized_segments", diarized)
    save_speaker_map(audio, {"SPEAKER_00": "Alice"})

    export_transcript(audio)

    full = load_transcript_full(audio)
    assert full["meta"]["show"] == ""
    assert full["meta"]["episode"] == ""


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


def test_export_transcript_diarized_has_diarized_true(tmp_path):
    """export_transcript(diarized=True) includes diarized=True in meta."""
    from podcodex.core.transcribe import export_transcript, save_speaker_map

    audio = tmp_path / "ep.mp3"
    diarized = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello"},
    ]
    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    base = ep_dir / "ep"
    _save_parquet_version(base, "diarized_segments", diarized)
    save_speaker_map(audio, {"SPEAKER_00": "Alice"})

    export_transcript(audio)
    full = load_transcript_full(audio)
    assert full["meta"]["diarized"] is True


def test_is_segment_flagged_undiarized():
    """When diarized=False, speaker-based checks are skipped."""
    from podcodex.core.transcribe import is_segment_flagged

    # UNKNOWN speaker — flagged in diarized mode, not in nodiar
    seg = {"speaker": "UNKNOWN", "start": 0, "end": 5, "text": "Hello world test"}
    assert is_segment_flagged(seg, diarized=True) is True
    assert is_segment_flagged(seg, diarized=False) is False

    # Low density — flagged in both modes
    low_density = {"speaker": "Narrator", "start": 0, "end": 10, "text": "Hi"}
    assert is_segment_flagged(low_density, diarized=True) is True
    assert is_segment_flagged(low_density, diarized=False) is True

    # Normal segment — not flagged in either mode
    normal = {
        "speaker": "Alice",
        "start": 0,
        "end": 5,
        "text": "Hello world how are you doing today",
    }
    assert is_segment_flagged(normal, diarized=True) is False
    assert is_segment_flagged(normal, diarized=False) is False


# ──────────────────────────────────────────────
# clean_transcript integration with export
# ──────────────────────────────────────────────


def test_export_transcript_clean_removes_flagged(tmp_path):
    """export_transcript(clean=True) removes unknown speakers and low-density segments."""
    from podcodex.core.transcribe import export_transcript, save_speaker_map

    audio = tmp_path / "ep.mp3"
    diarized = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "SPEAKER_00",
            "text": "Hello world this is a good segment",
        },
        {
            "start": 5.5,
            "end": 12.0,
            "speaker": "SPEAKER_01",
            "text": "Hi",
        },  # low density (3 chars / 6.5s < 2)
        {
            "start": 13.0,
            "end": 18.0,
            "speaker": "UNKNOWN",
            "text": "Unknown speaker segment here",
        },
    ]
    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    base = ep_dir / "ep"
    _save_parquet_version(base, "diarized_segments", diarized)
    save_speaker_map(audio, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})

    segments = export_transcript(audio, clean=True)
    # Only Alice's good segment should survive
    assert len(segments) == 1
    assert segments[0]["speaker"] == "Alice"


def test_export_transcript_no_clean_preserves_all(tmp_path):
    """export_transcript(clean=False) preserves all segments including flagged ones."""
    from podcodex.core.transcribe import export_transcript, save_speaker_map

    audio = tmp_path / "ep.mp3"
    diarized = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "SPEAKER_00",
            "text": "Hello world this is a good segment",
        },
        {
            "start": 5.5,
            "end": 12.0,
            "speaker": "SPEAKER_01",
            "text": "Hi",
        },  # low density
        {
            "start": 13.0,
            "end": 18.0,
            "speaker": "UNKNOWN",
            "text": "Unknown speaker segment here",
        },
    ]
    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    base = ep_dir / "ep"
    _save_parquet_version(base, "diarized_segments", diarized)
    save_speaker_map(audio, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})

    segments = export_transcript(audio, clean=False)
    assert len(segments) == 3


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


def test_audio_paths_naming():
    """AudioPaths produces consistent file names for all pipeline steps."""
    from podcodex.core._utils import AudioPaths

    p = AudioPaths(audio_path=Path("/tmp/ep.mp3"), base=Path("/tmp/ep/ep"))
    assert p.transcript_raw.name == "ep.transcript.raw.json"
    assert p.transcript.name == "ep.transcript.json"
    assert p.speaker_map.name == "ep.speaker_map.json"
