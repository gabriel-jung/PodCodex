"""Tests for podcodex.core.transcribe — pure functions only (no GPU, no models)."""

from podcodex.core.transcribe import simplify_transcript, transcript_to_text


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
