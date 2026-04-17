"""Tests for podcodex.core._utils — shared pipeline utilities.

Consolidates tests for functions previously duplicated between
test_correct.py and test_translate.py (call_and_parse,
batch_segments_by_duration, segments_to_text).
"""

import json


from podcodex.core._utils import (
    batch_segments_by_duration,
    call_and_parse,
    segments_to_srt,
    segments_to_text,
    segments_to_vtt,
)


def make_call_fn(response: str):
    return lambda messages: response


def make_segments(*texts, speaker: str = "Alice", seg_duration: float = 10.0):
    return [
        {
            "speaker": speaker,
            "start": i * seg_duration,
            "end": (i + 1) * seg_duration,
            "text": t,
        }
        for i, t in enumerate(texts)
    ]


# ──────────────────────────────────────────────
# call_and_parse
# ──────────────────────────────────────────────


def test_call_and_parse_happy_path():
    batch = make_segments("Bonjour", "Au revoir")
    response = json.dumps(
        [{"index": 0, "text": "Hello"}, {"index": 1, "text": "Goodbye"}]
    )
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "Hello"
    assert result[1]["text"] == "Goodbye"


def test_call_and_parse_bad_json_keeps_original():
    batch = make_segments("Bonjour le monde")
    result = call_and_parse(batch, "sys", make_call_fn("not json at all"))
    assert len(result) == 1
    assert result[0]["text"] == "Bonjour le monde"


def test_call_and_parse_missing_index_keeps_original():
    batch = make_segments("Premier", "Deuxieme")
    response = json.dumps([{"index": 0, "text": "First"}])
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "First"
    assert result[1]["text"] == "Deuxieme"


def test_call_and_parse_strips_think_tags():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Hello"}])
    response = f"<think>some reasoning</think>\n{inner}"
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "Hello"


def test_call_and_parse_strips_markdown_fences():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Hello"}])
    response = f"```json\n{inner}\n```"
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "Hello"


def test_call_and_parse_truncation_guard_keeps_original():
    """Corrections shorter than min_length_ratio × original keep the original."""
    batch = make_segments("This is a very long sentence that should not be truncated")
    response = json.dumps([{"index": 0, "text": "Short"}])
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0.7)
    assert (
        result[0]["text"] == "This is a very long sentence that should not be truncated"
    )


def test_call_and_parse_truncation_guard_disabled():
    """min_length_ratio=0 disables the guard."""
    batch = make_segments("This is a very long sentence that should not be truncated")
    response = json.dumps([{"index": 0, "text": "Short"}])
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "Short"


def test_call_and_parse_skips_break_segments():
    """[BREAK] segments pass through without being sent to the LLM."""
    batch = [
        {"speaker": "Alice", "start": 0.0, "end": 10.0, "text": "Bonjour"},
        {"speaker": "[BREAK]", "start": 10.0, "end": 15.0, "text": ""},
        {"speaker": "Alice", "start": 15.0, "end": 25.0, "text": "On continue"},
    ]
    response = json.dumps(
        [
            {"index": 0, "text": "Hello"},
            {"index": 1, "text": "Continuing"},
        ]
    )
    calls = []

    def tracking(messages):
        calls.append(messages)
        # [BREAK] must not appear in the user message
        assert "[BREAK]" not in messages[1]["content"]
        assert "2 numbered segments" in messages[1]["content"]
        return response

    result = call_and_parse(batch, "sys", tracking, min_length_ratio=0)
    assert len(result) == 3
    assert result[0]["text"] == "Hello"
    assert result[1]["speaker"] == "[BREAK]"
    assert result[1]["text"] == ""
    assert result[2]["text"] == "Continuing"
    assert len(calls) == 1


def test_call_and_parse_all_breaks_no_llm_call():
    batch = [
        {"speaker": "[BREAK]", "start": 0.0, "end": 5.0, "text": ""},
        {"speaker": "[BREAK]", "start": 10.0, "end": 15.0, "text": ""},
    ]

    def should_not_be_called(messages):
        raise AssertionError("LLM should not be called for all-break batches")

    result = call_and_parse(batch, "sys", should_not_be_called)
    assert len(result) == 2
    assert all(seg["speaker"] == "[BREAK]" for seg in result)


# ──────────────────────────────────────────────
# batch_segments_by_duration
# ──────────────────────────────────────────────


def test_batch_single_batch_when_under_limit():
    segments = make_segments("Short")
    batches = batch_segments_by_duration(segments, batch_minutes=15)
    assert len(batches) == 1


def test_batch_splits_by_duration():
    # Each segment is 10 minutes → 15-minute limit fits one per batch.
    segments = [
        {"speaker": "A", "start": i * 600, "end": (i + 1) * 600, "text": f"s{i}"}
        for i in range(3)
    ]
    batches = batch_segments_by_duration(segments, batch_minutes=15)
    assert len(batches) == 3


def test_batch_groups_short_segments():
    # 4 × 5-minute segments → 15-minute limit fits 3 per batch, leaving 1.
    segments = [
        {"speaker": "A", "start": i * 300, "end": (i + 1) * 300, "text": f"s{i}"}
        for i in range(4)
    ]
    batches = batch_segments_by_duration(segments, batch_minutes=15)
    assert len(batches) == 2
    assert sum(len(b) for b in batches) == 4


def test_batch_empty():
    assert batch_segments_by_duration([], batch_minutes=15) == []


# ──────────────────────────────────────────────
# segments_to_text / _srt / _vtt formatters
# ──────────────────────────────────────────────


def test_segments_to_text_empty():
    assert segments_to_text([]) == ""


def test_segments_to_text_contains_speaker_text_and_timestamps():
    segments = [{"speaker": "Alice", "start": 1.0, "end": 3.5, "text": "Hello"}]
    out = segments_to_text(segments)
    assert "Alice" in out
    assert "Hello" in out
    assert "1.000s" in out
    assert "3.500s" in out


def test_segments_to_text_preserves_order():
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"},
        {"speaker": "Bob", "start": 2.0, "end": 4.0, "text": "Hi"},
    ]
    out = segments_to_text(segments)
    assert out.index("Alice") < out.index("Bob")


def test_segments_to_srt_format():
    segments = [{"speaker": "Alice", "start": 0.0, "end": 1.5, "text": "Hi"}]
    out = segments_to_srt(segments)
    assert "1" in out  # cue index
    assert "00:00:00,000 --> 00:00:01,500" in out
    assert "Alice: Hi" in out


def test_segments_to_vtt_format():
    segments = [{"speaker": "Alice", "start": 0.0, "end": 1.0, "text": "Hi"}]
    out = segments_to_vtt(segments)
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.000" in out
    assert "<v Alice>Hi" in out
