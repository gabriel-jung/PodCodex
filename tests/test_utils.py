"""Tests for podcodex.core._utils — shared pipeline utilities.

Consolidates tests for functions previously duplicated between
test_correct.py and test_translate.py (call_and_parse,
batch_segments_by_duration, segments_to_text).
"""

import json

import pytest

from podcodex.rag.hit import SpeakerTurn
from podcodex.core._utils import (
    batch_segments_by_duration,
    call_and_parse,
    merge_display_turns,
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


def test_call_and_parse_count_mismatch_rejects_whole_batch():
    """LLM returning fewer items than the batch = index-drift risk; reject everything."""
    batch = make_segments("Premier", "Deuxieme")
    response = json.dumps([{"index": 0, "text": "First"}])
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "Premier"
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
        assert "all 2 segments" in messages[1]["content"]
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
    # Segments start every ~17 min → each falls in its own 15-min window.
    segments = [
        {"speaker": "A", "start": i * 1000, "end": i * 1000 + 300, "text": f"s{i}"}
        for i in range(3)
    ]
    batches = batch_segments_by_duration(segments, batch_minutes=15)
    assert len(batches) == 3


def test_batch_merges_tiny_overshoot_tail():
    # Segments run 5 s past the second 15-min cutoff. Expect 2 batches,
    # not 3 — the tiny tail is absorbed into the previous batch.
    segments = [
        {"speaker": "A", "start": float(s), "end": float(s) + 0.5, "text": f"s{s}"}
        for s in range(0, 1906, 100)
    ]
    batches = batch_segments_by_duration(segments, batch_minutes=15)
    assert len(batches) == 2
    assert sum(len(b) for b in batches) == len(segments)


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


def test_batch_count_produces_exact_count():
    # 100 segments spanning ~50 min. Request 4 batches → exactly 4, even
    # though batch_minutes is left at the default. No spurious extra batch.
    segments = [
        {
            "speaker": "A",
            "start": float(i * 30),
            "end": float(i * 30) + 1,
            "text": f"s{i}",
        }
        for i in range(100)
    ]
    batches = batch_segments_by_duration(segments, batch_count=4)
    assert len(batches) == 4
    assert sum(len(b) for b in batches) == 100


def test_batch_count_overrides_minutes():
    # batch_minutes alone would split these into 10 batches; batch_count wins.
    segments = [
        {
            "speaker": "A",
            "start": float(i * 600),
            "end": float(i * 600) + 1,
            "text": f"s{i}",
        }
        for i in range(10)
    ]
    batches = batch_segments_by_duration(segments, batch_minutes=15, batch_count=2)
    assert len(batches) == 2
    assert sum(len(b) for b in batches) == 10


def test_batch_count_one_is_single_batch():
    segments = make_segments("a", "b", "c")
    assert len(batch_segments_by_duration(segments, batch_count=1)) == 1


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


# ──────────────────────────────────────────────
# merge_display_turns
# ──────────────────────────────────────────────


def test_merge_display_turns_collapses_same_speaker_runs():
    """Consecutive turns from the same speaker merge into a single block
    with concatenated text and the last turn's ``end`` timestamp."""
    turns = [
        SpeakerTurn(speaker="Alice", text="Hello there.", start=0.0, end=1.0),
        SpeakerTurn(speaker="Alice", text="How are you?", start=1.0, end=2.5),
        SpeakerTurn(speaker="Bob", text="Good, you?", start=2.5, end=3.5),
        SpeakerTurn(speaker="Bob", text="And yourself?", start=3.5, end=4.0),
    ]
    merged = merge_display_turns(turns)
    assert len(merged) == 2
    assert merged[0]["speaker"] == "Alice"
    assert merged[0]["text"] == "Hello there. How are you?"
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 2.5
    assert merged[1]["speaker"] == "Bob"
    assert merged[1]["text"] == "Good, you? And yourself?"
    assert merged[1]["end"] == 4.0


def test_merge_display_turns_preserves_alternation():
    """Alternating speakers produce one block per turn (no collapsing)."""
    turns = [
        SpeakerTurn(speaker="Alice", text="A1", start=0.0, end=1.0),
        SpeakerTurn(speaker="Bob", text="B1", start=1.0, end=2.0),
        SpeakerTurn(speaker="Alice", text="A2", start=2.0, end=3.0),
    ]
    merged = merge_display_turns(turns)
    assert [(m["speaker"], m["text"]) for m in merged] == [
        ("Alice", "A1"),
        ("Bob", "B1"),
        ("Alice", "A2"),
    ]


def test_merge_display_turns_skips_empty_and_unknown_speaker():
    """Empty text is dropped; blank speaker labels default to ``Unknown``."""
    turns = [
        SpeakerTurn(speaker="Alice", text="   ", start=0.0, end=1.0),
        SpeakerTurn(speaker="", text="ghost line", start=1.0, end=2.0),
        SpeakerTurn(speaker="", text="another ghost", start=2.0, end=3.0),
        SpeakerTurn(speaker="Alice", text="real line", start=3.0, end=4.0),
    ]
    merged = merge_display_turns(turns)
    # Whitespace-only turn dropped; ghost turns collapse under "Unknown"; Alice
    # closes as a separate block.
    assert [m["speaker"] for m in merged] == ["Unknown", "Alice"]
    assert merged[0]["text"] == "ghost line another ghost"
    assert merged[1]["text"] == "real line"


# ── Timestamp helpers (format_hms / parse_time) ──────────────────────────

from podcodex.core._utils import format_hms, parse_time  # noqa: E402


def test_format_hms_under_hour():
    assert format_hms(0) == "0m00"
    assert format_hms(578) == "9m38"
    assert format_hms(3599) == "59m59"


def test_format_hms_hour_and_over():
    assert format_hms(3600) == "1h00m00"
    assert format_hms(4186) == "1h09m46"


def test_format_hms_floors_not_rounds():
    # Fractional seconds truncate down (canonical wiki floor form), never up:
    # a start timestamp must not point past the passage it marks.
    assert format_hms(1.5) == "0m01"
    assert format_hms(292.787) == "4m52"
    assert format_hms(292.999) == "4m52"


def test_parse_time_seconds_forms():
    assert parse_time(4186) == 4186.0
    assert parse_time(4186.0) == 4186.0
    assert parse_time("4186") == 4186.0


def test_parse_time_clock_forms_equivalent():
    assert parse_time("1h09m46") == 4186.0
    assert parse_time("69m46") == 4186.0
    assert parse_time("9m38") == 578.0


def test_parse_time_rejects_out_of_range_fields():
    with pytest.raises(ValueError):
        parse_time("1h09m60")
    with pytest.raises(ValueError):
        parse_time("1h60m00")


def test_merge_display_turns_absent_end_does_not_rewind_run():
    """A turn with no timing (end defaults to 0.0) must not drag the merged
    run's end backwards below its start."""
    turns = [
        SpeakerTurn(speaker="Alice", text="one", start=1.0, end=9.0),
        SpeakerTurn(speaker="Alice", text="two"),  # legacy turn, no timing
    ]
    merged = merge_display_turns(turns)
    assert len(merged) == 1
    assert merged[0]["end"] == 9.0


def test_placeholder_cannot_collide_with_a_real_name():
    """The current placeholder is not a plausible human name.

    Declaring it changes nothing, unlike the legacy value it replaced, which
    a documentary could legitimately call someone.
    """
    from podcodex.core._utils import (
        LEGACY_NARRATOR_SPEAKER,
        NARRATOR_SPEAKER,
        is_unattributed,
    )

    assert is_unattributed(NARRATOR_SPEAKER) is True
    assert is_unattributed(NARRATOR_SPEAKER, {NARRATOR_SPEAKER}) is True
    # Libraries written before the switch keep reading as unattributed.
    assert is_unattributed(LEGACY_NARRATOR_SPEAKER) is True
    assert is_unattributed("Alice") is False


def test_declared_speaker_outranks_the_placeholder():
    """A show that declares a speaker called "Narrator" means it.

    The name doubles as the no-diarization placeholder, so without this a
    documentary's narrator would vanish from its own roster and airtime.
    """
    from podcodex.core._utils import is_unattributed, speaker_airtime

    segs = [{"speaker": "Narrator", "start": 0.0, "end": 10.0, "text": "x"}]
    assert is_unattributed("Narrator") is True
    assert is_unattributed("Narrator", {"Narrator"}) is False
    # The empty label is never a name, declared or not.
    assert is_unattributed("", {"Narrator"}) is True
    assert speaker_airtime(segs) == {}
    assert list(speaker_airtime(segs, {"Narrator"})) == ["Narrator"]


def test_exports_keep_a_declared_narrator_named():
    """A show's own narrator must not be the only unnamed line in an export."""
    from podcodex.core._utils import segments_to_srt

    segs = [
        {"speaker": "Alice", "start": 0.0, "end": 1.0, "text": "bonjour"},
        {"speaker": "Narrator", "start": 1.0, "end": 2.0, "text": "plus tard"},
    ]
    # Undeclared, it is the no-diarization placeholder and names nobody.
    assert "Narrator:" not in segments_to_srt(segs)
    assert "Alice: bonjour" in segments_to_srt(segs)
    # Declared, it is a name like any other.
    assert "Narrator: plus tard" in segments_to_srt(segs, declared={"Narrator"})
