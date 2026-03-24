"""Tests for podcodex.core.polish — pure functions only (no API calls, no Ollama)."""

import json
from podcodex.core._utils import call_and_parse
from podcodex.core.polish import (
    build_manual_prompt,
    build_manual_prompts_batched,
)


def make_call_fn(response: str):
    return lambda messages: response


def make_segments(*texts):
    return [
        {"speaker": "Alice", "start": i * 10.0, "end": (i + 1) * 10.0, "text": t}
        for i, t in enumerate(texts)
    ]


# ──────────────────────────────────────────────
# call_and_parse
# ──────────────────────────────────────────────


def test_call_and_parse_happy_path():
    batch = make_segments("Bonjour", "Au revoir")
    response = json.dumps(
        [
            {"index": 0, "text": "Bonjour corrige"},
            {"index": 1, "text": "Au revoir corrige"},
        ]
    )
    result = call_and_parse(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Bonjour corrige"
    assert result[1]["text"] == "Au revoir corrige"


def test_call_and_parse_truncation_falls_back_to_original():
    """If LLM returns a suspiciously short correction, keep the original text."""
    batch = make_segments(
        "Une longue phrase originale avec beaucoup de mots importants"
    )
    response = json.dumps([{"index": 0, "text": "Hi"}])
    result = call_and_parse(batch, "sys", make_call_fn(response))
    assert (
        result[0]["text"]
        == "Une longue phrase originale avec beaucoup de mots importants"
    )


def test_call_and_parse_bad_json_does_not_crash():
    batch = make_segments("Bonjour le monde")
    result = call_and_parse(batch, "sys", make_call_fn("not json at all"))
    assert len(result) == 1
    assert result[0]["text"] == "Bonjour le monde"


def test_call_and_parse_missing_index_falls_back():
    """If LLM omits a segment, that segment keeps its original text."""
    batch = make_segments("Premier", "Deuxieme")
    response = json.dumps([{"index": 0, "text": "Premier corrige"}])
    result = call_and_parse(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Premier corrige"
    assert result[1]["text"] == "Deuxieme"


def test_call_and_parse_strips_think_tags():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Bonjour corrige"}])
    response = f"<think>some reasoning</think>\n{inner}"
    result = call_and_parse(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Bonjour corrige"


def test_call_and_parse_strips_markdown_fences():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Bonjour corrige"}])
    response = f"```json\n{inner}\n```"
    result = call_and_parse(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Bonjour corrige"


def test_call_and_parse_skips_break_segments():
    """[BREAK] segments should pass through without being sent to the LLM."""
    batch = [
        {"speaker": "Alice", "start": 0.0, "end": 10.0, "text": "Bonjour"},
        {"speaker": "[BREAK]", "start": 10.0, "end": 15.0, "text": ""},
        {"speaker": "Alice", "start": 15.0, "end": 25.0, "text": "On continue"},
    ]
    # LLM only sees 2 segments (indices 0 and 1), not the [BREAK]
    response = json.dumps(
        [
            {"index": 0, "text": "Bonjour corrigé"},
            {"index": 1, "text": "On continue corrigé"},
        ]
    )

    call_count = 0

    def tracking_call_fn(messages):
        nonlocal call_count
        call_count += 1
        # Verify [BREAK] is not in the user message
        user_msg = messages[1]["content"]
        assert "[BREAK]" not in user_msg
        assert "2 numbered segments" in user_msg
        return response

    result = call_and_parse(batch, "sys", tracking_call_fn)

    assert len(result) == 3
    assert result[0]["text"] == "Bonjour corrigé"
    assert result[1]["speaker"] == "[BREAK]"
    assert result[1]["text"] == ""
    assert result[2]["text"] == "On continue corrigé"
    assert call_count == 1


def test_call_and_parse_all_breaks_no_llm_call():
    """A batch of only [BREAK] segments should not call the LLM."""
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
# build_manual_prompts_batched
# ──────────────────────────────────────────────


def test_batching_single_batch():
    segments = make_segments("Short")
    batches = build_manual_prompts_batched(segments, batch_minutes=15)
    assert len(batches) == 1


def test_batching_splits_by_duration():
    segments = [
        {"speaker": "Alice", "start": i * 600, "end": (i + 1) * 600, "text": f"Seg {i}"}
        for i in range(3)
    ]
    batches = build_manual_prompts_batched(segments, batch_minutes=15)
    assert len(batches) == 3


def test_batching_empty():
    batches = build_manual_prompts_batched([], batch_minutes=15)
    assert batches == []


# ──────────────────────────────────────────────
# build_manual_prompt
# ──────────────────────────────────────────────


def test_build_manual_prompt_contains_segments():
    segments = make_segments("Bonjour", "Au revoir")
    prompt = build_manual_prompt(segments, context="French podcast")
    assert "Bonjour" in prompt
    assert "Au revoir" in prompt


def test_build_manual_prompt_contains_context():
    segments = make_segments("Bonjour")
    prompt = build_manual_prompt(segments, context="Film music podcast")
    assert "Film music podcast" in prompt
