"""Tests for podcodex.core.translate — pure functions only (no API calls, no Ollama)."""

import json
from podcodex.core._utils import call_and_parse, segments_to_text
from podcodex.core.translate import (
    build_manual_prompt,
    build_manual_prompts_batched,
)


def make_call_fn(response: str):
    """Return a call_fn that always returns the given string."""
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
            {"index": 0, "text": "Hello"},
            {"index": 1, "text": "Goodbye"},
        ]
    )
    result = call_and_parse(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Hello"
    assert result[1]["text"] == "Goodbye"


def test_call_and_parse_overwrites_text():
    """LLM result replaces the text field."""
    batch = make_segments("Bonjour le monde")
    response = json.dumps([{"index": 0, "text": "Hello world"}])
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "Hello world"


def test_call_and_parse_bad_json_keeps_original():
    batch = make_segments("Bonjour le monde")
    result = call_and_parse(batch, "sys", make_call_fn("not json at all"))
    assert len(result) == 1
    assert result[0]["text"] == "Bonjour le monde"


def test_call_and_parse_missing_index_keeps_original():
    """If LLM omits a segment, that segment keeps its original text."""
    batch = make_segments("Premier", "Deuxieme")
    response = json.dumps([{"index": 0, "text": "First"}])
    result = call_and_parse(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "First"
    assert result[1]["text"] == "Deuxieme"


def test_call_and_parse_strips_think_tags():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Hello"}])
    response = f"<think>some reasoning</think>\n{inner}"
    result = call_and_parse(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Hello"


def test_call_and_parse_strips_markdown_fences():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Hello"}])
    response = f"```json\n{inner}\n```"
    result = call_and_parse(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Hello"


def test_call_and_parse_truncation_guard():
    """Segments truncated below min_length_ratio keep original text."""
    batch = make_segments("This is a very long sentence that should not be truncated")
    response = json.dumps([{"index": 0, "text": "Short"}])
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0.7)
    assert (
        result[0]["text"] == "This is a very long sentence that should not be truncated"
    )


def test_call_and_parse_no_truncation_guard_when_disabled():
    """min_length_ratio=0 disables the truncation guard."""
    batch = make_segments("This is a very long sentence that should not be truncated")
    response = json.dumps([{"index": 0, "text": "Short"}])
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "Short"


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


def test_batching_groups_short_segments():
    segments = [
        {"speaker": "Alice", "start": i * 300, "end": (i + 1) * 300, "text": f"Seg {i}"}
        for i in range(4)
    ]
    batches = build_manual_prompts_batched(segments, batch_minutes=15)
    assert len(batches) == 2


def test_batching_empty():
    batches = build_manual_prompts_batched([], batch_minutes=15)
    assert batches == []


def test_batching_returns_prompt_strings():
    segments = make_segments("Bonjour")
    batches = build_manual_prompts_batched(segments, batch_minutes=15)
    batch_segs, prompt = batches[0]
    assert isinstance(prompt, str)
    assert len(prompt) > 0


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


def test_build_manual_prompt_asks_for_index_and_text():
    """Translate prompt asks the LLM to return only index and translated text."""
    segments = make_segments("Bonjour")
    prompt = build_manual_prompt(segments)
    assert '"index"' in prompt
    assert '"text"' in prompt


# ──────────────────────────────────────────────
# segments_to_text
# ──────────────────────────────────────────────


def test_segments_to_text():
    segments = [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"}]
    out = segments_to_text(segments)
    assert "Hello" in out
