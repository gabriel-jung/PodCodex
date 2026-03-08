"""Tests for podcodex.core.polish — pure functions only (no API calls, no Ollama)."""

import json
from podcodex.core.polish import (
    _polish_batch,
    build_manual_polish_prompt,
    build_manual_polish_prompts_batched,
)


def make_call_fn(response: str):
    return lambda messages: response


def make_segments(*texts):
    return [
        {"speaker": "Alice", "start": i * 10.0, "end": (i + 1) * 10.0, "text": t}
        for i, t in enumerate(texts)
    ]


# ──────────────────────────────────────────────
# _polish_batch
# ──────────────────────────────────────────────


def test_polish_batch_happy_path():
    batch = make_segments("Bonjour", "Au revoir")
    response = json.dumps(
        [
            {"index": 0, "text": "Bonjour corrigé"},
            {"index": 1, "text": "Au revoir corrigé"},
        ]
    )
    result = _polish_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Bonjour corrigé"
    assert result[1]["text"] == "Au revoir corrigé"
    assert "text_trad" not in result[0]


def test_polish_batch_no_text_trad():
    """Polish never produces a text_trad field."""
    batch = make_segments("Bonjour")
    response = json.dumps([{"index": 0, "text": "Bonjour corrigé"}])
    result = _polish_batch(batch, "sys", make_call_fn(response))
    assert "text_trad" not in result[0]


def test_polish_batch_truncation_falls_back_to_original():
    """If LLM returns a suspiciously short correction, keep the original text."""
    batch = make_segments(
        "Une longue phrase originale avec beaucoup de mots importants"
    )
    response = json.dumps([{"index": 0, "text": "Hi"}])
    result = _polish_batch(batch, "sys", make_call_fn(response))
    assert (
        result[0]["text"]
        == "Une longue phrase originale avec beaucoup de mots importants"
    )


def test_polish_batch_bad_json_does_not_crash():
    batch = make_segments("Bonjour le monde")
    result = _polish_batch(batch, "sys", make_call_fn("not json at all"))
    assert len(result) == 1
    assert result[0]["text"] == "Bonjour le monde"


def test_polish_batch_missing_index_falls_back():
    """If LLM omits a segment, that segment keeps its original text."""
    batch = make_segments("Premier", "Deuxième")
    response = json.dumps([{"index": 0, "text": "Premier corrigé"}])
    result = _polish_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Premier corrigé"
    assert result[1]["text"] == "Deuxième"


def test_polish_batch_strips_think_tags():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Bonjour corrigé"}])
    response = f"<think>some reasoning</think>\n{inner}"
    result = _polish_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Bonjour corrigé"


def test_polish_batch_strips_markdown_fences():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Bonjour corrigé"}])
    response = f"```json\n{inner}\n```"
    result = _polish_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Bonjour corrigé"


# ──────────────────────────────────────────────
# build_manual_polish_prompts_batched
# ──────────────────────────────────────────────


def test_polish_batching_single_batch():
    segments = make_segments("Short")
    batches = build_manual_polish_prompts_batched(segments, batch_minutes=15)
    assert len(batches) == 1


def test_polish_batching_splits_by_duration():
    segments = [
        {"speaker": "Alice", "start": i * 600, "end": (i + 1) * 600, "text": f"Seg {i}"}
        for i in range(3)
    ]
    batches = build_manual_polish_prompts_batched(segments, batch_minutes=15)
    assert len(batches) == 3


def test_polish_batching_empty():
    batches = build_manual_polish_prompts_batched([], batch_minutes=15)
    assert batches == []


# ──────────────────────────────────────────────
# build_manual_polish_prompt
# ──────────────────────────────────────────────


def test_build_manual_polish_prompt_contains_segments():
    segments = make_segments("Bonjour", "Au revoir")
    prompt = build_manual_polish_prompt(segments, context="French podcast")
    assert "Bonjour" in prompt
    assert "Au revoir" in prompt


def test_build_manual_polish_prompt_contains_context():
    segments = make_segments("Bonjour")
    prompt = build_manual_polish_prompt(segments, context="Film music podcast")
    assert "Film music podcast" in prompt


def test_build_manual_polish_prompt_no_text_trad():
    """Polish prompt must not ask for a text_trad field."""
    segments = make_segments("Bonjour")
    prompt = build_manual_polish_prompt(segments)
    assert "text_trad" not in prompt
