"""Tests for podcodex.core.translate — pure functions only (no API calls, no Ollama)."""

import json
from podcodex.core.translate import (
    _translate_batch,
    build_manual_prompt,
    build_manual_prompts_batched,
    translation_to_text,
)


def make_call_fn(response: str):
    """Return a call_fn that always returns the given string."""
    return lambda messages: response


def make_segments(*texts):
    """Build minimal segment dicts for testing."""
    return [
        {"speaker": "Alice", "start": i * 10.0, "end": (i + 1) * 10.0, "text": t}
        for i, t in enumerate(texts)
    ]


# ──────────────────────────────────────────────
# _translate_batch
# ──────────────────────────────────────────────


def test_translate_batch_happy_path():
    batch = make_segments("Bonjour", "Au revoir")
    response = json.dumps(
        [
            {"index": 0, "text": "Bonjour", "text_trad": "Hello"},
            {"index": 1, "text": "Au revoir", "text_trad": "Goodbye"},
        ]
    )
    result = _translate_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text_trad"] == "Hello"
    assert result[1]["text_trad"] == "Goodbye"


def test_translate_batch_bad_json_does_not_crash():
    batch = make_segments("Bonjour le monde")
    result = _translate_batch(batch, "sys", make_call_fn("not json at all"))
    assert len(result) == 1
    assert result[0]["text"] == "Bonjour le monde"


def test_translate_batch_truncation_falls_back_to_original():
    """If LLM returns a suspiciously short correction, keep the original text."""
    batch = make_segments(
        "Une longue phrase originale avec beaucoup de mots importants"
    )
    response = json.dumps([{"index": 0, "text": "Hi", "text_trad": "Hi"}])
    result = _translate_batch(batch, "sys", make_call_fn(response))
    assert (
        result[0]["text"]
        == "Une longue phrase originale avec beaucoup de mots importants"
    )


def test_translate_batch_missing_index_falls_back():
    """If LLM omits a segment, that segment keeps its original text."""
    batch = make_segments("Premier", "Deuxième")
    response = json.dumps(
        [
            {"index": 0, "text": "Premier", "text_trad": "First"},
            # index 1 missing
        ]
    )
    result = _translate_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text_trad"] == "First"
    assert result[1]["text"] == "Deuxième"  # kept original


def test_translate_batch_strips_think_tags():
    """LLM output wrapped in <think>...</think> should be cleaned before parsing."""
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Bonjour", "text_trad": "Hello"}])
    response = f"<think>some reasoning</think>\n{inner}"
    result = _translate_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text_trad"] == "Hello"


def test_translate_batch_strips_markdown_fences():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Bonjour", "text_trad": "Hello"}])
    response = f"```json\n{inner}\n```"
    result = _translate_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text_trad"] == "Hello"


def test_translate_batch_polish_mode_no_text_trad():
    batch = make_segments("Bonjour")
    response = json.dumps([{"index": 0, "text": "Bonjour corrigé"}])
    result = _translate_batch(batch, "sys", make_call_fn(response), task="polish")
    assert result[0]["text"] == "Bonjour corrigé"
    assert "text_trad" not in result[0]


# ──────────────────────────────────────────────
# build_manual_prompts_batched
# ──────────────────────────────────────────────


def test_batching_single_batch():
    segments = make_segments("Short")  # 10s, well under 15 min
    batches = build_manual_prompts_batched(segments, batch_minutes=15)
    assert len(batches) == 1


def test_batching_splits_by_duration():
    # 3 segments × 10 min each = 30 min → each exceeds 15 min limit alone, so 3 batches
    segments = [
        {"speaker": "Alice", "start": i * 600, "end": (i + 1) * 600, "text": f"Seg {i}"}
        for i in range(3)
    ]
    batches = build_manual_prompts_batched(segments, batch_minutes=15)
    assert len(batches) == 3


def test_batching_groups_short_segments():
    # 4 segments × 5 min each → should fit 3 per batch at 15 min → 2 batches
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


def test_build_manual_prompt_polish_mode():
    segments = make_segments("Bonjour")
    prompt = build_manual_prompt(segments, task="polish")
    # Polish mode should not ask for a text_trad field
    assert "text_trad" not in prompt


# ──────────────────────────────────────────────
# translation_to_text
# ──────────────────────────────────────────────


def test_translation_to_text_both():
    segments = [
        {
            "speaker": "Alice",
            "start": 0.0,
            "end": 2.0,
            "text": "Bonjour",
            "text_trad": "Hello",
        }
    ]
    out = translation_to_text(segments, lang="both")
    assert "Bonjour" in out
    assert "Hello" in out


def test_translation_to_text_original_only():
    segments = [
        {
            "speaker": "Alice",
            "start": 0.0,
            "end": 2.0,
            "text": "Bonjour",
            "text_trad": "Hello",
        }
    ]
    out = translation_to_text(segments, lang="fr")
    assert "Bonjour" in out
    assert "Hello" not in out


def test_translation_to_text_trad_only():
    segments = [
        {
            "speaker": "Alice",
            "start": 0.0,
            "end": 2.0,
            "text": "Bonjour",
            "text_trad": "Hello",
        }
    ]
    out = translation_to_text(segments, lang="trad")
    assert "Hello" in out
    assert "Bonjour" not in out


def test_translation_to_text_missing_trad_shows_placeholder():
    segments = [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Bonjour"}]
    out = translation_to_text(segments, lang="trad")
    assert "[not translated]" in out
