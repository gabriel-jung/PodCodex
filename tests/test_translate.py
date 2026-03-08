"""Tests for podcodex.core.translate — pure functions only (no API calls, no Ollama)."""

import json
from podcodex.core.translate import (
    _translate_batch,
    build_manual_translate_prompt,
    build_manual_translate_prompts_batched,
    translation_to_text,
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
# _translate_batch
# ──────────────────────────────────────────────


def test_translate_batch_happy_path():
    batch = make_segments("Bonjour", "Au revoir")
    response = json.dumps(
        [
            {"index": 0, "text": "Hello"},
            {"index": 1, "text": "Goodbye"},
        ]
    )
    result = _translate_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text_trad"] == "Hello"
    assert result[1]["text_trad"] == "Goodbye"


def test_translate_batch_keeps_original_text():
    """Source text field is never modified; LLM text goes to text_trad."""
    batch = make_segments("Bonjour le monde")
    response = json.dumps([{"index": 0, "text": "Hello world"}])
    result = _translate_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text"] == "Bonjour le monde"
    assert result[0]["text_trad"] == "Hello world"


def test_translate_batch_bad_json_does_not_crash():
    batch = make_segments("Bonjour le monde")
    result = _translate_batch(batch, "sys", make_call_fn("not json at all"))
    assert len(result) == 1
    assert result[0]["text"] == "Bonjour le monde"
    assert result[0]["text_trad"] == ""


def test_translate_batch_missing_index_falls_back():
    """If LLM omits a segment, that segment gets empty text_trad."""
    batch = make_segments("Premier", "Deuxième")
    response = json.dumps([{"index": 0, "text": "First"}])
    result = _translate_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text_trad"] == "First"
    assert result[1]["text"] == "Deuxième"
    assert result[1]["text_trad"] == ""


def test_translate_batch_strips_think_tags():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Hello"}])
    response = f"<think>some reasoning</think>\n{inner}"
    result = _translate_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text_trad"] == "Hello"


def test_translate_batch_strips_markdown_fences():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Hello"}])
    response = f"```json\n{inner}\n```"
    result = _translate_batch(batch, "sys", make_call_fn(response))
    assert result[0]["text_trad"] == "Hello"


# ──────────────────────────────────────────────
# build_manual_translate_prompts_batched
# ──────────────────────────────────────────────


def test_batching_single_batch():
    segments = make_segments("Short")
    batches = build_manual_translate_prompts_batched(segments, batch_minutes=15)
    assert len(batches) == 1


def test_batching_splits_by_duration():
    segments = [
        {"speaker": "Alice", "start": i * 600, "end": (i + 1) * 600, "text": f"Seg {i}"}
        for i in range(3)
    ]
    batches = build_manual_translate_prompts_batched(segments, batch_minutes=15)
    assert len(batches) == 3


def test_batching_groups_short_segments():
    segments = [
        {"speaker": "Alice", "start": i * 300, "end": (i + 1) * 300, "text": f"Seg {i}"}
        for i in range(4)
    ]
    batches = build_manual_translate_prompts_batched(segments, batch_minutes=15)
    assert len(batches) == 2


def test_batching_empty():
    batches = build_manual_translate_prompts_batched([], batch_minutes=15)
    assert batches == []


def test_batching_returns_prompt_strings():
    segments = make_segments("Bonjour")
    batches = build_manual_translate_prompts_batched(segments, batch_minutes=15)
    batch_segs, prompt = batches[0]
    assert isinstance(prompt, str)
    assert len(prompt) > 0


# ──────────────────────────────────────────────
# build_manual_translate_prompt
# ──────────────────────────────────────────────


def test_build_manual_translate_prompt_contains_segments():
    segments = make_segments("Bonjour", "Au revoir")
    prompt = build_manual_translate_prompt(segments, context="French podcast")
    assert "Bonjour" in prompt
    assert "Au revoir" in prompt


def test_build_manual_translate_prompt_contains_context():
    segments = make_segments("Bonjour")
    prompt = build_manual_translate_prompt(segments, context="Film music podcast")
    assert "Film music podcast" in prompt


def test_build_manual_translate_prompt_asks_for_index_and_text():
    """Translate prompt asks the LLM to return only index and translated text."""
    segments = make_segments("Bonjour")
    prompt = build_manual_translate_prompt(segments)
    assert '"index"' in prompt
    assert '"text"' in prompt
    assert "text_trad" not in prompt


# ──────────────────────────────────────────────
# translation_to_text
# ──────────────────────────────────────────────


def test_translation_to_text_source():
    segments = [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello"}]
    out = translation_to_text(segments)
    assert "Hello" in out


def test_translation_to_text_trad():
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


def test_translation_to_text_missing_trad_shows_placeholder():
    segments = [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Bonjour"}]
    out = translation_to_text(segments, lang="trad")
    assert "[not translated]" in out
