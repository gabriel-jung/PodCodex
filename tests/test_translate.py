"""Tests for podcodex.core.translate — module-specific prompt helpers."""

from podcodex.core.translate import build_manual_prompt


def test_build_manual_prompt_contains_segments_and_context():
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 10.0, "text": "Bonjour"},
        {"speaker": "Alice", "start": 10.0, "end": 20.0, "text": "Au revoir"},
    ]
    prompt = build_manual_prompt(segments, context="French podcast")
    assert "Bonjour" in prompt
    assert "Au revoir" in prompt
    assert "French podcast" in prompt


def test_build_manual_prompt_asks_for_index_and_text():
    """Translate prompt asks the LLM to return index and translated text."""
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 10.0, "text": "Bonjour"},
    ]
    prompt = build_manual_prompt(segments)
    assert '"index"' in prompt
    assert '"text"' in prompt
