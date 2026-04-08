"""Tests for podcodex.core.correct — module-specific prompt helpers."""

from podcodex.core.correct import build_manual_prompt


def test_build_manual_prompt_contains_segments_and_context():
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 10.0, "text": "Bonjour"},
        {"speaker": "Alice", "start": 10.0, "end": 20.0, "text": "Au revoir"},
    ]
    prompt = build_manual_prompt(segments, context="Film music podcast")
    assert "Bonjour" in prompt
    assert "Au revoir" in prompt
    assert "Film music podcast" in prompt
