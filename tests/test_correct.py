"""Tests for podcodex.core.correct — module-specific prompt helpers."""

from podcodex.core._utils import BREAK_SPEAKER
from podcodex.core.correct import build_manual_prompt, build_manual_prompts_batched
from podcodex.api.routes._helpers import format_prompt_batches


def test_build_manual_prompt_contains_segments_and_context():
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 10.0, "text": "Bonjour"},
        {"speaker": "Alice", "start": 10.0, "end": 20.0, "text": "Au revoir"},
    ]
    prompt = build_manual_prompt(segments, context="Film music podcast")
    assert "Bonjour" in prompt
    assert "Au revoir" in prompt
    assert "Film music podcast" in prompt


def test_format_prompt_batches_segment_count_excludes_breaks():
    # A batch with a [BREAK] marker: segment_count must report the real
    # (non-break) count so it matches the prompt's "exactly N entries" line
    # and the apply-path count check. Counting the break would make the
    # per-batch UI validation reject a correct LLM response.
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 10.0, "text": "Bonjour"},
        {"speaker": BREAK_SPEAKER, "start": 10.0, "end": 12.0, "text": "[music]"},
        {"speaker": "Alice", "start": 12.0, "end": 20.0, "text": "Au revoir"},
    ]
    batches = build_manual_prompts_batched(segments)
    formatted = format_prompt_batches(batches)
    assert len(formatted) == 1
    assert formatted[0]["segment_count"] == 2  # two real segments, break excluded
    assert "exactly 2 entries" in formatted[0]["prompt"]
