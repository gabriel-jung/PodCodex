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


def test_the_prompt_carries_the_shared_output_contract():
    """The contract lives in one place (`llm.output_format_rules`), next to
    the parser that relies on it; the prompt must include it as is."""
    from podcodex.core.llm import output_format_rules

    prompt = build_manual_prompt([{"speaker": "A", "start": 0, "end": 1, "text": "x"}])
    assert output_format_rules("untranslatable") in prompt


def test_an_answer_in_the_requested_shape_maps_by_position():
    """What the contract asks for is what the parser applies: a JSON array of
    `{"text"}` objects, mapped by position whatever indices it carries."""
    from podcodex.core.llm import apply_corrections, parse_llm_response

    segments = [
        {"speaker": "A", "start": 0.0, "end": 1.0, "text": "Hello"},
        {"speaker": "A", "start": 1.0, "end": 2.0, "text": "Goodbye"},
    ]
    parsed = parse_llm_response(
        '[{"text": "Bonjour"}, {"index": 7, "text": "Au revoir"}]'
    )
    out = apply_corrections(segments, parsed, min_length_ratio=0)
    assert [s["text"] for s in out] == ["Bonjour", "Au revoir"]
