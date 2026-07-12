"""Tests for podcodex.core.synthesize — pure functions only (no GPU, no models)."""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from podcodex.core.synthesize import _split_text, assemble_episode


# ──────────────────────────────────────────────
# _split_text
# ──────────────────────────────────────────────


def test_split_empty_string():
    assert _split_text("", 3) == []


def test_split_single_chunk():
    assert _split_text("Hello world.", 1) == ["Hello world."]


def test_split_at_sentence_boundaries():
    text = "First sentence. Second sentence. Third sentence."
    result = _split_text(text, 3)
    assert len(result) == 3
    assert all(len(c) > 0 for c in result)


def test_split_preserves_all_content():
    text = "First sentence. Second sentence. Third sentence."
    result = _split_text(text, 3)
    # Reassembled content should match original words
    reassembled = " ".join(result)
    for word in text.replace(".", "").split():
        assert word in reassembled


def test_split_fewer_sentences_than_chunks():
    """Should return what's available without crashing."""
    text = "Only one sentence."
    result = _split_text(text, 5)
    assert 1 <= len(result) <= 5
    assert result[0] == "Only one sentence."


def test_split_falls_back_to_commas():
    text = "One, two, three, four, five"
    result = _split_text(text, 3)
    assert len(result) >= 2
    assert all(len(c) > 0 for c in result)


def test_split_exclamation_and_question_marks():
    text = "Really? Yes! Absolutely."
    result = _split_text(text, 3)
    assert len(result) == 3


def test_split_no_breakpoints_returns_single_chunk():
    text = "A sentence with no punctuation at all"
    result = _split_text(text, 3)
    assert len(result) == 1
    assert result[0] == text


# ──────────────────────────────────────────────
# assemble_episode
# ──────────────────────────────────────────────


SR = 16000


def make_wav(tmp_path: Path, name: str, duration: float) -> Path:
    """Write a silent WAV file and return its path."""
    path = tmp_path / name
    audio = np.zeros(int(duration * SR), dtype=np.float32)
    sf.write(str(path), audio, SR)
    return path


def make_generated(tmp_path, segments_data):
    """Build a generated list of segment dicts with "audio_file" set."""
    result = []
    for i, (start, end) in enumerate(segments_data):
        wav = make_wav(tmp_path, f"{i:04d}.wav", end - start)
        result.append(
            {
                "speaker": "Alice",
                "start": start,
                "end": end,
                "text": "Hello",
                "audio_file": wav,
                "sample_rate": SR,
            }
        )
    return result


def test_assemble_silence_strategy_same_speaker(tmp_path):
    # Speaker-aware silence: within-turn pause = max(silence_duration * 0.4,
    # 0.05). Two Alice segments → 2s + 0.2s + 2s = 4.2s.
    generated = make_generated(tmp_path, [(0, 2), (5, 7)])
    out_path = tmp_path / "out.wav"
    out = assemble_episode(
        generated, out_path, strategy="silence", silence_duration=0.5
    )
    assert out.exists()
    audio, sr = sf.read(str(out))
    assert abs(len(audio) / sr - 4.2) < 0.1


def test_assemble_silence_strategy_cross_speaker(tmp_path):
    # Different speakers → full silence_duration between turns.
    # 2s + 0.5s + 2s = 4.5s.
    generated = make_generated(tmp_path, [(0, 2), (5, 7)])
    generated[1]["speaker"] = "Bob"
    out_path = tmp_path / "out.wav"
    out = assemble_episode(
        generated, out_path, strategy="silence", silence_duration=0.5
    )
    assert out.exists()
    audio, sr = sf.read(str(out))
    assert abs(len(audio) / sr - 4.5) < 0.1


def test_assemble_original_timing_strategy(tmp_path):
    generated = make_generated(tmp_path, [(0, 2), (5, 7)])
    out_path = tmp_path / "out.wav"
    out = assemble_episode(generated, out_path, strategy="original_timing")
    assert out.exists()
    audio, sr = sf.read(str(out))
    # Should be at least as long as the last segment's generated audio end
    assert len(audio) / sr > 0


def test_assemble_original_timing_no_blank_lead_in_for_narrowed_selection(tmp_path):
    # First segment starts at t=12s but selection is narrow. Output should
    # anchor at 12s, not pad 12s of silence at the front. Two segments at
    # (12,14) and (15,17) → 2s + 1s gap + 2s = 5s, not 17s.
    generated = make_generated(tmp_path, [(12.0, 14.0), (15.0, 17.0)])
    out_path = tmp_path / "out.wav"
    out = assemble_episode(generated, out_path, strategy="original_timing")
    audio, sr = sf.read(str(out))
    assert abs(len(audio) / sr - 5.0) < 0.1


def test_assemble_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="No generated segments"):
        assemble_episode([], tmp_path / "out.wav", strategy="silence")


def test_assemble_unknown_strategy_raises(tmp_path):
    generated = make_generated(tmp_path, [(0, 2)])
    with pytest.raises(ValueError, match="Unknown strategy"):
        assemble_episode(generated, tmp_path / "out.wav", strategy="invalid")


def test_assemble_writes_to_output_path(tmp_path):
    generated = make_generated(tmp_path, [(0, 2)])
    out_path = tmp_path / "custom_name.wav"
    out = assemble_episode(generated, out_path, strategy="silence")
    assert out == out_path
    assert out.is_file()
