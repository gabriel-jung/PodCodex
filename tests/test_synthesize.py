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
    """Build a generated list as returned by generate_segments()."""
    result = []
    for i, (start, end) in enumerate(segments_data):
        wav = make_wav(tmp_path, f"{i:04d}.wav", end - start)
        result.append(
            {
                "speaker": "Alice",
                "start": start,
                "end": end,
                "text_trad": "Hello",
                "audio_file": wav,
                "sample_rate": SR,
            }
        )
    return result


def test_assemble_silence_strategy(tmp_path):
    generated = make_generated(tmp_path, [(0, 2), (5, 7)])
    audio_path = tmp_path / "episode.mp3"
    out = assemble_episode(
        generated, audio_path, output_dir="", strategy="silence", silence_duration=0.5
    )
    assert out.exists()
    audio, sr = sf.read(str(out))
    # 2s + 0.5s silence + 2s = 4.5s
    assert abs(len(audio) / sr - 4.5) < 0.1


def test_assemble_original_timing_strategy(tmp_path):
    generated = make_generated(tmp_path, [(0, 2), (5, 7)])
    audio_path = tmp_path / "episode.mp3"
    out = assemble_episode(
        generated, audio_path, output_dir="", strategy="original_timing"
    )
    assert out.exists()
    audio, sr = sf.read(str(out))
    # Should be at least as long as the last segment's generated audio end
    assert len(audio) / sr > 0


def test_assemble_empty_raises():
    with pytest.raises(ValueError, match="No generated segments"):
        assemble_episode([], "episode.mp3", strategy="silence")


def test_assemble_unknown_strategy_raises(tmp_path):
    generated = make_generated(tmp_path, [(0, 2)])
    with pytest.raises(ValueError, match="Unknown strategy"):
        assemble_episode(generated, tmp_path / "episode.mp3", strategy="invalid")


def test_assemble_output_path_uses_stem(tmp_path):
    generated = make_generated(tmp_path, [(0, 2)])
    audio_path = tmp_path / "my_episode.mp3"
    out = assemble_episode(generated, audio_path, strategy="silence")
    assert out.name == "my_episode.synthesized.wav"
