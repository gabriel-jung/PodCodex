"""Tests for podcodex.core.synthesize — pure functions only (no GPU, no models)."""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from podcodex.core.synthesize import split_text, assemble_episode


# ──────────────────────────────────────────────
# split_text
# ──────────────────────────────────────────────


def test_split_empty_string():
    assert split_text("", 3) == []


def test_split_single_chunk():
    assert split_text("Hello world.", 1) == ["Hello world."]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "First sentence. Second sentence. Third sentence.",
            ["First sentence.", "Second sentence.", "Third sentence."],
        ),
        ("Really? Yes! Absolutely.", ["Really?", "Yes!", "Absolutely."]),
        # No sentence ends to split on: commas are the fallback.
        ("One, two, three, four, five", ["One, two,", "three, four,", "five"]),
    ],
)
def test_split_into_three_chunks(text, expected):
    assert split_text(text, 3) == expected


def test_split_fewer_sentences_than_chunks():
    """Should return what's available without crashing."""
    text = "Only one sentence."
    result = split_text(text, 5)
    assert 1 <= len(result) <= 5
    assert result[0] == "Only one sentence."


def test_split_no_breakpoints_returns_single_chunk():
    text = "A sentence with no punctuation at all"
    result = split_text(text, 3)
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


# ── Voice-sample filenames are confined to voice_samples/ ────────────────


def test_speaker_file_slug_neutralizes_paths_and_globs():
    from podcodex.core._utils import speaker_file_slug

    assert speaker_file_slug("../../x") == ".._.._x"
    assert speaker_file_slug("/tmp/x") == "_tmp_x"
    assert speaker_file_slug(r"..\..\x") == ".._.._x"
    assert speaker_file_slug("*") == "_"
    assert speaker_file_slug("[a-z]") == "_a-z_"
    # Ordinary labels keep the filenames they already have on disk.
    assert speaker_file_slug("Dr. Smith") == "Dr. Smith"
    assert speaker_file_slug("SPEAKER_00") == "SPEAKER_00"
    assert speaker_file_slug("") == ""


def test_extract_selected_samples_keeps_hostile_speaker_inside_dir(
    tmp_path, monkeypatch
):
    """A subtitle-supplied "../../x" speaker must not write outside the dir."""
    from podcodex.core import synthesize as synth

    show = tmp_path / "show"
    (show / "ep").mkdir(parents=True)
    audio = show / "ep.mp3"
    audio.touch()

    written: list = []
    monkeypatch.setattr(
        synth,
        "_extract_clip",
        lambda src, seg, out: (
            written.append(out),
            out.write_bytes(b""),
            {"file": out, "duration": 1.0, "text": ""},
        )[-1],
    )
    # samples_dir.glob("../../x_*.wav") resolves to <show>/x_*.wav: pathlib
    # follows ".." segments, so an unslugged label would unlink this file.
    victim = show / "x_00.wav"
    victim.write_bytes(b"keep")

    synth.extract_selected_samples(
        audio,
        [{"speaker": "../../x", "start": 0.0, "end": 1.0, "text": "hi"}],
    )
    samples_dir = show / "ep" / "voice_samples"
    assert written and all(p.parent == samples_dir for p in written)
    assert victim.exists()
