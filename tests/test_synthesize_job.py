"""The TTS step worker (`core/synthesize_job.run_generate`) and its manifest.

Runs the real job against a fake Qwen3-TTS model: speaker renames made after
the translation, the per-segment manifest that decides which WAVs are reused,
a run that dies half-way, and uploaded voice samples outranking extracted
ones. The speaker-map composition these rely on is pinned at the end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from podcodex.core import _utils, synthesize
from podcodex.core.synthesize_job import run_generate
from podcodex.core.versions import (
    compose_speaker_map,
    load_latest_speaker_map,
    save_speaker_map_version,
    save_version,
)
from tests.fixtures.episodes import make_episode, prov

SR = 24000


class _FakeTTS:
    def __init__(self, fail_on: int | None = None):
        self.calls = 0
        self.fail_on = fail_on

    def create_voice_clone_prompt(self, **kwargs):
        return kwargs["ref_audio"]

    def generate_voice_clone(self, **_kwargs):
        self.calls += 1
        if self.fail_on is not None and self.calls == self.fail_on:
            raise RuntimeError("model crashed")
        return [np.zeros(SR // 10, dtype=np.float32)], SR


@pytest.fixture
def episode(tmp_path, monkeypatch):
    """A show with one episode: a diarized source, a French translation still
    labelled with diarizer IDs, and a voice sample per speaker."""
    base, audio = make_episode(tmp_path)

    raw = [
        {"speaker": "SPEAKER_00", "text": "Bonjour.", "start": 0.0, "end": 1.0},
        {"speaker": "SPEAKER_01", "text": "Salut.", "start": 1.0, "end": 2.0},
        {"speaker": "SPEAKER_00", "text": "Ça va ?", "start": 2.0, "end": 3.0},
    ]
    save_version(base, "diarized_segments", raw, prov("diarized_segments"))
    save_version(base, "french", raw, prov("french"))

    samples = base.parent / _utils.VOICE_SAMPLES_DIR
    samples.mkdir()
    for label in ("SPEAKER_00", "SPEAKER_01"):
        sf.write(str(samples / f"{label}_00.wav"), np.zeros(SR), SR)

    monkeypatch.setattr(_utils, "check_vram", lambda *a, **k: None)
    monkeypatch.setattr(_utils, "free_vram", lambda: None)
    return base, audio


def _run(audio, model, *, keys=None, force=False, cancelled=lambda: False, monkeypatch):
    monkeypatch.setattr(synthesize, "load_tts_model", lambda model_size: model)
    return run_generate(
        progress_cb=lambda *_: None,
        cancelled=cancelled,
        audio_path=str(audio),
        output_dir=None,
        source_lang="french",
        source_version_id=None,
        model_size="0.6B",
        language="French",
        max_chunk_duration=20.0,
        force=force,
        only_speakers=None,
        keep_segment_keys=keys,
    )


def test_speakers_renamed_after_translation_are_synthesized(episode, monkeypatch):
    """The panel keys its selection on the renamed speakers (the translate
    read route applies the map); the job used to load the raw labels, miss
    every key and fail with "Selection dropped every segment"."""
    base, audio = episode
    save_speaker_map_version(base, {"SPEAKER_00": "Alice"})
    keys = [
        _utils.seg_key({"speaker": "Alice", "start": 0.0, "end": 1.0}),
        _utils.seg_key({"speaker": "Alice", "start": 2.0, "end": 3.0}),
    ]

    out = _run(audio, _FakeTTS(), keys=keys, monkeypatch=monkeypatch)

    assert out["count"] == 2


def test_a_crash_mid_run_keeps_the_segments_already_made(episode, monkeypatch):
    """The manifest used to be written once, after the loop; a crash (or the
    runner's SIGTERM after a cancel) orphaned every finished WAV."""
    _base, audio = episode
    with pytest.raises(RuntimeError, match="model crashed"):
        _run(audio, _FakeTTS(fail_on=3), monkeypatch=monkeypatch)

    rerun = _FakeTTS()
    out = _run(audio, rerun, monkeypatch=monkeypatch)

    assert out["reused"] == 2
    assert rerun.calls == 1


def test_segments_from_another_model_are_not_reused(episode, monkeypatch):
    """A cancelled run stamped its model on the whole manifest, so segments
    left from the previous model later passed as current."""
    _base, audio = episode
    _run(audio, _FakeTTS(), monkeypatch=monkeypatch)
    manifest = synthesize.load_manifest(audio.parent / "ep" / _utils.TTS_SEGMENTS_DIR)
    filename, entry = next(iter(manifest["segments"].items()))

    def current(model_size: str) -> bool:
        return synthesize.segment_is_current(
            manifest, filename, "Bonjour.", entry["voice_sample"], model_size, "French"
        )

    assert entry["model"] == "0.6B"
    assert current("0.6B")
    assert not current("1.7B")


def test_an_uploaded_sample_is_the_clone_reference(tmp_path):
    samples = tmp_path / _utils.VOICE_SAMPLES_DIR
    samples.mkdir()
    for name in ("Alice_00.wav", "Alice_01.wav", "Alice_custom_ab12cd34.wav"):
        sf.write(str(samples / name), np.zeros(SR), SR)

    loaded = synthesize.load_voice_samples(tmp_path, ["Alice"])

    assert Path(loaded["Alice"][0]["file"]).name == "Alice_custom_ab12cd34.wav"
    assert synthesize._sample_key(loaded, "Alice") == "Alice_custom_ab12cd34.wav"


def test_samples_under_the_diarizer_id_are_found_for_the_renamed_speaker(tmp_path):
    samples = tmp_path / _utils.VOICE_SAMPLES_DIR
    samples.mkdir()
    sf.write(str(samples / "SPEAKER_00_00.wav"), np.zeros(SR), SR)

    loaded = synthesize.load_voice_samples(
        tmp_path, ["Alice"], speaker_map={"SPEAKER_00": "Alice"}
    )

    assert list(loaded) == ["Alice"]


def test_unsupported_tts_language_names_the_supported_ones():
    assert synthesize._normalize_qwen_language("fr") == "French"
    with pytest.raises(ValueError, match="Supported"):
        synthesize._normalize_qwen_language("Klingon")


# ── speaker map composition ──────────────────────────────────────────────


def test_a_second_rename_does_not_undo_the_first(episode):
    """The editor sends only the pending renames, keyed by the label on
    screen. Replacing the map dropped every earlier entry."""
    base, _audio = episode
    save_speaker_map_version(base, {"SPEAKER_00": "Alice"})
    save_speaker_map_version(base, {"SPEAKER_01": "Bob"})
    save_speaker_map_version(base, {"Alice": "Alicia"})

    mapping = load_latest_speaker_map(base)
    assert mapping["SPEAKER_00"] == "Alicia"
    assert mapping["SPEAKER_01"] == "Bob"


def test_text_saved_under_an_intermediate_name_follows_the_rename(episode):
    """A correction of the transcript edited after the first rename carries
    "Alice"; the second rename must reach it too."""
    from podcodex.core.versions import apply_speaker_map

    base, _audio = episode
    save_speaker_map_version(base, {"SPEAKER_00": "Alice"})
    save_speaker_map_version(base, {"Alice": "Alicia"})

    mapping = load_latest_speaker_map(base)
    corrected = [{"speaker": "Alice", "text": "Bonjour."}]
    raw = [{"speaker": "SPEAKER_00", "text": "Bonjour."}]
    assert apply_speaker_map(corrected, mapping)[0]["speaker"] == "Alicia"
    assert apply_speaker_map(raw, mapping)[0]["speaker"] == "Alicia"


def test_swapping_two_names_follows_both():
    current = {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
    composed = compose_speaker_map(current, {"Alice": "Bob", "Bob": "Alice"})
    assert composed["SPEAKER_00"] == "Bob"
    assert composed["SPEAKER_01"] == "Alice"


def test_assemble_and_listings_read_the_translation_generate_used(episode):
    """Assemble and the listings used to rebuild segments from the canonical
    original-language source while generate synthesized the translation."""
    from podcodex.core.source import load_synth_source

    base, audio = episode
    save_version(
        base,
        "transcript",
        [{"speaker": "SPEAKER_00", "text": "Hello.", "start": 0.0, "end": 1.0}],
        prov("transcript"),
    )
    save_speaker_map_version(base, {"SPEAKER_00": "Alice"})

    translated, speaker_map = load_synth_source(str(audio), None, None, "french")
    canonical, _ = load_synth_source(str(audio), None, None)

    assert translated.segments[0]["text"] == "Bonjour."
    assert translated.segments[0]["speaker"] == "Alice"
    assert translated.step == "french"
    assert canonical.segments[0]["text"] == "Hello."
    assert canonical.step == "transcript"
    assert speaker_map == {"SPEAKER_00": "Alice"}


def test_assembled_audio_records_its_source_and_real_durations(
    episode, tmp_path, monkeypatch
):
    """The assembled version used to record only the pin (usually null), so
    nothing said it spoke the French translation; and the listing reported
    each segment's source span as the generated audio's length."""
    from podcodex.core.versions import list_versions
    from tests.fixtures.api_client import make_client

    base, audio = episode
    _run(audio, _FakeTTS(), monkeypatch=monkeypatch)
    client = make_client(tmp_path / "cfg", monkeypatch)

    listed = client.get(
        "/api/synthesize/generated-segments",
        params={"audio_path": str(audio), "source_lang": "french"},
    ).json()
    assert [round(seg["duration"], 2) for seg in listed] == [0.1, 0.1, 0.1]

    res = client.post(
        "/api/synthesize/assemble",
        json={"audio_path": str(audio), "source_lang": "french", "model_size": "0.6B"},
    )
    assert res.status_code == 200, res.text
    (version,) = list_versions(base, "synthesize")
    french = list_versions(base, "french")[0]
    assert version["params"]["source_step"] == "french"
    assert version["params"]["source_version_id"] == french["id"]


def test_a_legacy_manifest_keeps_its_model_per_entry(tmp_path):
    """Entries from before per-segment models used to fall back to the
    run-level model, which the next partial run overwrote."""
    import json

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model": "0.6B",
                "language": "French",
                "segments": {"a.wav": {"text_hash": "h", "voice_sample": "s"}},
            }
        ),
        encoding="utf-8",
    )
    manifest = synthesize.load_manifest(tmp_path)
    synthesize.record_segment(
        manifest,
        "b.wav",
        speaker="A",
        text="x",
        voice_sample_name="s",
        model_size="1.7B",
        language="French",
    )

    assert manifest["segments"]["a.wav"]["model"] == "0.6B"
    assert manifest["segments"]["b.wav"]["model"] == "1.7B"
