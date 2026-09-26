"""The shared correct / translate run path (routes and the batch runner)."""

from __future__ import annotations

from tests.fixtures.llm import fake_llm_run
from podcodex.core.source import load_source
from podcodex.core.versions import get_version_provenance, save_version

LLM = fake_llm_run("qwen3:4b")


def _episode(tmp_path):
    show = tmp_path / "show"
    (show / "ep").mkdir(parents=True)
    audio = show / "ep.mp3"
    audio.touch()
    base = show / "ep" / "ep"
    save_version(
        base,
        "transcript",
        [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "bonjour"}],
        {
            "step": "transcript",
            "type": "raw",
            "model": "large-v3",
            "params": {"language": "fr"},
        },
    )
    return str(audio), base


def test_correct_records_the_language_and_the_failures_with_the_version(
    tmp_path, monkeypatch
):
    import podcodex.core.correct as core_correct
    from podcodex.core.llm_failures import get_step

    audio, base = _episode(tmp_path)

    def fake(segments, records_out, **kwargs):
        # A run with a rejected batch, as run_llm_step hands it back.
        records_out.append({"batch": 1, "status": "rejected", "input": []})
        assert kwargs["source_lang"] == "French"
        return segments

    monkeypatch.setattr(core_correct, "correct_segments", fake)
    source = load_source(audio, None, None, step="transcript")
    _segs, vid = core_correct.correct_and_save(
        source, LLM, audio_path=audio, source_lang="English"
    )

    params = get_version_provenance(base, vid, "corrected")["params"]
    assert params["source_lang"] == "French"
    assert params["llm_mode"] == "ollama"
    assert get_step(audio, None, "corrected")["version_id"] == vid


def test_translate_saves_under_the_normalized_language(tmp_path, monkeypatch):
    import podcodex.core.translate as core_translate
    from podcodex.core.versions import list_versions

    audio, base = _episode(tmp_path)
    monkeypatch.setattr(core_translate, "translate_segments", lambda segs, **_k: segs)
    source = load_source(audio, None, None)
    _segs, vid = core_translate.translate_and_save(
        source, LLM, audio_path=audio, target_lang="Brazilian Portuguese"
    )
    assert [v["id"] for v in list_versions(base, "brazilian_portuguese")] == [vid]
