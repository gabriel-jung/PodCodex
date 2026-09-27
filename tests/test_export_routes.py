"""Export routes: text / SRT / VTT formatters, confinement to registered
shows, and which version an export reads."""

from pathlib import Path

import pytest

from podcodex.core.versions import save_version
from tests.fixtures.episodes import audio_and_output_dir


@pytest.fixture
def client(tmp_path, monkeypatch):
    from tests.fixtures.api_client import make_client

    return make_client(tmp_path, monkeypatch)


def _register_show_folder(ep_dir: str) -> None:
    """Register the episode's show folder so the export guard accepts it.

    The export routes confine ``audio_path``/``output_dir`` to a registered
    show root, the same way /api/audio/file does.
    """
    from podcodex.core.app_config import AppConfig, save_config

    save_config(AppConfig(show_folders=[str(Path(ep_dir).parent)]))


def test_export_outside_show_root_is_refused(client, tmp_path):
    """An unregistered directory is not zippable or readable through export."""
    audio, ep_dir = audio_and_output_dir(tmp_path)
    segs = [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "secret"}]
    save_version(
        Path(ep_dir) / "ep", "transcript", segs, {"step": "transcript", "type": "raw"}
    )
    # No show folder registered at all.
    r = client.get(
        "/api/export/text", params={"audio_path": audio, "source": "transcript"}
    )
    assert r.status_code == 403
    r = client.get("/api/export/zip", params={"audio_path": audio})
    assert r.status_code == 403

    # Registering an unrelated folder does not open the door either.
    other = tmp_path / "other"
    other.mkdir()
    from podcodex.core.app_config import AppConfig, save_config

    save_config(AppConfig(show_folders=[str(other)]))
    r = client.get("/api/export/zip", params={"audio_path": audio})
    assert r.status_code == 403


def test_export_text_from_transcript(client, tmp_path):
    audio, ep_dir = audio_and_output_dir(tmp_path)
    _register_show_folder(ep_dir)
    segs = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "hello"},
        {"speaker": "Bob", "start": 2.0, "end": 4.0, "text": "world"},
    ]
    save_version(
        Path(ep_dir) / "ep", "transcript", segs, {"step": "transcript", "type": "raw"}
    )

    r = client.get(
        "/api/export/text",
        params={"audio_path": audio, "source": "transcript"},
    )
    assert r.status_code == 200
    assert "Alice" in r.text
    assert "hello" in r.text
    assert "Bob" in r.text


def test_export_srt_has_timestamps(client, tmp_path):
    audio, ep_dir = audio_and_output_dir(tmp_path)
    _register_show_folder(ep_dir)
    segs = [{"speaker": "A", "start": 0.0, "end": 1.5, "text": "go"}]
    save_version(
        Path(ep_dir) / "ep", "transcript", segs, {"step": "transcript", "type": "raw"}
    )

    r = client.get(
        "/api/export/srt",
        params={"audio_path": audio, "source": "transcript"},
    )
    assert r.status_code == 200
    assert "-->" in r.text
    assert "00:00:00,000" in r.text


def test_export_vtt_has_header(client, tmp_path):
    audio, ep_dir = audio_and_output_dir(tmp_path)
    _register_show_folder(ep_dir)
    segs = [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hey"}]
    save_version(
        Path(ep_dir) / "ep", "transcript", segs, {"step": "transcript", "type": "raw"}
    )

    r = client.get(
        "/api/export/vtt",
        params={"audio_path": audio, "source": "transcript"},
    )
    assert r.status_code == 200
    assert r.text.startswith("WEBVTT")


def test_export_missing_source_returns_404(client, tmp_path):
    audio, ep_dir = audio_and_output_dir(tmp_path)
    _register_show_folder(ep_dir)
    r = client.get(
        "/api/export/text",
        params={"audio_path": audio, "source": "transcript"},
    )
    assert r.status_code == 404


def test_a_loaded_version_has_the_shape_of_the_default_one(
    client, tmp_path, monkeypatch
):
    """A version picked in a selector must match /segments: flags always, and
    the speaker map for translations (the synth panel keys on mapped names)."""
    import podcodex.core.versions as versions_mod

    audio, ep_dir = audio_and_output_dir(tmp_path)
    base = Path(ep_dir) / "ep"
    segs = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "salut"}]
    fr = save_version(base, "french", segs, {"step": "french", "type": "raw"})
    tr = save_version(base, "transcript", segs, {"step": "transcript", "type": "raw"})
    monkeypatch.setattr(
        versions_mod, "load_latest_speaker_map", lambda _b: {"SPEAKER_00": "Ann"}
    )

    r = client.get(
        f"/api/translate/versions/{fr}", params={"audio_path": audio, "lang": "French"}
    )
    assert r.status_code == 200, r.text
    default = client.get(
        "/api/translate/segments", params={"audio_path": audio, "lang": "French"}
    ).json()
    assert r.json() == default
    assert r.json()[0]["speaker"] == "Ann"
    assert "flagged" in r.json()[0]

    r = client.get(f"/api/transcribe/versions/{tr}", params={"audio_path": audio})
    assert r.json()[0]["speaker"] == "SPEAKER_00"
    assert "flagged" in r.json()[0]


def test_batch_fixes_patch_the_run_that_recorded_them(client, tmp_path):
    """An older hand-edited version outranks the auto run in the default
    pick; the fixes must still land on the auto run's own version."""
    from podcodex.core.llm_failures import save_batch_records, stamp_run_version
    from podcodex.core.versions import load_latest, load_version

    audio, ep_dir = audio_and_output_dir(tmp_path)
    base = Path(ep_dir) / "ep"
    seg = {"speaker": "A", "start": 0.0, "end": 1.0}
    save_version(
        base,
        "corrected",
        [{**seg, "text": "edited zero"}, {**seg, "text": "edited one"}],
        {"step": "corrected", "type": "validated", "manual_edit": True},
    )
    run = save_version(
        base,
        "corrected",
        [{**seg, "text": "auto zero"}, {**seg, "text": "auto one"}],
        {"step": "corrected", "type": "raw", "model": "m"},
    )
    save_batch_records(
        base,
        "corrected",
        model="m",
        mode="ollama",
        records=[
            {"batch": 1, "status": "ok", "input": [{"index": 0}]},
            {"batch": 2, "status": "rejected", "input": [{"index": 1}]},
        ],
    )
    stamp_run_version(audio, None, "corrected", run)
    assert load_latest(base, "corrected")[0]["text"] == "edited zero"

    r = client.post(
        "/api/correct/apply-batches",
        json={
            "audio_path": audio,
            "fixes": [{"batch": 2, "corrections": [{"text": "fixed one"}]}],
        },
    )
    assert r.status_code == 200, r.text
    from podcodex.core.versions import list_versions

    newest = list_versions(base, "corrected")[0]
    texts = [s["text"] for s in load_version(base, "corrected", newest["id"])]
    assert texts == ["auto zero", "fixed one"]


def test_export_an_episode_without_audio(client, tmp_path):
    """YouTube subtitle-only episodes send audio_path "" and an output_dir."""
    show = tmp_path / "show"
    ep_dir = show / "vid1"
    ep_dir.mkdir(parents=True)
    from podcodex.core._utils import episode_base
    from podcodex.core.app_config import AppConfig, save_config

    save_config(AppConfig(show_folders=[str(show)]))
    segs = [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "sub only"}]
    save_version(episode_base(show, "vid1"), "transcript", segs, {"step": "transcript"})

    params = {"audio_path": "", "output_dir": str(ep_dir), "source": "transcript"}
    for fmt in ("text", "srt", "vtt"):
        r = client.get(f"/api/export/{fmt}", params=params)
        assert r.status_code == 200, (fmt, r.text)
        assert "vid1" in r.headers["content-disposition"]
    r = client.get(
        "/api/export/zip", params={"audio_path": "", "output_dir": str(ep_dir)}
    )
    assert r.status_code == 200, r.text

    dest = tmp_path / "out.txt"
    r = client.post(
        "/api/export/save",
        json={
            "audio_path": "",
            "output_dir": str(ep_dir),
            "format": "txt",
            "dest": str(dest),
        },
    )
    assert r.status_code == 200, r.text
    assert "sub only" in dest.read_text()


def test_export_takes_the_version_on_screen(client, tmp_path):
    audio, ep_dir = audio_and_output_dir(tmp_path)
    _register_show_folder(ep_dir)
    base = Path(ep_dir) / "ep"
    old = save_version(
        base,
        "transcript",
        [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "old"}],
        {"step": "transcript", "type": "raw"},
    )
    save_version(
        base,
        "transcript",
        [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "new"}],
        {"step": "transcript", "type": "raw"},
    )
    params = {"audio_path": audio, "source": "transcript"}
    assert "new" in client.get("/api/export/text", params=params).text
    r = client.get("/api/export/text", params={**params, "version_id": old})
    assert r.status_code == 200
    assert "old" in r.text and "new" not in r.text
    r = client.get("/api/export/text", params={**params, "version_id": "../x"})
    assert r.status_code == 400


def test_export_filename_outside_latin1(client, tmp_path):
    show = tmp_path / "show"
    show.mkdir()
    audio = show / "Épisode 東京.mp3"
    audio.write_bytes(b"")
    from podcodex.core._utils import AudioPaths
    from podcodex.core.app_config import AppConfig, save_config

    save_config(AppConfig(show_folders=[str(show)]))
    base = AudioPaths.from_audio(str(audio)).base
    save_version(
        base,
        "transcript",
        [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "x"}],
        {"step": "transcript"},
    )
    r = client.get(
        "/api/export/srt", params={"audio_path": str(audio), "source": "transcript"}
    )
    assert r.status_code == 200, r.text
    assert "filename*=UTF-8''" in r.headers["content-disposition"]


def test_export_save_audio_is_confined_to_shows(client, tmp_path):
    """The source file must be inside a show, like every other export branch."""
    audio, ep_dir = audio_and_output_dir(tmp_path)
    _register_show_folder(ep_dir)
    outside = tmp_path / "secret.mp3"
    outside.write_bytes(b"secret")
    not_audio = Path(ep_dir).parent / "notes.txt"
    not_audio.write_text("x")

    def save(src):
        return client.post(
            "/api/export/save",
            json={
                "audio_path": str(src),
                "format": "audio",
                "dest": str(tmp_path / "copy"),
            },
        )

    assert save(outside).status_code == 403
    assert save(not_audio).status_code == 400
    assert not (tmp_path / "copy").exists()


def test_a_translation_exports_with_the_current_speaker_names(
    client, tmp_path, monkeypatch
):
    """The viewer shows the speaker map applied; the export must match it."""
    import podcodex.core.versions as versions_mod

    audio, ep_dir = audio_and_output_dir(tmp_path)
    _register_show_folder(ep_dir)
    base = Path(ep_dir) / "ep"
    segs = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "salut"}]
    vid = save_version(base, "french", segs, {"step": "french", "type": "raw"})
    monkeypatch.setattr(
        versions_mod, "load_latest_speaker_map", lambda _b: {"SPEAKER_00": "Ann"}
    )
    for extra in ({}, {"version_id": vid}):
        r = client.get(
            "/api/export/srt", params={"audio_path": audio, "source": "french", **extra}
        )
        assert r.status_code == 200, r.text
        assert "Ann" in r.text and "SPEAKER_00" not in r.text
