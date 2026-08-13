"""Tests for POST /api/shows/files/import (standalone-file bucket)."""

from pathlib import Path

import pytest

from tests.fixtures.api_client import make_client


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with isolated config whose default_save_path is tmp."""
    from podcodex.core.app_config import AppConfig

    return make_client(
        tmp_path,
        monkeypatch,
        config=AppConfig(default_save_path=str(tmp_path / "library")),
    )


@pytest.fixture
def audio_file(tmp_path):
    src = tmp_path / "New Recording.m4a"
    src.write_bytes(b"fake audio bytes")
    return src


def _import(client, src, name=None):
    return client.post(
        "/api/shows/files/import",
        json={"file_path": str(src), "name": name},
    )


def test_first_import_creates_and_registers_bucket(client, tmp_path, audio_file):
    r = _import(client, audio_file)
    assert r.status_code == 200
    body = r.json()
    bucket = tmp_path / "library" / "Files"
    assert body == {"folder": str(bucket), "stem": "New Recording"}
    assert (bucket / "New Recording.m4a").read_bytes() == b"fake audio bytes"
    # original untouched
    assert audio_file.exists()
    # registered as a show
    shows = client.get("/api/shows/").json()
    assert any(Path(s["path"]) == bucket for s in shows)


def test_second_import_reuses_bucket(client, tmp_path, audio_file):
    other = tmp_path / "other.mp3"
    other.write_bytes(b"x")
    assert _import(client, audio_file).status_code == 200
    assert _import(client, other).status_code == 200
    shows = client.get("/api/shows/").json()
    bucket = tmp_path / "library" / "Files"
    assert sum(1 for s in shows if Path(s["path"]) == bucket) == 1


def test_collision_returns_409_with_suggestion(client, audio_file):
    assert _import(client, audio_file).status_code == 200
    r = _import(client, audio_file)
    assert r.status_code == 409
    assert r.json()["detail"]["suggested"] == "New Recording-2"


def test_name_override_lands_under_chosen_stem(client, tmp_path, audio_file):
    r = _import(client, audio_file, name="Interview Alice")
    assert r.status_code == 200
    assert r.json()["stem"] == "Interview Alice"
    assert (tmp_path / "library" / "Files" / "Interview Alice.m4a").exists()


def test_missing_source_404(client, tmp_path):
    r = _import(client, tmp_path / "nope.mp3")
    assert r.status_code == 404


def test_non_audio_extension_400(client, tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("hi")
    assert _import(client, src).status_code == 400


def test_invalid_name_400(client, audio_file):
    assert _import(client, audio_file, name="../evil").status_code == 400


def test_cross_extension_collision_409(client, tmp_path, audio_file):
    """The folder scanner keys episodes by stem alone, so a same-stem file
    with a different extension must 409 too (and the suggestion must skip it)."""
    assert _import(client, audio_file).status_code == 200
    other = tmp_path / "New Recording.mp3"
    other.write_bytes(b"y")
    r = _import(client, other)
    assert r.status_code == 409
    assert r.json()["detail"]["suggested"] == "New Recording-2"


def test_feed_backed_registered_files_show_shifts_bucket(client, tmp_path, audio_file):
    """A feed-backed show at <library>/Files is never hijacked, even when it
    is already registered."""
    from podcodex.core.app_config import load_config, save_config

    bucket = tmp_path / "library" / "Files"
    bucket.mkdir(parents=True)
    (bucket / ".feed_cache.json").write_text("{}")
    cfg = load_config()
    cfg.show_folders = [str(bucket)]
    save_config(cfg)

    r = _import(client, audio_file)
    assert r.status_code == 200
    assert Path(r.json()["folder"]) == tmp_path / "library" / "Files-2"
    assert not (bucket / "New Recording.m4a").exists()


def test_stray_file_at_bucket_path_shifts_bucket(client, tmp_path, audio_file):
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "Files").write_text("stray")
    r = _import(client, audio_file)
    assert r.status_code == 200
    assert Path(r.json()["folder"]) == lib / "Files-2"
