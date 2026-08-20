"""Tests for POST /api/shows/files/import (standalone-file bucket)."""

import json
from pathlib import Path
from urllib.parse import quote

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


def _import(client, src, name=None, folder=None):
    body = {"file_path": str(src), "name": name}
    if folder is not None:
        body["folder"] = str(folder)
    return client.post("/api/shows/files/import", json=body)


def _make_local_show(client, tmp_path, name: str) -> Path:
    """Create + register a plain local show folder."""
    show = tmp_path / name
    show.mkdir()
    r = client.post("/api/shows/register", json={"path": str(show)})
    assert r.status_code == 200, r.text
    return show


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


# ── Create-local show (picker's "New show…") ──


def test_create_local_show_registers_folder(client, tmp_path):
    r = client.post("/api/shows/create-local", json={"name": "Voice Memos"})
    assert r.status_code == 200, r.text
    folder = Path(r.json()["folder"])
    assert folder == tmp_path / "library" / "Voice Memos"
    assert folder.is_dir()
    shows = client.get("/api/shows/").json()
    assert any(Path(s["path"]) == folder for s in shows)


def test_create_local_show_then_import_into_it(client, tmp_path, audio_file):
    r = client.post("/api/shows/create-local", json={"name": "Voice Memos"})
    folder = r.json()["folder"]
    assert _import(client, audio_file, folder=folder).status_code == 200
    assert "New Recording" in _unified_stems(client, folder)


def test_create_local_show_existing_dir_409(client, tmp_path):
    (tmp_path / "library" / "Taken").mkdir(parents=True)
    r = client.post("/api/shows/create-local", json={"name": "Taken"})
    assert r.status_code == 409


def test_create_local_show_409_suggests_a_free_name(client, tmp_path):
    """Same shape as the import 409, so the picker can offer a name instead
    of making the user guess which ones are free."""
    lib = tmp_path / "library"
    (lib / "Taken").mkdir(parents=True)
    (lib / "Taken-2").mkdir()
    r = client.post("/api/shows/create-local", json={"name": "Taken"})
    assert r.status_code == 409
    assert r.json()["detail"]["suggested"] == "Taken-3"


def test_create_local_show_bad_name_400(client, tmp_path):
    r = client.post("/api/shows/create-local", json={"name": "../evil"})
    assert r.status_code == 400


# ── Target-folder imports ──


def test_import_into_chosen_local_show(client, tmp_path, audio_file):
    show = _make_local_show(client, tmp_path, "Voice Memos")
    r = _import(client, audio_file, folder=show)
    assert r.status_code == 200
    assert r.json() == {"folder": str(show), "stem": "New Recording"}
    assert (show / "New Recording.m4a").read_bytes() == b"fake audio bytes"
    # No Files bucket materialized as a side effect.
    assert not (tmp_path / "library" / "Files").exists()


def test_import_into_unregistered_folder_403(client, tmp_path, audio_file):
    stray = tmp_path / "stray"
    stray.mkdir()
    assert _import(client, audio_file, folder=stray).status_code == 403


def test_import_into_feed_backed_show_400(client, tmp_path, audio_file):
    show = _make_local_show(client, tmp_path, "SomePodcast")
    (show / ".feed_cache.json").write_text("{}")
    r = _import(client, audio_file, folder=show)
    assert r.status_code == 400


def _set_meta(client, folder, **fields):
    meta = client.get(f"/api/shows/{quote(str(folder), safe='')}/meta").json()
    meta.update(fields)
    r = client.put(f"/api/shows/{quote(str(folder), safe='')}/meta", json=meta)
    assert r.status_code == 200, r.text


def test_import_into_rss_url_show_400(client, tmp_path, audio_file):
    """A show with a feed URL but no cache yet is still feed-backed: its
    episodes come from the feed, so loose audio must not land there."""
    show = _make_local_show(client, tmp_path, "Podcast")
    _set_meta(client, show, rss_url="https://feed.example/x.xml")
    assert not (show / ".feed_cache.json").exists()
    r = _import(client, audio_file, folder=show)
    assert r.status_code == 400


def test_import_into_youtube_show_400(client, tmp_path, audio_file):
    show = _make_local_show(client, tmp_path, "Channel")
    _set_meta(client, show, youtube_url="https://youtube.com/@chan")
    assert _import(client, audio_file, folder=show).status_code == 400


def test_accepts_imports_flag_matches_the_import_gate(client, tmp_path, audio_file):
    """The flag the pickers gate on must agree with what the endpoint does,
    or the UI offers destinations the server rejects."""
    local = _make_local_show(client, tmp_path, "Voice Memos")
    feed = _make_local_show(client, tmp_path, "Podcast")
    _set_meta(client, feed, rss_url="https://feed.example/x.xml")

    summaries = {s["path"]: s for s in client.get("/api/shows/").json()}
    assert summaries[str(local)]["accepts_imports"] is True
    assert summaries[str(feed)]["accepts_imports"] is False

    for folder, expected in ((local, True), (feed, False)):
        meta = client.get(f"/api/shows/{quote(str(folder), safe='')}/meta").json()
        assert meta["accepts_imports"] is expected
        accepted = _import(client, audio_file, folder=folder).status_code == 200
        assert accepted is expected


def test_import_into_chosen_show_collision_409(client, tmp_path, audio_file):
    show = _make_local_show(client, tmp_path, "Voice Memos")
    assert _import(client, audio_file, folder=show).status_code == 200
    r = _import(client, audio_file, folder=show)
    assert r.status_code == 409
    assert r.json()["detail"]["suggested"] == "New Recording-2"


def test_import_into_chosen_show_visible_in_unified(client, tmp_path, audio_file):
    show = _make_local_show(client, tmp_path, "Voice Memos")
    # Populate the DB first so the import exercises the heal path too.
    assert _unified_stems(client, str(show)) == set()
    assert _import(client, audio_file, folder=show).status_code == 200
    assert "New Recording" in _unified_stems(client, str(show))


def _unified_stems(client, folder: str) -> set:
    r = client.get(f"/api/shows/{quote(folder, safe='')}/unified")
    assert r.status_code == 200, r.text
    return {e.get("stem") for e in r.json()}


def test_first_import_visible_in_unified(client, tmp_path, audio_file):
    r = _import(client, audio_file)
    assert r.status_code == 200
    assert "New Recording" in _unified_stems(client, r.json()["folder"])


def test_import_after_db_populated_visible_in_unified(client, tmp_path, audio_file):
    """Regression: pipeline.db only bootstraps from a scan while empty, so a
    file imported after the first /unified fetch never got a DB row and the
    episode list never showed it."""
    r1 = _import(client, audio_file)
    assert r1.status_code == 200
    folder = r1.json()["folder"]
    # First fetch populates pipeline.db from the scan.
    assert "New Recording" in _unified_stems(client, folder)

    other = tmp_path / "other.mp3"
    other.write_bytes(b"x")
    assert _import(client, other).status_code == 200
    assert "other" in _unified_stems(client, folder)


def test_hand_copied_audio_visible_in_unified(client, tmp_path, audio_file):
    """A file copied into the show folder out of band (no API call, so no
    cache invalidation) must still show up."""
    r = _import(client, audio_file)
    assert r.status_code == 200
    folder = r.json()["folder"]
    assert "New Recording" in _unified_stems(client, folder)

    (Path(folder) / "manual.mp3").write_bytes(b"y")
    assert "manual" in _unified_stems(client, folder)


def test_meta_only_dir_visible_in_unified(client, tmp_path, audio_file):
    """An episode directory restored out of band (bundle import writes dirs,
    not root audio) must show up once it carries episode metadata."""
    r = _import(client, audio_file)
    assert r.status_code == 200
    folder = r.json()["folder"]
    assert "New Recording" in _unified_stems(client, folder)

    ep_dir = Path(folder) / "restored"
    ep_dir.mkdir()
    (ep_dir / ".episode_meta.json").write_text(
        json.dumps({"title": "Restored", "guid": "g-restored"}), encoding="utf-8"
    )
    assert "restored" in _unified_stems(client, folder)


def test_stray_file_at_bucket_path_shifts_bucket(client, tmp_path, audio_file):
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "Files").write_text("stray")
    r = _import(client, audio_file)
    assert r.status_code == 200
    assert Path(r.json()["folder"]) == lib / "Files-2"
