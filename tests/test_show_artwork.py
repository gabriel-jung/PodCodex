"""Tests for POST /api/shows/artwork (local cover upload)."""

from pathlib import Path

import pytest

from tests.fixtures.api_client import make_client

# Smallest valid PNG (1x1, from the PNG spec examples).
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd40000000049454e44ae426082"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from podcodex.core.app_config import AppConfig

    return make_client(
        tmp_path,
        monkeypatch,
        config=AppConfig(default_save_path=str(tmp_path / "library")),
    )


@pytest.fixture
def show(client, tmp_path) -> Path:
    folder = tmp_path / "MyShow"
    folder.mkdir()
    r = client.post("/api/shows/register", json={"path": str(folder)})
    assert r.status_code == 200
    return folder


def _upload(client, folder, filename="cover.png", data=PNG_BYTES):
    return client.post(
        "/api/shows/artwork",
        params={"show_folder": str(folder)},
        files={"file": (filename, data, "image/png")},
    )


def test_upload_writes_artwork_and_marks_meta(client, show):
    r = _upload(client, show)
    assert r.status_code == 200, r.text
    assert (show / "artwork.png").read_bytes() == PNG_BYTES
    meta = client.get(f"/api/shows/{str(show)}/meta").json()
    assert meta["artwork_url"] == "local"


def test_uploaded_artwork_is_served(client, show):
    assert _upload(client, show).status_code == 200
    r = client.get("/api/shows/artwork", params={"show_folder": str(show)})
    assert r.status_code == 200
    assert r.content == PNG_BYTES


def test_upload_replaces_previous_extension(client, show):
    (show / "artwork.jpg").write_bytes(b"old")
    assert _upload(client, show).status_code == 200
    assert not (show / "artwork.jpg").exists()
    assert (show / "artwork.png").exists()


def test_upload_rejects_non_image_extension(client, show):
    r = _upload(client, show, filename="cover.txt")
    assert r.status_code == 400
    assert not (show / "artwork.txt").exists()


def test_upload_rejects_oversized_file(client, show):
    big = b"x" * (5 * 1024 * 1024 + 1)
    r = _upload(client, show, data=big)
    assert r.status_code == 413
    assert not (show / "artwork.png").exists()


def test_upload_into_unregistered_folder_403(client, tmp_path):
    stray = tmp_path / "stray"
    stray.mkdir()
    assert _upload(client, stray).status_code == 403


def _set_artwork_url(client, show, url: str) -> None:
    meta = client.get(f"/api/shows/{str(show)}/meta").json()
    meta["artwork_url"] = url
    r = client.put(f"/api/shows/{str(show)}/meta", json=meta)
    assert r.status_code == 200, r.text


def test_switching_from_upload_to_url_redownloads(client, show, monkeypatch):
    """Upload clears the URL hash, so a cached file with no hash file must
    re-download instead of serving the stale uploaded image forever."""
    from podcodex.api.routes import shows as shows_routes

    assert _upload(client, show).status_code == 200

    downloaded = b"fetched-from-url"

    def fake_download(url: str, show_path: Path):
        for old in show_path.glob("artwork.*"):
            old.unlink()
        dest = show_path / "artwork.jpg"
        dest.write_bytes(downloaded)
        return dest

    monkeypatch.setattr(shows_routes, "_download_artwork", fake_download)
    _set_artwork_url(client, show, "https://example.com/cover.jpg")

    r = client.get("/api/shows/artwork", params={"show_folder": str(show)})
    assert r.status_code == 200
    assert r.content == downloaded


def test_youtube_refresh_keeps_uploaded_cover(client, show, monkeypatch):
    """An uploaded cover is the user's explicit choice; the YouTube artwork
    upgrade must not overwrite the marker with the channel thumbnail."""
    from podcodex.ingest import youtube as youtube_ingest
    from podcodex.ingest.rss import RSSEpisode
    from podcodex.ingest.show import load_show_meta

    assert _upload(client, show).status_code == 200
    _set_artwork_url(client, show, "local")
    meta = load_show_meta(show)
    assert meta is not None
    meta.youtube_url = "https://youtube.com/@chan"
    from podcodex.ingest.show import save_show_meta

    save_show_meta(show, meta)

    def fake_fetch(url):
        ep = RSSEpisode(guid="v1", title="Video 1", audio_url="", pub_date="2026-01-01")
        return [ep], {"artwork_url": "https://yt3.googleusercontent.com/new.jpg"}

    monkeypatch.setattr(youtube_ingest, "fetch_youtube", fake_fetch)

    r = client.post(f"/api/shows/{str(show)}/youtube/fetch")
    assert r.status_code == 200, r.text

    refreshed = load_show_meta(show)
    assert refreshed is not None
    assert refreshed.artwork_url == "local"


def test_local_marker_without_file_404(client, show):
    meta = client.get(f"/api/shows/{str(show)}/meta").json()
    meta["artwork_url"] = "local"
    r = client.put(f"/api/shows/{str(show)}/meta", json=meta)
    assert r.status_code == 200, r.text
    r = client.get("/api/shows/artwork", params={"show_folder": str(show)})
    assert r.status_code == 404
