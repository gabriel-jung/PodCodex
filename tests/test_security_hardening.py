"""Security-hardening regression tests.

Covers the loopback-only desktop hardening pass:
- Host-header guard (anti DNS-rebinding)
- registered-show gate on destructive show routes
- .app bundle refusal in fs/open
- gpu/download no longer accepts a caller-supplied manifest URL
"""

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from tests.fixtures.api_client import make_client

    return make_client(tmp_path, monkeypatch)


# ── Host-header guard ────────────────────────────────────────────────────


def test_host_guard_allows_loopback(client):
    """The fixture's base_url sends a loopback Host, so requests pass."""
    assert client.get("/api/health").status_code == 200


def test_host_guard_rejects_foreign_host(client):
    """A rebound request carries the attacker's hostname, not a loopback name."""
    r = client.get("/api/health", headers={"host": "evil.example.com"})
    assert r.status_code == 421


def test_host_guard_rejects_loopback_wrong_port(client):
    """Host must match the bound port, not just the loopback name."""
    r = client.get("/api/health", headers={"host": "127.0.0.1:9999"})
    assert r.status_code == 421


def test_host_guard_rejects_empty_host(client):
    """A missing/empty Host is not a loopback name, so it is rejected too."""
    r = client.get("/api/health", headers={"host": ""})
    assert r.status_code == 421


# ── Registered-show gate on destructive routes ───────────────────────────


def test_delete_unregistered_show_forbidden(client, tmp_path):
    """delete_files runs rmtree; an unregistered directory must be refused."""
    victim = tmp_path / "not_a_show"
    victim.mkdir()
    (victim / "keep.txt").write_text("important")

    r = client.post(f"/api/shows/{victim}/delete", json={"delete_files": True})
    assert r.status_code == 403
    assert victim.exists()  # nothing deleted


def test_delete_registered_show_allowed(client, tmp_path):
    """A registered show still deletes normally."""
    show = tmp_path / "myshow"
    show.mkdir()
    client.post("/api/shows/register", json={"path": str(show)})

    r = client.post(f"/api/shows/{show}/delete", json={"delete_files": True})
    assert r.status_code == 200
    assert not show.exists()


def test_update_meta_unregistered_show_forbidden(client, tmp_path):
    """update_show_meta writes show.toml; an unregistered dir must be refused."""
    victim = tmp_path / "not_a_show"
    victim.mkdir()

    meta = {
        "name": "Injected",
        "rss_url": "",
        "youtube_url": "",
        "language": "en",
        "speakers": [],
        "artwork_url": "",
        "broadcast_number_pattern": "",
        "pipeline": {},
    }
    r = client.put(f"/api/shows/{victim}/meta", json=meta)
    assert r.status_code == 403
    assert not (victim / "show.toml").exists()


def test_move_unregistered_show_forbidden(client, tmp_path):
    """move runs shutil.move/rmtree on the source; refuse an unregistered dir."""
    victim = tmp_path / "not_a_show"
    victim.mkdir()
    dest = tmp_path / "dest"

    r = client.post(f"/api/shows/{victim}/move", json={"new_path": str(dest)})
    assert r.status_code == 403
    assert victim.exists()


def test_rss_fetch_unregistered_show_forbidden(client, tmp_path):
    """rss_fetch writes .feed_cache.json; refuse an unregistered dir."""
    victim = tmp_path / "not_a_show"
    victim.mkdir()
    r = client.post(f"/api/shows/{victim}/rss/fetch", params={"rss_url": "http://x/f"})
    assert r.status_code == 403


def test_rss_download_unregistered_show_forbidden(client, tmp_path):
    """rss_download writes episode audio; refuse an unregistered dir."""
    victim = tmp_path / "not_a_show"
    victim.mkdir()
    r = client.post(f"/api/shows/{victim}/rss/download")
    assert r.status_code == 403


def test_youtube_fetch_unregistered_show_forbidden(client, tmp_path):
    """youtube_fetch writes into the folder; refuse an unregistered dir."""
    victim = tmp_path / "not_a_show"
    victim.mkdir()
    r = client.post(f"/api/shows/{victim}/youtube/fetch")
    assert r.status_code == 403


# ── fs/open bundle refusal ───────────────────────────────────────────────


def test_fs_open_rejects_app_bundle(client, tmp_path):
    """`open <bundle>` would launch the app; the route must refuse .app dirs."""
    bundle = tmp_path / "Evil.app"
    bundle.mkdir()

    r = client.post("/api/fs/open", params={"path": str(bundle)})
    assert r.status_code == 200
    assert "bundle" in (r.json().get("error") or "").lower()


# ── gpu/download no longer honors a caller manifest URL ───────────────────


def test_gpu_download_takes_no_body(client):
    """The route must not require (or accept) a manifest URL in the body.

    In dev mode it short-circuits with 400 before any download, but the key
    assertion is that a missing body is NOT a 422 validation error, proving
    the caller-supplied manifest URL is gone.
    """
    r = client.post("/api/gpu/download")
    assert r.status_code == 400  # dev-mode guard, not a body-validation 422


def test_gpu_download_ignores_stray_manifest_body(client):
    """A body attempting to inject a manifest URL is ignored, not honored."""
    r = client.post(
        "/api/gpu/download", json={"manifest_url": "http://attacker.example/m.json"}
    )
    assert r.status_code == 400  # still just the dev-mode guard
