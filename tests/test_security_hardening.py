"""Security-hardening regression tests.

Covers the loopback-only desktop hardening pass:
- Host-header guard (anti DNS-rebinding)
- registered-show gate on destructive show routes
- .app bundle refusal in fs/open
- gpu/download no longer accepts a caller-supplied manifest URL
"""

import platform

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from tests.fixtures.api_client import make_client

    return make_client(tmp_path, monkeypatch)


# ── CORS on rejections ───────────────────────────────────────────────────


TAURI_ORIGIN = "tauri://localhost"


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"X-PodCodex-Token": ""}, 401),  # token not resolved yet (first boot)
        ({"X-PodCodex-Token": "wrong"}, 401),
        ({"host": "evil.example.com"}, 421),
    ],
)
def test_guard_rejections_carry_cors_headers(client, headers, expected):
    """A rejected cross-origin request must still be readable by the caller.

    CORSMiddleware has to stay outermost so the guards' 401/421 travel back
    out through it. In the Tauri build the document origin differs from the
    API origin, so a rejection without `Access-Control-Allow-Origin` is
    blocked by the webview: `fetch` raises a network error instead of
    resolving with a status, and the first-boot token-refresh retry in
    `frontend/src/api/client.ts` never runs.
    """
    r = client.get(
        "/api/config",
        headers={"Origin": TAURI_ORIGIN, **headers},
    )

    assert r.status_code == expected
    assert r.headers.get("access-control-allow-origin") == TAURI_ORIGIN


def test_csrf_rejection_carries_cors_headers(client):
    """Same for the CSRF guard's 403 (it sits inside CORS too)."""
    r = client.post(
        "/api/shows/register",
        json={"path": "/tmp"},
        headers={"Origin": TAURI_ORIGIN, "X-PodCodex": ""},
    )

    assert r.status_code == 403
    assert r.headers.get("access-control-allow-origin") == TAURI_ORIGIN


def test_preflight_needs_no_token(client):
    """OPTIONS is answered by CORS before the token guard sees it."""
    r = client.options(
        "/api/config",
        headers={
            "Origin": TAURI_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-podcodex-token",
        },
    )

    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == TAURI_ORIGIN


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


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="the bundle refusal is macOS-only: `open` launches .app dirs there",
)
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


# ── WebSocket host guard ─────────────────────────────────────────────────


def test_ws_allows_loopback(client):
    # TestClient sends "testserver" as ws Host by default; a real local
    # client sends the loopback name, so set it explicitly. Token rides
    # the query string (browser WebSocket can't send custom headers).
    with client.websocket_connect(
        "/api/ws?token=test-token", headers={"host": "127.0.0.1:18811"}
    ):
        pass


def test_ws_rejects_foreign_host(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/api/ws?token=test-token", headers={"host": "evil.example.com"}
        ):
            pass


def test_ws_rejects_missing_token(client):
    from starlette.websockets import WebSocketDisconnect

    # Blank out the client's default token header; no query param either.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/api/ws",
            headers={"host": "127.0.0.1:18811", "X-PodCodex-Token": ""},
        ):
            pass


def test_ws_accepts_header_token(client):
    # The unified guard accepts the header form on websockets too (the
    # client fixture sends it by default).
    with client.websocket_connect("/api/ws", headers={"host": "127.0.0.1:18811"}):
        pass


# ── Loopback auth token ──────────────────────────────────────────────────


def test_token_required_on_api_routes(client):
    r = client.get("/api/config", headers={"X-PodCodex-Token": ""})
    assert r.status_code == 401


def test_token_rejects_wrong_value(client):
    r = client.get("/api/config", headers={"X-PodCodex-Token": "nope"})
    assert r.status_code == 401


def test_token_header_accepted(client):
    assert client.get("/api/config").status_code == 200


def test_token_query_param_accepted(client):
    # <img>/<audio>/download URLs can't send headers; the query param form
    # must work for them.
    r = client.get("/api/config?token=test-token", headers={"X-PodCodex-Token": ""})
    assert r.status_code == 200


def test_token_non_ascii_rejected_cleanly(client):
    # secrets.compare_digest raises TypeError on non-ASCII str; the guard
    # compares bytes so this must be a clean 401, not a 500.
    r = client.get("/api/config?token=caf%C3%A9", headers={"X-PodCodex-Token": ""})
    assert r.status_code == 401


def test_health_exempt_from_token(client):
    # Boot probe runs before the UI has the token.
    r = client.get("/api/health", headers={"X-PodCodex-Token": ""})
    assert r.status_code == 200


def test_options_preflight_exempt_from_token(client):
    # CORS preflights can't carry custom headers; blocking them would break
    # every cross-origin request from the Tauri webview.
    r = client.options(
        "/api/config",
        headers={
            "Origin": "http://tauri.localhost",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-podcodex,x-podcodex-token",
            "X-PodCodex-Token": "",
        },
    )
    assert r.status_code == 200


def test_token_file_created_0600(tmp_path, monkeypatch):
    import os
    import stat

    from podcodex.core import app_paths
    from podcodex.core.api_token import get_or_create_api_token

    monkeypatch.delenv("PODCODEX_API_TOKEN", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    app_paths.config_dir.cache_clear()
    try:
        token = get_or_create_api_token()
        assert token
        f = tmp_path / "podcodex" / "api_token"
        assert f.read_text() == token
        if os.name == "posix":
            assert stat.S_IMODE(f.stat().st_mode) == 0o600
        # Second call reuses, not regenerates.
        assert get_or_create_api_token() == token
    finally:
        app_paths.config_dir.cache_clear()
