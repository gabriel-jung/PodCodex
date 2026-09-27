"""The loopback guard every request passes: Host check (anti DNS-rebinding),
the API token (header or query param, persisted 0600), and CORS headers on
its rejections so the Tauri webview can read them."""

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
    """OPTIONS is answered by CORS before the token guard sees it: preflights
    can't carry custom headers, so blocking them would break every
    cross-origin request from the Tauri webview."""
    r = client.options(
        "/api/config",
        headers={
            "Origin": TAURI_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-podcodex-token",
            "X-PodCodex-Token": "",
        },
    )

    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == TAURI_ORIGIN


# ── Host-header guard ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"X-PodCodex-Token": ""}, 401),
        ({"X-PodCodex-Token": "wrong"}, 401),
        ({"host": "evil.example.com"}, 421),
    ],
)
def test_guard_rejects_without_an_origin(client, headers, expected):
    """curl, a local process or a rebound request sends no Origin; the token
    and Host checks apply to them exactly as to the webview."""
    assert client.get("/api/config", headers=headers).status_code == expected


def test_host_guard_rejects_loopback_wrong_port(client):
    """Host must match the bound port, not just the loopback name."""
    r = client.get("/api/health", headers={"host": "127.0.0.1:9999"})
    assert r.status_code == 421


def test_host_guard_rejects_empty_host(client):
    """A missing/empty Host is not a loopback name, so it is rejected too."""
    r = client.get("/api/health", headers={"host": ""})
    assert r.status_code == 421


# ── WebSocket host guard ─────────────────────────────────────────────────


def test_ws_allows_loopback(client):
    # TestClient sends "testserver" as ws Host by default; a real local
    # client sends the loopback name, so set it explicitly. Token rides
    # the query string only (browser WebSocket can't send custom headers),
    # so the client's default token header is blanked.
    with client.websocket_connect(
        "/api/ws?token=test-token",
        headers={"host": "127.0.0.1:18811", "X-PodCodex-Token": ""},
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


def test_token_file_created_0600(tmp_path, monkeypatch):
    import os
    import stat

    from podcodex.core import app_paths
    from podcodex.api.api_token import get_or_create_api_token

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
