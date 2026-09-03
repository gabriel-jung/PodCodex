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


# ── Version routes: lang / version_id stay one path component ────────────


def _episode(tmp_path):
    """Stub show folder + episode dir, returning (audio_path, ep_dir)."""
    show = tmp_path / "show"
    show.mkdir()
    audio = show / "ep.mp3"
    audio.touch()
    (show / "ep").mkdir()
    return str(audio), str(show / "ep")


def test_translate_version_lang_traversal_rejected(client, tmp_path):
    """`lang` becomes a directory name, so traversal must 400, not read files.

    `normalize_lang` only lowercases and de-spaces, so a lang of
    "../../../../.config/podcodex" used to resolve version_path onto any
    JSON file on disk: the GET returned it and the DELETE unlinked it.
    """
    audio, _ = _episode(tmp_path)
    secret = tmp_path / "api_keys.json"
    secret.write_text('{"openai": "sk-secret"}', encoding="utf-8")

    params = {"audio_path": audio, "lang": "../../../.."}
    r = client.get("/api/translate/versions/api_keys", params=params)
    assert r.status_code == 400
    r = client.delete("/api/translate/versions/api_keys", params=params)
    assert r.status_code == 400
    assert secret.exists()

    r = client.get("/api/translate/versions", params=params)
    assert r.status_code == 400


def test_version_path_rejects_traversal_components():
    from podcodex.core.versions import version_path

    with pytest.raises(ValueError):
        version_path(__import__("pathlib").Path("/tmp/x/ep"), "../..", "api_keys")
    with pytest.raises(ValueError):
        version_path(__import__("pathlib").Path("/tmp/x/ep"), "english", "../../x")


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
    # follows ".." segments, so an unslugged label unlinked this file.
    victim = show / "x_00.wav"
    victim.write_bytes(b"keep")

    synth.extract_selected_samples(
        audio,
        [{"speaker": "../../x", "start": 0.0, "end": 1.0, "text": "hi"}],
    )
    samples_dir = show / "ep" / "voice_samples"
    assert written and all(p.parent == samples_dir for p in written)
    assert victim.exists()


def test_upload_sample_rejects_path_speaker(client, tmp_path):
    audio, _ = _episode(tmp_path)
    r = client.post(
        "/api/synthesize/upload-sample",
        data={"audio_path": audio, "speaker": "../../evil"},
        files={"file": ("a.wav", b"RIFF", "audio/wav")},
    )
    assert r.status_code == 400


# ── Artwork downloads are http(s)-only ───────────────────────────────────


def test_download_artwork_refuses_file_scheme(tmp_path):
    """A feed-controlled artwork URL must not read a local file."""
    from podcodex.api.routes.shows import _download_artwork

    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"\xff\xd8\xffnot-yours")
    show = tmp_path / "show"
    show.mkdir()

    assert _download_artwork(secret.as_uri(), show) is None
    assert not list(show.iterdir())


# ── GPU sidecar archive is verified, never on a best-effort basis ────────


def test_gpu_install_refuses_manifest_without_server_hash(tmp_path, monkeypatch):
    """server-core.tar.gz becomes the executed sidecar, so its hash is required.

    The digest used to come from an optional ``<archive>.sha256`` sidecar
    fetch, so a 404 or a network blip downgraded the integrity check on the
    one archive that carries code to a log warning.
    """
    import json

    from podcodex.api import gpu_backend

    monkeypatch.setattr(gpu_backend, "_ensure_bundle_mode", lambda: None)
    monkeypatch.setattr(gpu_backend, "_ensure_platform_supported", lambda: None)
    monkeypatch.setattr(gpu_backend, "gpu_install_dir", lambda: tmp_path / "gpu")
    monkeypatch.setattr(
        gpu_backend,
        "_fetch_text",
        lambda url, **kw: json.dumps(
            {
                "version": "cu128-v1",
                "archive": "cuda-libs-cu128-v1.tar.gz",
                "sha256": "a" * 64,
            }
        ),
    )

    with pytest.raises(RuntimeError, match="server_sha256"):
        gpu_backend.download_and_install(lambda *a: None, "https://x/cuda-libs.json")


def test_gpu_packager_publishes_the_server_hash():
    """The packager must emit what the installer now requires."""
    from pathlib import Path

    src = Path("packaging/package_gpu.py").read_text(encoding="utf-8")
    assert '"server_sha256": core_sha,' in src


# ── Reserved index names in an imported manifest ─────────────────


def test_reserved_collection_names_are_refused_by_the_member_filter():
    """`bad_path_component` passes these, so the allowlist has to refuse them itself."""
    from podcodex.bundle.import_show import _collection_member
    from podcodex.core._utils import bad_path_component
    from podcodex.rag.index_store import reserved_index_names

    for name in reserved_index_names():
        # Not a traversal, so the path check alone would let it through.
        assert not bad_path_component(name)
        for member in (f"{name}.lance", f"{name}.json", f"{name}.txn"):
            assert not _collection_member(member, {name}), member
    # A real collection is still admitted.
    assert _collection_member("myshow.lance", {"myshow"})
    assert _collection_member("myshow.txn", {"myshow"})


def test_import_refuses_a_manifest_declaring_a_reserved_collection(
    tmp_path, monkeypatch
):
    """A crafted bundle must not overwrite the ownership marker or the sidecar tables."""
    import pytest

    from podcodex.bundle.import_show import _plan_collections
    from podcodex.bundle.manifest import ArchiveCorruptError, CollectionEntry, Manifest
    from podcodex.bundle.manifest import Mode, ShowEntry

    class _Store:
        def list_collections(self):
            return []

    for reserved in ("index_origin", "_show_passwords", "_collections"):
        manifest = Manifest(
            mode=Mode.INDEX_ONLY,
            podcodex_version="0.0.0",
            exported_at="2026-01-01T00:00:00Z",
            shows=[
                ShowEntry(
                    name="Evil",
                    folder="evil",
                    collections=[
                        CollectionEntry(
                            name=reserved,
                            model="bge-m3",
                            chunker="semantic",
                            dim=8,
                            rows=0,
                        )
                    ],
                )
            ],
        )
        with pytest.raises(ArchiveCorruptError):
            _plan_collections(manifest, None, _Store(), {})
