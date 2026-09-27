"""Shared FastAPI TestClient factory with an isolated config file.

Patches ``core.app_config.CONFIG_PATH`` (the canonical source read by
``load_config``/``save_config``) and clears the load cache so the patched
path is honored. Patching ``routes.config.CONFIG_PATH`` alone is a no-op;
that name is just a re-export.
"""

import os
from pathlib import Path

from fastapi.testclient import TestClient

from podcodex.api.api_token import TOKEN_HEADER


def client_for(app) -> TestClient:
    """TestClient over an existing app object (module-level route apps).

    Sends the CSRF header and the loopback auth token the guard middleware
    requires. The token comes from ``app.state``: module-level apps resolve
    it at import time via ``get_or_create_api_token()``.
    """
    return TestClient(
        app,
        base_url="http://127.0.0.1:18811",
        headers={"X-PodCodex": "1", TOKEN_HEADER: app.state.api_token},
    )


def make_client(tmp_path, monkeypatch, config=None, *, fresh=False) -> TestClient:
    """TestClient whose config lives under ``tmp_path``.

    ``config``: optional AppConfig persisted before the client is built.
    ``fresh``: build a new app instead of sharing the session's. Needed by a
    test that enters the lifespan (``with make_client(...)``: MCP mount,
    recovery, the task loop) or that patches something ``create_app`` reads.
    """
    from podcodex.core import app_config as app_config_mod
    from podcodex.core.app_config import save_config

    # Fixed token via env so the app never touches the real config dir's
    # api_token file during tests.
    monkeypatch.setenv("PODCODEX_API_TOKEN", "test-token")
    monkeypatch.setattr(app_config_mod, "CONFIG_PATH", tmp_path / "config.json")

    # Isolate the index too, not just the config. Any route that opens the
    # store would otherwise resolve the developer's real index and mutate it
    # (the show-id migration runs on first open). Only set when the caller
    # has not chosen a path itself, so explicit per-test indexes still win.
    if not os.environ.get("PODCODEX_INDEX", "").strip():
        monkeypatch.setenv("PODCODEX_INDEX", str(Path(tmp_path) / "index"))
    from podcodex.rag import index_store as _index_store

    # Defensive: some tests replace get_index_store with a plain stub, which
    # has no cache to clear.
    getattr(_index_store.get_index_store, "cache_clear", lambda: None)()
    if config is not None:
        save_config(config)

    # base_url sets the Host header to a loopback name so the app's
    # host-guard middleware (anti DNS-rebinding) accepts the request.
    if fresh:
        from podcodex.api.app import create_app

        client_cls, app = TestClient, create_app()
    else:
        client_cls, app = _SharedAppClient, _shared_app()
    return client_cls(
        app,
        base_url="http://127.0.0.1:18811",
        headers={"X-PodCodex": "1", "X-PodCodex-Token": "test-token"},
    )


class _SharedAppClient(TestClient):
    """A client over the session's shared app, which must never run its
    lifespan: the MCP session manager starts once per app, so a second
    ``with`` would hang, and startup work would leak into later tests."""

    def __enter__(self):
        raise RuntimeError(
            "entering the lifespan needs an app of its own: "
            "make_client(..., fresh=True)"
        )


def library_client(tmp_path, monkeypatch) -> TestClient:
    """``make_client`` whose default save path is ``tmp_path / "library"``."""
    from podcodex.core.app_config import AppConfig

    return make_client(
        tmp_path,
        monkeypatch,
        config=AppConfig(default_save_path=str(Path(tmp_path) / "library")),
    )


def registered_show(client: TestClient, folder: Path, name: str = "") -> Path:
    """Create *folder* (with a ``show.toml`` when *name* is given) and
    register it through the API, as the app's "add show" does."""
    folder.mkdir(parents=True)
    if name:
        from podcodex.ingest.show import ShowMeta, save_show_meta

        save_show_meta(folder, ShowMeta(name=name))
    r = client.post("/api/shows/register", json={"path": str(folder)})
    assert r.status_code == 200, r.text
    return folder


_APP = None


def _shared_app():
    """One app per session. Building it costs ~40 ms (route registration)
    and config, index and data dir are all read per request through the paths
    ``make_client`` points at tmp. What ``create_app`` does read at build time
    (``running_in_bundle``, whether MCP is installed, the token) is fixed by
    whichever test builds it first, so a test that patches one of those needs
    ``fresh=True``."""
    global _APP
    if _APP is None:
        from podcodex.api.app import create_app

        _APP = create_app()
    return _APP
