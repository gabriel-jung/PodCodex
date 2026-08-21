"""Shared FastAPI TestClient factory with an isolated config file.

Patches ``core.app_config.CONFIG_PATH`` (the canonical source read by
``load_config``/``save_config``) and clears the load cache so the patched
path is honored. Patching ``routes.config.CONFIG_PATH`` alone is a no-op;
that name is just a re-export.
"""

import os
from pathlib import Path

from fastapi.testclient import TestClient

from podcodex.core.api_token import TOKEN_HEADER


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


def make_client(tmp_path, monkeypatch, config=None) -> TestClient:
    """TestClient over a fresh app whose config lives under ``tmp_path``.

    ``config``: optional AppConfig persisted before the app is created.
    """
    from podcodex.api.app import create_app
    from podcodex.core import app_config as app_config_mod
    from podcodex.core.app_config import save_config

    # Fixed token via env so the app never touches the real config dir's
    # api_token file during tests.
    monkeypatch.setenv("PODCODEX_API_TOKEN", "test-token")
    monkeypatch.setattr(app_config_mod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(app_config_mod, "_LOAD_CACHE", None)

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
    return TestClient(
        create_app(),
        base_url="http://127.0.0.1:18811",
        headers={"X-PodCodex": "1", "X-PodCodex-Token": "test-token"},
    )
