"""Shared FastAPI TestClient factory with an isolated config file.

Patches ``core.app_config.CONFIG_PATH`` (the canonical source read by
``load_config``/``save_config``) and clears the load cache so the patched
path is honored. Patching ``routes.config.CONFIG_PATH`` alone is a no-op;
that name is just a re-export.
"""

from fastapi.testclient import TestClient


def make_client(tmp_path, monkeypatch, config=None) -> TestClient:
    """TestClient over a fresh app whose config lives under ``tmp_path``.

    ``config``: optional AppConfig persisted before the app is created.
    """
    from podcodex.api.app import create_app
    from podcodex.core import app_config as app_config_mod
    from podcodex.core.app_config import save_config

    monkeypatch.setattr(app_config_mod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(app_config_mod, "_LOAD_CACHE", None)
    if config is not None:
        save_config(config)

    # base_url sets the Host header to a loopback name so the app's
    # host-guard middleware (anti DNS-rebinding) accepts the request.
    return TestClient(
        create_app(),
        base_url="http://127.0.0.1:18811",
        headers={"X-PodCodex": "1"},
    )
