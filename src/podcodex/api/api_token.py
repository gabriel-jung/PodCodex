"""Loopback API auth token.

The API binds to 127.0.0.1, but loopback alone does not authenticate the
caller: any local process or OS user can reach the port. A shared secret
between the UI and the server closes that gap. The token is persisted at
``config_dir()/api_token`` (created 0600 on first use) so the Tauri shell
and the Vite dev server can hand it to the frontend; the Rust side mirrors
the path in ``src-tauri/src/lib.rs``.

``PODCODEX_API_TOKEN`` overrides the file (tests, CI); when set, the file
is neither read nor written.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from podcodex.core._utils import atomic_write
from podcodex.core.app_paths import config_dir

TOKEN_FILENAME = "api_token"
TOKEN_HEADER = "X-PodCodex-Token"
TOKEN_QUERY_PARAM = "token"


def get_or_create_api_token() -> str:
    """Return the loopback auth token, creating and persisting one if needed."""
    env = os.environ.get("PODCODEX_API_TOKEN")
    if env:
        return env
    path = config_dir() / TOKEN_FILENAME
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    token = secrets.token_urlsafe(32)

    def _writer(p: Path) -> None:
        p.write_text(token, encoding="utf-8")
        os.chmod(p, 0o600)

    atomic_write(path, _writer)
    return token
