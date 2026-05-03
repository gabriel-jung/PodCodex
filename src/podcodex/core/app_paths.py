"""Canonical paths for user-scoped PodCodex data (config, secrets, app data).

Shared by dev checkouts and packaged .dmg/.msi installs so the backend
finds the same files either way. Config (`config_dir`) lives at
`~/.config/podcodex/` on every platform; *app data* (`data_dir`) follows
OS conventions — Tauri uses these paths for logs and the bundled-sidecar
spawn already writes to them, so the GPU backend installs alongside.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

# OS-level bundle identifier for the desktop app (Tauri code-signing,
# URI handlers, etc.) — kept at reverse-DNS form, defined in
# src-tauri/tauri.conf.json. Distinct from the data-dir folder name
# below, which is just a directory label.
APP_BUNDLE_ID = "com.podcodex.desktop"

# Directory name used under each platform's app-data root. Mirrors
# config_dir()'s use of ``podcodex`` so config and data are symmetric,
# and matches what the bot/CLI/MCP server expect when sharing the index.
# Tauri's Rust shell builds the same path manually before injecting it
# via PODCODEX_DATA_DIR — keep the two in sync.
APP_DATA_DIRNAME = "podcodex"


@lru_cache(maxsize=1)
def config_dir() -> Path:
    """Return the user-scoped config directory, creating it if needed.

    Mode is restricted to 0700 so secrets.env (also 0600) is not exposed
    to other local users via directory listings. Best-effort on Windows
    where POSIX modes have no effect.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) / "podcodex" if xdg else Path.home() / ".config" / "podcodex"
    base.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        try:
            base.chmod(0o700)
        except OSError:
            pass
    return base


@lru_cache(maxsize=1)
def data_dir() -> Path:
    """Return the user-scoped app data directory, creating it if needed.

    Resolution order:
        1. ``$PODCODEX_DATA_DIR`` if set (Tauri shell injects this).
        2. ``$XDG_DATA_HOME/podcodex`` on Linux.
        3. ``~/Library/Application Support/podcodex`` on macOS.
        4. ``%APPDATA%\\podcodex`` on Windows.
        5. ``~/.local/share/podcodex`` as final fallback.

    Used by the GPU backend service for the optional CUDA install
    (`backends/gpu/`) and by Tauri for log output. Matches the path table
    in `deploy/BUILD.md`.
    """
    override = os.environ.get("PODCODEX_DATA_DIR")
    if override:
        base = Path(override)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / APP_DATA_DIRNAME
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = (
            Path(appdata) / APP_DATA_DIRNAME
            if appdata
            else Path.home() / APP_DATA_DIRNAME
        )
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = (
            Path(xdg) / APP_DATA_DIRNAME
            if xdg
            else Path.home() / ".local" / "share" / APP_DATA_DIRNAME
        )
    base.mkdir(parents=True, exist_ok=True)
    return base


def secrets_env_path() -> Path:
    """Return the path to the user's secrets.env file (may not exist)."""
    return config_dir() / "secrets.env"


def running_in_bundle() -> bool:
    """True when this Python process is the PyInstaller-frozen sidecar.

    Used to gate features that only make sense in the shipped app
    (e.g. the GPU backend install/activate flow). In dev mode (uvicorn
    from `.venv`), this returns False and the GPU service degrades to
    read-only status reporting.
    """
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
