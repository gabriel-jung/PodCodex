"""Canonical paths for user-scoped PodCodex data (config, secrets).

Shared by dev checkouts and packaged .app/.deb/.exe installs so the
backend finds the same files either way. All platforms currently use
`~/.config/podcodex/` for continuity with existing installs; the
packaging milestone may move macOS/Windows to native dirs with a
one-shot migration.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def config_dir() -> Path:
    """Return the user-scoped config directory, creating it if needed."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) / "podcodex" if xdg else Path.home() / ".config" / "podcodex"
    base.mkdir(parents=True, exist_ok=True)
    return base


def secrets_env_path() -> Path:
    """Return the path to the user's secrets.env file (may not exist)."""
    return config_dir() / "secrets.env"
