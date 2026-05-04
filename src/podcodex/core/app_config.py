"""Persisted app-level configuration (config.json).

Lives in ``core`` so leaf modules (``_ffmpeg``, future TTS / index code)
can read it without importing from ``api/routes`` — the FastAPI route at
``/api/config`` is the *transport* surface, not the source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from podcodex.core.app_paths import config_dir

CONFIG_PATH = config_dir() / "config.json"


class AppConfig(BaseModel):
    show_folders: list[str] = []
    default_save_path: str = ""  # suggested location for new shows
    # Absolute path to a non-PATH ffmpeg binary. Wired through Tauri to
    # the sidecar's PODCODEX_FFMPEG_EXE env, and read directly here so
    # the dev path (no Tauri) works too.
    ffmpeg_exe_override: str = ""


# Hit on every search/list_shows; mtime-keyed so writes auto-invalidate.
_LOAD_CACHE: tuple[float, AppConfig] | None = None


def load_config() -> AppConfig:
    """Load app config from disk, migrating legacy formats if needed."""
    global _LOAD_CACHE
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except FileNotFoundError:
        return AppConfig()
    except OSError:
        mtime = -1.0

    if _LOAD_CACHE is not None and _LOAD_CACHE[0] == mtime:
        return _LOAD_CACHE[1]

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if "podcast_dir" in data and "show_folders" not in data:
            data["show_folders"] = []
            data["default_save_path"] = data.pop("podcast_dir", "")
        cfg = AppConfig(**data)
    except (json.JSONDecodeError, OSError):
        logger.opt(exception=True).warning(
            "Failed to load config from {}, using defaults", CONFIG_PATH
        )
        return AppConfig()

    _LOAD_CACHE = (mtime, cfg)
    return cfg


def save_config(cfg: AppConfig) -> None:
    """Persist app config to disk as JSON (atomic write)."""
    from podcodex.core._utils import atomic_write

    atomic_write(
        CONFIG_PATH,
        lambda p: p.write_text(cfg.model_dump_json(indent=2), encoding="utf-8"),
        suffix=".json",
    )
    global _LOAD_CACHE
    _LOAD_CACHE = None  # invalidate; next load_config() picks up new mtime


def strip_user_path(raw: str) -> str:
    """Trim whitespace and surrounding quotes from a user-supplied path.

    Windows users often paste ``"C:\\path with spaces\\foo.exe"`` literally
    from a batch file or env var. Mirrors the equivalent strip in
    ``src-tauri/src/lib.rs:read_ffmpeg_override_from_config`` so both
    sides resolve the same way.
    """
    return raw.strip().strip('"').strip("'")
