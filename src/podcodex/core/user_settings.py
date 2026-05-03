"""Persisted, user-scoped app settings (single JSON file under data_dir).

Lightweight wrapper for state the user changes from the desktop UI and
expects to survive a restart — currently just the device override (force
CPU vs auto), but kept generic so other small prefs can join later.

Two layered overrides at runtime:

1. ``PODCODEX_DEVICE`` env var — wins when set. Gives dev / CI / shell
   ``make dev-no-tauri-cpu`` a way to override the persisted value
   without touching the file.
2. ``device_override`` key in this file — applied at bootstrap when the
   env var is unset (see ``bootstrap._apply_persisted_device_override``).

Reads / writes are best-effort: a corrupt or unreadable file falls back
to defaults rather than crashing the app — settings are never load-bearing
for correctness, just convenience.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from podcodex.core.app_paths import data_dir

DeviceOverride = Literal["auto", "cpu", "cuda"]
_VALID_DEVICE_OVERRIDES: frozenset[str] = frozenset({"auto", "cpu", "cuda"})

_SETTINGS_FILENAME = "settings.json"


def _path() -> Path:
    return data_dir() / _SETTINGS_FILENAME


def load() -> dict[str, Any]:
    """Return the settings dict, or empty dict on missing / unreadable file."""
    p = _path()
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def save(data: dict[str, Any]) -> None:
    """Atomically write ``data`` to the settings file."""
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def get_device_override() -> DeviceOverride:
    """Return the persisted ``device_override``, defaulting to ``"auto"``."""
    val = load().get("device_override", "auto")
    if isinstance(val, str) and val in _VALID_DEVICE_OVERRIDES:
        return val  # type: ignore[return-value]
    return "auto"


def set_device_override(value: DeviceOverride) -> None:
    """Persist ``device_override``. ``"auto"`` clears the key entirely."""
    if value not in _VALID_DEVICE_OVERRIDES:
        raise ValueError(f"invalid device_override: {value!r}")
    data = load()
    if value == "auto":
        data.pop("device_override", None)
    else:
        data["device_override"] = value
    save(data)
