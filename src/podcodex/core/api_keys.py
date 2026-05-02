"""Named API key pool — generic credential storage.

Replaces the old single-slot `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
`MISTRAL_API_KEY` model. The pool is a flat list of `(name, value,
suggested_provider?)` entries: storage decoupled from provider so the
same key can be paired with any provider profile at usage sites.

Persisted at `<config_dir>/api_keys.json` with mode 0600.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from podcodex.core._utils import atomic_write
from podcodex.core.app_paths import config_dir

_KNOWN_PROVIDER_PREFIXES: dict[str, str] = {
    "OPENAI": "openai",
    "ANTHROPIC": "anthropic",
    "MISTRAL": "mistral",
}

_API_KEY_VAR_RE = re.compile(r"^([A-Z0-9_]+)_API_KEY$")


def api_keys_path() -> Path:
    """Filesystem path of the pool JSON."""
    return config_dir() / "api_keys.json"


def mask_secret(value: str) -> str:
    """Render a short preview of a secret: first 4 chars + ``****``."""
    if len(value) <= 4:
        return "****"
    return value[:4] + "****"


class APIKey(BaseModel):
    """A single entry in the named key pool."""

    name: str = Field(..., min_length=1, max_length=80)
    value: str
    suggested_provider: str | None = None
    source: Literal["ui", "env"] = "ui"


class APIKeyPublic(BaseModel):
    """Pool entry as returned to the UI — value masked."""

    name: str
    masked: str
    suggested_provider: str | None = None
    source: Literal["ui", "env"]


class APIKeysFile(BaseModel):
    keys: list[APIKey] = []


def to_public(key: APIKey) -> APIKeyPublic:
    return APIKeyPublic(
        name=key.name,
        masked=mask_secret(key.value),
        suggested_provider=key.suggested_provider,
        source=key.source,
    )


# mtime-keyed cache so high-volume callers (batch resolver, per-route
# CRUD) avoid re-parsing the file on every hit.
_KEYS_CACHE: tuple[float, APIKeysFile] | None = None


def load_keys() -> APIKeysFile:
    """Load the pool from disk; empty if missing or unreadable."""
    global _KEYS_CACHE
    path = api_keys_path()
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return APIKeysFile()
    except OSError:
        mtime = -1.0

    if _KEYS_CACHE is not None and _KEYS_CACHE[0] == mtime:
        return _KEYS_CACHE[1].model_copy(deep=True)

    try:
        file = APIKeysFile(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return APIKeysFile()

    _KEYS_CACHE = (mtime, file)
    return file.model_copy(deep=True)


def save_keys(file: APIKeysFile) -> None:
    """Persist atomically with mode 0600."""
    global _KEYS_CACHE
    path = api_keys_path()

    def _writer(p: Path) -> None:
        p.write_text(file.model_dump_json(indent=2), encoding="utf-8")
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass

    atomic_write(path, _writer, suffix=".json")
    _KEYS_CACHE = None


def find_key(file: APIKeysFile, name: str) -> APIKey | None:
    for k in file.keys:
        if k.name == name:
            return k
    return None


def parse_env_var_name(var: str) -> tuple[str, str | None] | None:
    """Parse an env-var name like `OPENAI_WORK_API_KEY` into (name, suggested).

    Returns ``None`` if `var` doesn't match the `*_API_KEY` shape.
    Returns ``(name, None)`` when no known provider prefix is recognised.

    Naming rules:
      * `OPENAI_API_KEY`               → ("openai", "openai")
      * `OPENAI_WORK_API_KEY`          → ("work", "openai")
      * `WORK_API_KEY`                 → ("work", None)
      * `API_KEY` (no stem)            → not matched; returns None
    """
    m = _API_KEY_VAR_RE.match(var)
    if not m:
        return None
    stem = m.group(1)  # everything before _API_KEY, uppercase
    if not stem:
        return None
    # Look for a known provider prefix at the start, e.g. OPENAI_WORK
    for prefix, provider in _KNOWN_PROVIDER_PREFIXES.items():
        if stem == prefix:
            # `OPENAI_API_KEY` exactly — name == provider
            return provider, provider
        if stem.startswith(prefix + "_"):
            remainder = stem[len(prefix) + 1 :]
            return remainder.lower(), provider
    return stem.lower(), None


def discover_env_keys(env: dict[str, str] | None = None) -> list[APIKey]:
    """Scan environment-style variables for `*_API_KEY` candidates.

    Reads from the live process environment plus the user's
    `secrets.env` file (legacy single-slot keys live there too).
    Returns unsaved `APIKey` objects with ``source="env"``.
    """
    if env is None:
        env = dict(os.environ)
        # File values win on collision so a managed key trumps a stale shell var.
        try:
            from podcodex.api.routes.config import _read_secrets_file

            for k, v in _read_secrets_file().items():
                env[k] = v
        except Exception:
            pass

    found: list[APIKey] = []
    for var, value in env.items():
        if not value:
            continue
        parsed = parse_env_var_name(var)
        if parsed is None:
            continue
        name, suggested = parsed
        found.append(
            APIKey(
                name=name,
                value=value,
                suggested_provider=suggested,
                source="env",
            )
        )
    found.sort(key=lambda k: k.name)
    return found


def merge_discovered(
    file: APIKeysFile, discovered: list[APIKey]
) -> tuple[APIKeysFile, list[str]]:
    """Append newly-discovered keys; never overwrite existing names.

    Returns the updated file and the list of names added.
    """
    existing_names = {k.name for k in file.keys}
    added: list[str] = []
    for k in discovered:
        if k.name in existing_names:
            continue
        file.keys.append(k)
        existing_names.add(k.name)
        added.append(k.name)
    return file, added
