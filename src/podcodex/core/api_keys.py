"""Named API key pool — generic credential storage.

Replaces the old single-slot `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
`MISTRAL_API_KEY` model. The pool is a flat list of `(name, value,
suggested_provider?)` entries: storage decoupled from provider so the
same key can be paired with any provider profile at usage sites.

Persisted at `<config_dir>/api_keys.json` with mode 0600.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from podcodex.core.app_paths import config_dir
from podcodex.core.json_store import JsonModelStore

_KNOWN_PROVIDER_PREFIXES: dict[str, str] = {
    "OPENAI": "openai",
    "ANTHROPIC": "anthropic",
    "MISTRAL": "mistral",
    "DEEPSEEK": "deepseek",
    "GEMINI": "gemini",
    "GROQ": "groq",
    "OPENROUTER": "openrouter",
}

_API_KEY_VAR_RE = re.compile(r"^([A-Z0-9_]+)_API_KEY$")


def read_secrets_file() -> dict[str, str]:
    """Parse ``secrets.env`` into a dict. Empty dict if absent.

    Lives here rather than in the config route because ``core`` reads it
    too, and reaching it through ``api.routes.config`` made the base layer
    import the whole API surface (and fastapi) to read a dotenv file.
    """
    from dotenv import dotenv_values

    from podcodex.core.app_paths import secrets_env_path

    path = secrets_env_path()
    if not path.exists():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v}


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


# mtime-keyed so high-volume callers (batch resolver, per-route CRUD) avoid
# re-parsing the file on every hit. Mode 0600: the values are secrets.
_STORE = JsonModelStore(lambda: api_keys_path(), APIKeysFile, file_mode=0o600)


def load_keys() -> APIKeysFile:
    """Load the pool from disk (a copy); empty if missing or unreadable.

    An unreadable file is moved aside on the next save instead of being
    overwritten by the empty pool (see ``JsonModelStore``).
    """
    return _STORE.load()


def save_keys(file: APIKeysFile) -> None:
    """Persist atomically with mode 0600."""
    _STORE.save(file)


def mutate_keys(fn: Callable[[APIKeysFile], bool | None]) -> APIKeysFile:
    """Load-modify-save the pool under one lock (see ``JsonModelStore.mutate``)."""
    return _STORE.mutate(fn)


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


def discover_env_keys(
    env: dict[str, str] | None = None,
    secrets: dict[str, str] | None = None,
) -> list[APIKey]:
    """Scan for `*_API_KEY` candidates to seed the pool.

    Two sources: the process environment (*env*, default ``os.environ``) and
    the user's ``secrets.env`` (*secrets*, read from disk when *env* is not
    given), whose
    values win on collision so a managed key trumps a stale shell var.
    From the environment, only names with a known provider prefix are taken
    (`OPENAI_API_KEY`, `OPENAI_WORK_API_KEY`, not `WORK_API_KEY`): a shell
    carries plenty of other services' secrets (cloud, CI, payment) that have
    no business in a pool any provider profile can be paired with.
    ``secrets.env`` is PodCodex's own file, so every entry in it counts.
    Returns unsaved `APIKey` objects with ``source="env"``.
    """
    if env is None:
        env = dict(os.environ)
        if secrets is None:
            try:
                secrets = read_secrets_file()
            except Exception:
                secrets = {}
    secrets = secrets or {}

    candidates: dict[str, tuple[str, bool]] = {
        var: (value, False) for var, value in env.items()
    }
    candidates.update({var: (value, True) for var, value in secrets.items()})

    found: list[APIKey] = []
    for var, (value, from_secrets) in candidates.items():
        if not value:
            continue
        parsed = parse_env_var_name(var)
        if parsed is None:
            continue
        name, suggested = parsed
        if suggested is None and not from_secrets:
            continue
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
