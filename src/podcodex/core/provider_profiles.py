"""Provider profile catalog — built-in + user-defined.

Built-in profiles (openai, anthropic, mistral, ollama) are hardcoded
read-only and resolve to the canonical SDK + base URL in the LLM
runtime. User-added profiles are always ``openai-compatible``: they
need a ``base_url`` and route through the OpenAI SDK with that URL.

Custom profiles persist at ``<config_dir>/provider_profiles.json``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from podcodex.core.app_paths import config_dir
from podcodex.core.json_store import JsonModelStore

ProviderType = Literal["openai", "anthropic", "mistral", "ollama", "openai-compatible"]


class ProviderProfile(BaseModel):
    """Profile that resolves to an LLM client at runtime.

    ``builtin`` is computed at read time, never persisted.
    """

    name: str = Field(..., min_length=1, max_length=80)
    type: ProviderType
    base_url: str | None = None
    builtin: bool = False

    @field_validator("base_url")
    @classmethod
    def _strip_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v.strip() or None


class CustomProfile(BaseModel):
    """A user-added profile. Always openai-compatible — base_url required."""

    name: str = Field(..., min_length=1, max_length=80)
    base_url: str = Field(..., min_length=1)

    @field_validator("base_url")
    @classmethod
    def _strip_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("base_url is required for custom profiles")
        return v


class ProviderProfilesFile(BaseModel):
    profiles: list[CustomProfile] = []


# Hardcoded built-ins. Order is render order in the UI dropdown.
# The "openai-compatible" entries below all route through the OpenAI SDK with
# a custom base_url — see ``llm_resolver.resolve_llm`` and ``run_api`` in
# ``core/_utils.py``. Adding more is just appending here; no runtime changes.
BUILTIN_PROFILES: tuple[ProviderProfile, ...] = (
    ProviderProfile(
        name="openai",
        type="openai",
        base_url="https://api.openai.com/v1",
        builtin=True,
    ),
    ProviderProfile(
        name="anthropic",
        type="anthropic",
        base_url="https://api.anthropic.com/v1",
        builtin=True,
    ),
    ProviderProfile(
        name="mistral",
        type="mistral",
        base_url="https://api.mistral.ai/v1",
        builtin=True,
    ),
    ProviderProfile(
        name="deepseek",
        type="openai-compatible",
        base_url="https://api.deepseek.com",
        builtin=True,
    ),
    ProviderProfile(
        name="gemini",
        type="openai-compatible",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        builtin=True,
    ),
    ProviderProfile(
        name="groq",
        type="openai-compatible",
        base_url="https://api.groq.com/openai/v1",
        builtin=True,
    ),
    ProviderProfile(
        name="openrouter",
        type="openai-compatible",
        base_url="https://openrouter.ai/api/v1",
        builtin=True,
    ),
    ProviderProfile(
        name="ollama",
        type="ollama",
        base_url=None,
        builtin=True,
    ),
)

_BUILTIN_NAMES: frozenset[str] = frozenset(p.name for p in BUILTIN_PROFILES)


def provider_profiles_path() -> Path:
    return config_dir() / "provider_profiles.json"


_STORE = JsonModelStore(lambda: provider_profiles_path(), ProviderProfilesFile)


def load_custom() -> ProviderProfilesFile:
    """Load the persisted custom profiles (a copy). Empty if missing or unreadable."""
    return _STORE.load()


def save_custom(file: ProviderProfilesFile) -> None:
    _STORE.save(file)


def mutate_custom(
    fn: Callable[[ProviderProfilesFile], bool | None],
) -> ProviderProfilesFile:
    """Load-modify-save the custom profiles under one lock."""
    return _STORE.mutate(fn)


def list_all() -> list[ProviderProfile]:
    """Return built-ins followed by custom profiles."""
    custom = load_custom()
    return [
        *BUILTIN_PROFILES,
        *(
            ProviderProfile(
                name=c.name,
                type="openai-compatible",
                base_url=c.base_url,
                builtin=False,
            )
            for c in custom.profiles
        ),
    ]


def find_profile(name: str) -> ProviderProfile | None:
    for p in list_all():
        if p.name == name:
            return p
    return None


def is_builtin(name: str) -> bool:
    return name in _BUILTIN_NAMES


def find_custom(file: ProviderProfilesFile, name: str) -> CustomProfile | None:
    for c in file.profiles:
        if c.name == name:
            return c
    return None
