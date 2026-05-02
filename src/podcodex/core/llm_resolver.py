"""Resolve `(provider_profile, key_name)` → low-level LLM call params.

The API receives the user's pick: a provider profile name plus an
optional key name from the pool. Pipeline core code still works in
terms of `(provider, api_base_url, api_key)` so this module bridges
the two — looking up the profile in the catalog (built-ins + custom)
and the key value in the pool.

Returned ``provider`` matches the legacy `LLM_PROVIDERS` keys so
``run_api()`` can keep its existing dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from podcodex.core.api_keys import find_key, load_keys
from podcodex.core.provider_profiles import find_profile

LegacyProvider = Literal["openai", "anthropic", "mistral", "ollama", "custom"]


@dataclass(frozen=True)
class ResolvedLLM:
    """Concrete LLM params after profile + key lookup.

    ``provider`` is the shorthand expected by the legacy ``run_api()``
    dispatch. ``api_key`` is ``None`` only for local providers (Ollama).
    """

    provider: LegacyProvider
    api_base_url: str
    api_key: str | None


class LLMResolutionError(ValueError):
    """Raised when the requested profile/key pair can't be resolved."""


def resolve_llm(provider_profile: str | None, key_name: str | None) -> ResolvedLLM:
    """Resolve the user's profile + key pick to concrete params.

    Rules:
      * ``provider_profile`` must be set; ``ollama`` needs no key.
      * For api-mode profiles, ``key_name`` must point to a pool entry.
      * Missing profile or key raises ``LLMResolutionError`` so the
        caller can surface it as a 400 response.
    """
    if not provider_profile:
        raise LLMResolutionError("provider_profile is required")

    profile = find_profile(provider_profile)
    if profile is None:
        raise LLMResolutionError(f"Unknown provider profile: {provider_profile!r}")

    if profile.type == "ollama":
        return ResolvedLLM(provider="ollama", api_base_url="", api_key=None)

    if not key_name:
        raise LLMResolutionError(
            f"Profile {provider_profile!r} needs an API key — pick one from the pool"
        )
    entry = find_key(load_keys(), key_name)
    if entry is None:
        raise LLMResolutionError(f"Unknown API key: {key_name!r}")

    if not profile.base_url:
        raise LLMResolutionError(
            f"Profile {provider_profile!r} has no base_url configured"
        )

    legacy: LegacyProvider = (
        "custom" if profile.type == "openai-compatible" else profile.type
    )
    return ResolvedLLM(
        provider=legacy,
        api_base_url=profile.base_url,
        api_key=entry.value,
    )
