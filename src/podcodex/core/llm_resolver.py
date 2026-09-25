"""Resolve `(provider_profile, key_name)` → low-level LLM call params.

The API receives the user's pick: a provider profile name plus an
optional key name from the pool. Pipeline core code still works in
terms of `(provider, api_base_url, api_key)` so this module bridges
the two — looking up the profile in the catalog (built-ins + custom)
and the key value in the pool.

Returned ``provider`` matches the legacy ``LLM_PROVIDER_DEFAULT_MODEL`` keys
so ``run_api()`` can keep its existing dispatch.
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
            f"Profile {provider_profile!r} needs an API key. Pick one from the pool."
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


@dataclass(frozen=True)
class LLMRun:
    """Everything an auto correct / translate run needs, resolved once.

    ``model`` is the model that will actually run (the provider default when
    the pick was empty), so provenance, ``llm_failures.json`` and the batch
    skip check all record and compare the same value.
    """

    mode: str
    model: str
    provider: LegacyProvider | None
    api_base_url: str
    api_key: str | None

    def pipeline_kwargs(self) -> dict:
        """The keyword arguments ``correct_segments`` / ``translate_segments`` take."""
        return {
            "mode": self.mode,
            "model": self.model,
            "provider": self.provider,
            "api_base_url": self.api_base_url,
            "api_key": self.api_key,
        }


def resolve_llm_run(
    mode: str, provider_profile: str | None, key_name: str | None, model: str
) -> LLMRun:
    """Resolve a request's LLM pick for any mode.

    API mode resolves the profile and key (raising ``LLMResolutionError``);
    Ollama and manual need neither, but an Ollama model that is not pulled
    (or a daemon that does not answer) is refused here, with the reason,
    rather than as a 404 once the task is running.
    """
    from podcodex.core.llm import effective_llm_model

    resolved = resolve_llm(provider_profile, key_name) if mode == "api" else None
    provider = resolved.provider if resolved else None
    run_model = effective_llm_model(mode, model, provider)
    if mode == "ollama":
        _require_pulled(run_model, picked=bool(model))
    return LLMRun(
        mode=mode,
        model=run_model,
        provider=provider,
        api_base_url=resolved.api_base_url if resolved else "",
        api_key=resolved.api_key if resolved else None,
    )


def _require_pulled(model: str, *, picked: bool) -> None:
    """Raise ``LLMResolutionError`` unless the local Ollama has *model*."""
    from podcodex.core.llm import _ollama_model_key, ollama_host, probe_ollama

    host = ollama_host()
    probe = probe_ollama(host)
    if probe["problem"] == "port_taken":
        raise LLMResolutionError(
            f"The server at {host} is not Ollama ({probe['error']}): another "
            "program holds the port, so Ollama cannot start there. Stop it, or "
            "run Ollama on another port and set OLLAMA_HOST."
        )
    if probe["problem"]:
        raise LLMResolutionError(
            f"Ollama is not reachable at {host} ({probe['error']}). Is it running?"
        )
    pulled = probe["models"]
    if _ollama_model_key(model) in pulled:
        return
    which = (
        f"Model {model!r}" if picked else f"No model picked, and the default {model!r}"
    )
    available = ", ".join(pulled) or "none"
    raise LLMResolutionError(
        f"{which} is not pulled in Ollama (pulled: {available}). "
        f"Pick one, or run `ollama pull {model}`."
    )
