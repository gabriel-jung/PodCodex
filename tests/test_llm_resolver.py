"""Tests for the (provider_profile, key_name) → low-level params resolver."""

from __future__ import annotations

import pytest

from podcodex.core import api_keys as keys_mod
from podcodex.core import provider_profiles as pp_mod
from podcodex.core.api_keys import APIKey, APIKeysFile, save_keys
from podcodex.core.llm_resolver import LLMResolutionError, resolve_llm
from podcodex.core.provider_profiles import (
    CustomProfile,
    ProviderProfilesFile,
    save_custom,
)


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(keys_mod, "api_keys_path", lambda: tmp_path / "api_keys.json")
    monkeypatch.setattr(
        pp_mod,
        "provider_profiles_path",
        lambda: tmp_path / "provider_profiles.json",
    )
    return tmp_path


def test_resolve_builtin_openai(isolated_storage):
    save_keys(APIKeysFile(keys=[APIKey(name="my-openai", value="sk-secret")]))
    resolved = resolve_llm("openai", "my-openai")
    assert resolved.provider == "openai"
    assert resolved.api_base_url == "https://api.openai.com/v1"
    assert resolved.api_key == "sk-secret"


def test_resolve_ollama_no_key_needed(isolated_storage):
    resolved = resolve_llm("ollama", None)
    assert resolved.provider == "ollama"
    assert resolved.api_key is None


def test_resolve_custom_profile(isolated_storage):
    save_custom(
        ProviderProfilesFile(
            profiles=[CustomProfile(name="Groq", base_url="https://api.groq.com/v1")]
        )
    )
    save_keys(APIKeysFile(keys=[APIKey(name="groq-key", value="gsk-secret")]))
    resolved = resolve_llm("Groq", "groq-key")
    assert resolved.provider == "custom"
    assert resolved.api_base_url == "https://api.groq.com/v1"
    assert resolved.api_key == "gsk-secret"


def test_resolve_unknown_profile_raises(isolated_storage):
    with pytest.raises(LLMResolutionError):
        resolve_llm("not-a-profile", "anything")


def test_resolve_missing_profile_raises(isolated_storage):
    with pytest.raises(LLMResolutionError):
        resolve_llm("", "x")
    with pytest.raises(LLMResolutionError):
        resolve_llm(None, "x")


def test_resolve_api_profile_without_key_raises(isolated_storage):
    with pytest.raises(LLMResolutionError):
        resolve_llm("openai", None)
    with pytest.raises(LLMResolutionError):
        resolve_llm("openai", "")


def test_resolve_unknown_key_raises(isolated_storage):
    with pytest.raises(LLMResolutionError):
        resolve_llm("openai", "missing")
