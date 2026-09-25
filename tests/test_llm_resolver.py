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


def test_a_run_resolves_the_effective_model_once(isolated_storage, monkeypatch):
    """Routes used to recompute the model for provenance in several places;
    the run carries the model that will actually be called."""
    from podcodex.core.constants import DEFAULT_OLLAMA_MODEL, LLM_PROVIDER_DEFAULT_MODEL
    from podcodex.core.llm_resolver import resolve_llm_run

    monkeypatch.setattr(
        "podcodex.core.llm.list_pulled_ollama_models",
        lambda _host=None: [DEFAULT_OLLAMA_MODEL],
    )
    save_keys(APIKeysFile(keys=[APIKey(name="my-openai", value="sk-secret")]))

    api = resolve_llm_run("api", "openai", "my-openai", "")
    assert api.model == LLM_PROVIDER_DEFAULT_MODEL["openai"]
    assert api.pipeline_kwargs() == {
        "mode": "api",
        "model": LLM_PROVIDER_DEFAULT_MODEL["openai"],
        "provider": "openai",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "sk-secret",
    }

    local = resolve_llm_run("ollama", None, None, "")
    assert (local.model, local.provider, local.api_key) == (
        DEFAULT_OLLAMA_MODEL,
        None,
        None,
    )

    with pytest.raises(LLMResolutionError):
        resolve_llm_run("api", "openai", None, "gpt-4o")


def test_an_ollama_model_that_is_not_pulled_is_refused_up_front(monkeypatch):
    """Used to start the task and fail on the first batch with a bare 404."""
    from podcodex.core.llm_resolver import resolve_llm_run

    monkeypatch.setattr(
        "podcodex.core.llm.list_pulled_ollama_models",
        lambda _host=None: ["llama3.1:latest"],
    )
    assert resolve_llm_run("ollama", None, None, "llama3.1").model == "llama3.1"
    with pytest.raises(LLMResolutionError, match="No model picked.*llama3.1:latest"):
        resolve_llm_run("ollama", None, None, "")
    with pytest.raises(LLMResolutionError, match="'qwen3:4b' is not pulled"):
        resolve_llm_run("ollama", None, None, "qwen3:4b")


def test_an_unreachable_ollama_is_refused_up_front(monkeypatch):
    from podcodex.core.llm_resolver import resolve_llm_run

    def down(_host=None):
        raise ConnectionError("connection refused")

    monkeypatch.setattr("podcodex.core.llm.list_pulled_ollama_models", down)
    with pytest.raises(LLMResolutionError, match="not reachable"):
        resolve_llm_run("ollama", None, None, "llama3.1")


def test_another_server_on_the_ollama_port_is_named_as_such(monkeypatch):
    from ollama import ResponseError

    from podcodex.core.llm_resolver import resolve_llm_run

    def not_ollama(_host=None):
        raise ResponseError("", 404)

    monkeypatch.setattr("podcodex.core.llm.list_pulled_ollama_models", not_ollama)
    with pytest.raises(LLMResolutionError, match="is not Ollama"):
        resolve_llm_run("ollama", None, None, "llama3.1")
