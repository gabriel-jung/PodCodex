"""Tests for provider profile catalog: built-ins + custom CRUD."""

from __future__ import annotations

import pytest

from podcodex.api.app import create_app
from podcodex.core import provider_profiles as pp_mod
from podcodex.core.provider_profiles import (
    BUILTIN_PROFILES,
    CustomProfile,
    ProviderProfilesFile,
    list_all,
    load_custom,
    save_custom,
)
from tests.fixtures.api_client import client_for  # noqa: E402


# ── built-ins ────────────────────────────────────────────────────────


# ── persistence ──────────────────────────────────────────────────────


def test_save_load_custom_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pp_mod, "provider_profiles_path", lambda: tmp_path / "provider_profiles.json"
    )
    file = ProviderProfilesFile(
        profiles=[CustomProfile(name="Groq", base_url="https://api.groq.com/openai/v1")]
    )
    save_custom(file)
    loaded = load_custom()
    assert len(loaded.profiles) == 1
    assert loaded.profiles[0].name == "Groq"
    assert loaded.profiles[0].base_url == "https://api.groq.com/openai/v1"


def test_list_all_includes_builtins_and_custom(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pp_mod, "provider_profiles_path", lambda: tmp_path / "provider_profiles.json"
    )
    save_custom(
        ProviderProfilesFile(
            profiles=[CustomProfile(name="Groq", base_url="https://api.groq.com")]
        )
    )
    profiles = list_all()
    names = [p.name for p in profiles]
    assert names[: len(BUILTIN_PROFILES)] == [p.name for p in BUILTIN_PROFILES]
    # A builtin appended without builtin=True would be offered for edit/delete.
    assert all(p.builtin for p in profiles[: len(BUILTIN_PROFILES)])
    assert "Groq" in names
    groq = next(p for p in profiles if p.name == "Groq")
    assert groq.builtin is False
    assert groq.type == "openai-compatible"


# ── route tests ──────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pp_mod, "provider_profiles_path", lambda: tmp_path / "provider_profiles.json"
    )
    app = create_app()
    return client_for(app)


def test_create_custom_profile(client):
    r = client.post(
        "/api/provider-profiles",
        json={"name": "Groq", "base_url": "https://api.groq.com/openai/v1"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "Groq"
    assert body["type"] == "openai-compatible"
    assert body["base_url"] == "https://api.groq.com/openai/v1"
    assert body["builtin"] is False

    listed = client.get("/api/provider-profiles").json()["profiles"]
    assert any(p["name"] == "Groq" for p in listed)


def test_create_collides_with_builtin(client):
    r = client.post(
        "/api/provider-profiles",
        json={"name": "openai", "base_url": "https://example.com"},
    )
    assert r.status_code == 409


def test_create_duplicate_custom(client):
    client.post(
        "/api/provider-profiles",
        json={"name": "Groq", "base_url": "https://api.groq.com"},
    )
    r = client.post(
        "/api/provider-profiles",
        json={"name": "Groq", "base_url": "https://api.groq.com"},
    )
    assert r.status_code == 409


def test_patch_custom_base_url(client):
    client.post(
        "/api/provider-profiles",
        json={"name": "Groq", "base_url": "https://old.example.com"},
    )
    r = client.patch(
        "/api/provider-profiles/Groq",
        json={"base_url": "https://new.example.com"},
    )
    assert r.status_code == 200
    assert r.json()["base_url"] == "https://new.example.com"


def test_patch_builtin_rejected(client):
    r = client.patch(
        "/api/provider-profiles/openai",
        json={"base_url": "https://hijack.example.com"},
    )
    assert r.status_code == 403


def test_patch_unknown_returns_404(client):
    r = client.patch(
        "/api/provider-profiles/missing",
        json={"base_url": "https://example.com"},
    )
    assert r.status_code == 404


def test_delete_custom(client):
    client.post(
        "/api/provider-profiles",
        json={"name": "Groq", "base_url": "https://api.groq.com"},
    )
    r = client.delete("/api/provider-profiles/Groq")
    assert r.status_code == 204
    listed = client.get("/api/provider-profiles").json()["profiles"]
    assert all(p["name"] != "Groq" for p in listed)


def test_delete_builtin_rejected(client):
    r = client.delete("/api/provider-profiles/openai")
    assert r.status_code == 403
