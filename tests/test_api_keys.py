"""Tests for the named API key pool: storage, discovery, routes."""

from __future__ import annotations

import os
import stat

import pytest
from fastapi.testclient import TestClient

from podcodex.api.app import create_app
from podcodex.core import api_keys as keys_mod
from podcodex.core.api_keys import (
    APIKey,
    APIKeysFile,
    discover_env_keys,
    load_keys,
    merge_discovered,
    parse_env_var_name,
    save_keys,
)


# ── parse_env_var_name ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "var,expected",
    [
        ("OPENAI_API_KEY", ("openai", "openai")),
        ("ANTHROPIC_API_KEY", ("anthropic", "anthropic")),
        ("MISTRAL_API_KEY", ("mistral", "mistral")),
        ("OPENAI_WORK_API_KEY", ("work", "openai")),
        ("ANTHROPIC_PERSONAL_API_KEY", ("personal", "anthropic")),
        ("MYAPP_API_KEY", ("myapp", None)),
        ("LLAMA_CLOUD_API_KEY", ("llama_cloud", None)),
        ("API_KEY", None),  # No stem before _API_KEY
        ("HF_TOKEN", None),
        ("OPENAI_API_KEY_BACKUP", None),  # Suffix mismatch
        ("openai_api_key", None),  # Lowercase rejected
    ],
)
def test_parse_env_var_name(var, expected):
    assert parse_env_var_name(var) == expected


# ── discover_env_keys ────────────────────────────────────────────────


def test_discover_env_keys_explicit_env():
    env = {
        "OPENAI_API_KEY": "sk-openai-1",
        "OPENAI_WORK_API_KEY": "sk-work",
        "MYAPP_API_KEY": "myapp-secret",
        "HF_TOKEN": "hf_xxx",
        "PATH": "/usr/bin",
    }
    result = discover_env_keys(env)
    names = [k.name for k in result]
    # Sorted alphabetically
    assert names == ["myapp", "openai", "work"]
    by_name = {k.name: k for k in result}
    assert by_name["openai"].suggested_provider == "openai"
    assert by_name["openai"].value == "sk-openai-1"
    assert by_name["work"].suggested_provider == "openai"
    assert by_name["myapp"].suggested_provider is None
    assert all(k.source == "env" for k in result)


def test_discover_env_keys_skips_empty_values():
    env = {"OPENAI_API_KEY": "", "MYAPP_API_KEY": "real"}
    result = discover_env_keys(env)
    assert [k.name for k in result] == ["myapp"]


# ── merge_discovered ─────────────────────────────────────────────────


def test_merge_discovered_appends_only_new():
    file = APIKeysFile(keys=[APIKey(name="existing", value="old", source="ui")])
    discovered = [
        APIKey(name="existing", value="env-version", source="env"),
        APIKey(name="newone", value="fresh", source="env"),
    ]
    file, added = merge_discovered(file, discovered)
    assert added == ["newone"]
    by_name = {k.name: k for k in file.keys}
    assert by_name["existing"].value == "old"  # unchanged
    assert by_name["existing"].source == "ui"
    assert by_name["newone"].value == "fresh"


# ── persistence ──────────────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(keys_mod, "api_keys_path", lambda: tmp_path / "api_keys.json")
    file = APIKeysFile(
        keys=[
            APIKey(name="a", value="va", suggested_provider="openai", source="ui"),
            APIKey(name="b", value="vb", source="env"),
        ]
    )
    save_keys(file)
    loaded = load_keys()
    assert len(loaded.keys) == 2
    assert loaded.keys[0].name == "a"
    assert loaded.keys[0].value == "va"
    assert loaded.keys[0].suggested_provider == "openai"


def test_save_keys_writes_mode_0600(tmp_path, monkeypatch):
    path = tmp_path / "api_keys.json"
    monkeypatch.setattr(keys_mod, "api_keys_path", lambda: path)
    save_keys(APIKeysFile(keys=[APIKey(name="a", value="v")]))
    if os.name != "nt":  # chmod meaningful on POSIX only
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600


def test_load_returns_empty_for_corrupt_file(tmp_path, monkeypatch):
    path = tmp_path / "api_keys.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(keys_mod, "api_keys_path", lambda: path)
    loaded = load_keys()
    assert loaded.keys == []


# ── route tests ──────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with isolated api_keys.json + secrets.env."""
    from podcodex.api.routes import config as config_mod

    monkeypatch.setattr(keys_mod, "api_keys_path", lambda: tmp_path / "api_keys.json")
    monkeypatch.setattr(
        config_mod, "secrets_env_path", lambda: tmp_path / "secrets.env"
    )
    for var in list(os.environ):
        if var.endswith("_API_KEY"):
            monkeypatch.delenv(var, raising=False)
    app = create_app()
    return TestClient(app, headers={"X-PodCodex": "1"})


def test_list_keys_empty(client):
    r = client.get("/api/keys")
    assert r.status_code == 200
    body = r.json()
    assert body["keys"] == []
    assert body["path"].endswith("api_keys.json")


def test_create_then_list(client):
    r = client.post(
        "/api/keys",
        json={"name": "work", "value": "sk-secret", "suggested_provider": "openai"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "work"
    assert body["masked"] == "sk-s****"
    assert body["suggested_provider"] == "openai"
    assert body["source"] == "ui"

    r = client.get("/api/keys")
    assert r.status_code == 200
    keys = r.json()["keys"]
    assert len(keys) == 1
    assert keys[0]["name"] == "work"
    # Value never leaks back
    assert "value" not in keys[0]


def test_create_duplicate_name_rejected(client):
    client.post("/api/keys", json={"name": "x", "value": "v1"})
    r = client.post("/api/keys", json={"name": "x", "value": "v2"})
    assert r.status_code == 409


def test_patch_value_and_provider(client):
    client.post("/api/keys", json={"name": "a", "value": "v1"})
    r = client.patch(
        "/api/keys/a",
        json={"value": "v2-longer", "suggested_provider": "anthropic"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["masked"] == "v2-l****"
    assert body["suggested_provider"] == "anthropic"


def test_patch_clears_provider_with_empty_string(client):
    client.post(
        "/api/keys",
        json={"name": "a", "value": "v", "suggested_provider": "openai"},
    )
    r = client.patch("/api/keys/a", json={"suggested_provider": ""})
    assert r.status_code == 200
    assert r.json()["suggested_provider"] is None


def test_patch_unknown_returns_404(client):
    r = client.patch("/api/keys/nope", json={"value": "x"})
    assert r.status_code == 404


def test_delete_key(client):
    client.post("/api/keys", json={"name": "a", "value": "v"})
    r = client.delete("/api/keys/a")
    assert r.status_code == 204
    assert client.get("/api/keys").json()["keys"] == []


def test_delete_unknown_returns_404(client):
    r = client.delete("/api/keys/nope")
    assert r.status_code == 404


def test_scan_env_seeds_pool(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("MYAPP_API_KEY", "secret-app")
    r = client.post("/api/keys/scan-env")
    assert r.status_code == 200
    body = r.json()
    assert sorted(body["added"]) == ["myapp", "openai"]
    by_name = {k["name"]: k for k in body["keys"]}
    assert by_name["openai"]["suggested_provider"] == "openai"
    assert by_name["openai"]["source"] == "env"
    assert by_name["myapp"]["suggested_provider"] is None


def test_scan_env_does_not_overwrite_ui_keys(client, monkeypatch):
    client.post("/api/keys", json={"name": "openai", "value": "manual-value"})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fromenv")
    r = client.post("/api/keys/scan-env")
    body = r.json()
    assert body["added"] == []  # name collision
    by_name = {k["name"]: k for k in body["keys"]}
    assert by_name["openai"]["source"] == "ui"
    # masked should reflect the manual value, not the env one
    assert by_name["openai"]["masked"] == "manu****"


def test_scan_env_reads_secrets_file(client, tmp_path):
    secrets = tmp_path / "secrets.env"
    secrets.write_text('FROMFILE_API_KEY="filevalue"\n', encoding="utf-8")
    r = client.post("/api/keys/scan-env")
    body = r.json()
    assert "fromfile" in body["added"]
