"""Tests for Claude Desktop integration config round-trip.

Focus is JSON merge safety: every unrelated key must survive enable /
disable cycles, other ``mcpServers`` entries stay untouched, and a
corrupted existing config refuses rather than overwrites.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from podcodex.api.app import app  # noqa: E402
from podcodex.api.routes import integrations  # noqa: E402


@pytest.fixture(autouse=True)
def _redirect_config(tmp_path: Path, monkeypatch):
    """Point the integrations route at a per-test tmp config file."""
    cfg_path = tmp_path / "claude" / "claude_desktop_config.json"
    monkeypatch.setattr(integrations, "_claude_config_path", lambda: cfg_path)
    yield cfg_path


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, headers={"X-PodCodex": "1"})


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ── GET status ──────────────────────────────────────────────────────────


def test_status_reports_disabled_when_no_config(client, _redirect_config):
    r = client.get("/api/integrations/claude-desktop")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["config_path"].endswith("claude_desktop_config.json")
    assert body["command_path"].endswith("podcodex-mcp") or body[
        "command_path"
    ].endswith("podcodex-mcp.exe")
    assert body["mcp_available"] is True
    assert "Quit" in body["needs_restart_hint"]


def test_status_reports_enabled_when_entry_present(client, _redirect_config):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    _redirect_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "podcodex": {"command": "/some/path/.venv/bin/podcodex-mcp"}
                }
            }
        ),
        encoding="utf-8",
    )
    r = client.get("/api/integrations/claude-desktop")
    assert r.status_code == 200
    assert r.json()["enabled"] is True


def test_status_reports_disabled_for_unknown_command(client, _redirect_config):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    _redirect_config.write_text(
        json.dumps(
            {"mcpServers": {"podcodex": {"command": "/not/our/binary/something-else"}}}
        ),
        encoding="utf-8",
    )
    r = client.get("/api/integrations/claude-desktop")
    assert r.json()["enabled"] is False


def test_status_does_not_crash_on_corrupt_config(client, _redirect_config):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    _redirect_config.write_text("{ not: valid json", encoding="utf-8")
    r = client.get("/api/integrations/claude-desktop")
    assert r.status_code == 200
    assert r.json()["enabled"] is False


# ── Enable ──────────────────────────────────────────────────────────────


def test_enable_writes_entry_when_config_missing(client, _redirect_config):
    assert not _redirect_config.exists()
    r = client.post("/api/integrations/claude-desktop/enable")
    assert r.status_code == 200, r.text
    assert r.json()["enabled"] is True
    cfg = _read(_redirect_config)
    assert "podcodex" in cfg["mcpServers"]
    cmd = cfg["mcpServers"]["podcodex"]["command"]
    assert cmd.endswith("podcodex-mcp") or cmd.endswith("podcodex-mcp.exe")


def test_enable_preserves_unrelated_top_level_keys(client, _redirect_config):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    _redirect_config.write_text(
        json.dumps(
            {
                "preferences": {"theme": "dark", "telemetry": False},
                "someOtherKey": [1, 2, 3],
            }
        ),
        encoding="utf-8",
    )
    client.post("/api/integrations/claude-desktop/enable")
    cfg = _read(_redirect_config)
    assert cfg["preferences"] == {"theme": "dark", "telemetry": False}
    assert cfg["someOtherKey"] == [1, 2, 3]
    assert "podcodex" in cfg["mcpServers"]


def test_enable_preserves_other_mcp_servers(client, _redirect_config):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    _redirect_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "zotero": {"command": "uvx", "args": ["zotero-mcp"]},
                    "brave": {"command": "npx", "args": ["@brave/search"]},
                }
            }
        ),
        encoding="utf-8",
    )
    client.post("/api/integrations/claude-desktop/enable")
    cfg = _read(_redirect_config)
    assert set(cfg["mcpServers"]) == {"zotero", "brave", "podcodex"}
    assert cfg["mcpServers"]["zotero"] == {"command": "uvx", "args": ["zotero-mcp"]}
    assert cfg["mcpServers"]["brave"] == {"command": "npx", "args": ["@brave/search"]}


def test_enable_refuses_on_corrupt_config(client, _redirect_config):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    _redirect_config.write_text("{ not: valid json", encoding="utf-8")
    r = client.post("/api/integrations/claude-desktop/enable")
    assert r.status_code == 422
    assert "valid JSON" in r.json()["detail"]
    assert _redirect_config.read_text(encoding="utf-8") == "{ not: valid json"


# ── Disable ─────────────────────────────────────────────────────────────


def test_disable_removes_only_podcodex(client, _redirect_config):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    _redirect_config.write_text(
        json.dumps(
            {
                "preferences": {"theme": "dark"},
                "mcpServers": {
                    "podcodex": {"command": "/path/to/.venv/bin/podcodex-mcp"},
                    "zotero": {"command": "uvx", "args": ["zotero-mcp"]},
                },
            }
        ),
        encoding="utf-8",
    )
    r = client.post("/api/integrations/claude-desktop/disable")
    assert r.status_code == 200
    assert r.json()["enabled"] is False
    cfg = _read(_redirect_config)
    assert cfg["preferences"] == {"theme": "dark"}
    assert list(cfg["mcpServers"]) == ["zotero"]


def test_disable_drops_mcpservers_when_podcodex_was_only_entry(
    client, _redirect_config
):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    _redirect_config.write_text(
        json.dumps(
            {
                "preferences": {"x": 1},
                "mcpServers": {
                    "podcodex": {"command": "/path/to/.venv/bin/podcodex-mcp"}
                },
            }
        ),
        encoding="utf-8",
    )
    client.post("/api/integrations/claude-desktop/disable")
    cfg = _read(_redirect_config)
    assert cfg == {"preferences": {"x": 1}}


def test_disable_is_noop_when_entry_absent(client, _redirect_config):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    before = {"preferences": {"theme": "dark"}, "mcpServers": {"zotero": {}}}
    _redirect_config.write_text(json.dumps(before), encoding="utf-8")
    r = client.post("/api/integrations/claude-desktop/disable")
    assert r.status_code == 200
    assert _read(_redirect_config) == before


def test_enable_then_disable_round_trips_cleanly(client, _redirect_config):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    original = {
        "preferences": {"theme": "dark", "nested": {"a": [1, 2]}},
        "mcpServers": {"zotero": {"command": "uvx", "args": ["zotero-mcp"]}},
    }
    _redirect_config.write_text(json.dumps(original), encoding="utf-8")
    client.post("/api/integrations/claude-desktop/enable")
    client.post("/api/integrations/claude-desktop/disable")
    assert _read(_redirect_config) == original


# ── WSL ─────────────────────────────────────────────────────────────────


@pytest.fixture
def _force_wsl(monkeypatch):
    """Pretend we're inside WSL with a known distro."""
    monkeypatch.setattr(integrations, "_is_wsl", lambda: True)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")


def test_wsl_enable_writes_wsl_exe_entry(client, _redirect_config, _force_wsl):
    r = client.post("/api/integrations/claude-desktop/enable")
    assert r.status_code == 200, r.text
    assert r.json()["enabled"] is True
    entry = _read(_redirect_config)["mcpServers"]["podcodex"]
    assert entry["command"] == "wsl.exe"
    assert entry["args"][:2] == ["-d", "Ubuntu"]
    assert entry["args"][2] == "-e"
    assert entry["args"][3].endswith("podcodex-mcp")


def test_wsl_status_detects_wsl_exe_shape(client, _redirect_config, _force_wsl):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    _redirect_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "podcodex": {
                        "command": "wsl.exe",
                        "args": [
                            "-d",
                            "Ubuntu",
                            "-e",
                            "/home/me/PodCodex/.venv/bin/podcodex-mcp",
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    r = client.get("/api/integrations/claude-desktop")
    assert r.json()["enabled"] is True


def test_wsl_disable_removes_wsl_exe_entry(client, _redirect_config, _force_wsl):
    _redirect_config.parent.mkdir(parents=True, exist_ok=True)
    _redirect_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "podcodex": {
                        "command": "wsl.exe",
                        "args": [
                            "-e",
                            "/home/me/PodCodex/.venv/bin/podcodex-mcp",
                        ],
                    },
                    "zotero": {"command": "uvx", "args": ["zotero-mcp"]},
                }
            }
        ),
        encoding="utf-8",
    )
    client.post("/api/integrations/claude-desktop/disable")
    cfg = _read(_redirect_config)
    assert list(cfg["mcpServers"]) == ["zotero"]


def test_wsl_enable_without_distro_omits_distro_flag(
    client, _redirect_config, monkeypatch
):
    monkeypatch.setattr(integrations, "_is_wsl", lambda: True)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    client.post("/api/integrations/claude-desktop/enable")
    entry = _read(_redirect_config)["mcpServers"]["podcodex"]
    assert entry["command"] == "wsl.exe"
    assert "-d" not in entry["args"]
    assert entry["args"][0] == "-e"
    assert entry["args"][1].endswith("podcodex-mcp")
