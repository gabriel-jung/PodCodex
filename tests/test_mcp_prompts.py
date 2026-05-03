"""Tests for user-managed MCP prompts (storage + CRUD + validation)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from podcodex.api.app import app  # noqa: E402
from podcodex.mcp import prompts as prompts_mod  # noqa: E402
from podcodex.mcp.prompts import (  # noqa: E402
    PromptDef,
    PromptValidationError,
    SlotDef,
    load_prompts,
    save_prompts,
    validate_prompt,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch):
    """Redirect prompt storage to a tmp file for every test."""
    store = tmp_path / "mcp_prompts.json"
    monkeypatch.setattr(prompts_mod, "_prompts_path", lambda: store)
    yield store


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, headers={"X-PodCodex": "1"})


# ── validate_prompt ─────────────────────────────────────────────────────


def test_validate_accepts_well_formed():
    validate_prompt(
        PromptDef(
            id="my_prompt",
            name="mine",
            title="T",
            description="",
            template="Hello {name}",
            slots=[SlotDef(name="name")],
        )
    )


def test_validate_rejects_bad_slug():
    with pytest.raises(PromptValidationError):
        validate_prompt(
            PromptDef(
                id="BadID",
                name="bad",
                title="T",
                description="",
                template="hi",
            )
        )


def test_validate_rejects_tool_name_collision():
    with pytest.raises(PromptValidationError, match="built-in tool"):
        validate_prompt(
            PromptDef(
                id="search",
                name="x",
                title="T",
                description="",
                template="hi",
            )
        )


def test_validate_rejects_undeclared_slot():
    with pytest.raises(PromptValidationError, match="undeclared slots"):
        validate_prompt(
            PromptDef(
                id="ok_id",
                name="o",
                title="T",
                description="",
                template="Hi {name} and {other}",
                slots=[SlotDef(name="name")],
            )
        )


# ── Storage ─────────────────────────────────────────────────────────────


def test_load_seeds_builtins_on_first_run(_isolated_store):
    prompts = load_prompts()
    ids = [p.id for p in prompts]
    assert set(ids) == {"brief", "speaker", "quote", "compare", "timeline"}
    assert all(p.is_builtin for p in prompts)


def test_round_trip_preserves_user_prompt(_isolated_store):
    prompts = load_prompts()
    prompts.append(
        PromptDef(
            id="custom",
            name="custom",
            title="Custom",
            description="",
            template="Hi {x}",
            slots=[SlotDef(name="x")],
        )
    )
    save_prompts(prompts)
    reloaded = load_prompts()
    assert any(p.id == "custom" and not p.is_builtin for p in reloaded)


def test_deleted_builtin_is_reseeded_on_reload(_isolated_store):
    prompts = [p for p in load_prompts() if p.id != "brief"]
    save_prompts(prompts)
    reloaded = load_prompts()
    assert "brief" in {p.id for p in reloaded}


# ── CRUD API ────────────────────────────────────────────────────────────


def test_list_returns_builtins(client):
    r = client.get("/api/mcp/prompts")
    assert r.status_code == 200
    ids = {p["id"] for p in r.json()}
    assert {"brief", "speaker", "quote", "compare", "timeline"} <= ids


def test_create_persists_and_registers(client):
    payload = {
        "id": "my_tool",
        "name": "mine",
        "title": "My tool",
        "description": "test",
        "template": "Hi {name}",
        "slots": [{"name": "name"}],
    }
    r = client.post("/api/mcp/prompts", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] == "my_tool"
    assert body["is_builtin"] is False

    # Live registration
    from podcodex.mcp.server import mcp

    assert "my_tool" in mcp._prompt_manager._prompts


def test_create_rejects_duplicate(client):
    payload = {"id": "brief", "name": "b", "title": "T", "template": "hi"}
    r = client.post("/api/mcp/prompts", json=payload)
    assert r.status_code == 409


def test_create_rejects_invalid_slug(client):
    payload = {"id": "BadID", "name": "x", "title": "T", "template": "hi"}
    r = client.post("/api/mcp/prompts", json=payload)
    assert r.status_code == 422


def test_create_rejects_undeclared_slot(client):
    payload = {
        "id": "ok_id2",
        "name": "ok",
        "title": "T",
        "template": "Hi {name} and {extra}",
        "slots": [{"name": "name"}],
    }
    r = client.post("/api/mcp/prompts", json=payload)
    assert r.status_code == 422


def test_update_user_prompt(client):
    client.post(
        "/api/mcp/prompts",
        json={
            "id": "mine2",
            "name": "m",
            "title": "T",
            "template": "Hi {who}",
            "slots": [{"name": "who"}],
        },
    )
    r = client.put(
        "/api/mcp/prompts/mine2",
        json={"title": "New title"},
    )
    assert r.status_code == 200
    assert r.json()["title"] == "New title"


def test_update_rejects_title_change_on_builtin(client):
    r = client.put(
        "/api/mcp/prompts/brief",
        json={"title": "Renamed"},
    )
    assert r.status_code == 422


def test_update_allows_template_edit_on_builtin(client):
    r = client.put(
        "/api/mcp/prompts/brief",
        json={"template": "Override using {topic}"},
    )
    assert r.status_code == 200
    assert r.json()["template"] == "Override using {topic}"


def test_delete_builtin_rejected(client):
    r = client.delete("/api/mcp/prompts/brief")
    assert r.status_code == 409


def test_delete_user_prompt(client):
    client.post(
        "/api/mcp/prompts",
        json={"id": "temp_p", "name": "t", "title": "T", "template": "hi"},
    )
    r = client.delete("/api/mcp/prompts/temp_p")
    assert r.status_code == 204
    # Gone
    remaining = {p["id"] for p in client.get("/api/mcp/prompts").json()}
    assert "temp_p" not in remaining


def test_toggle_flips_enabled(client):
    r1 = client.post("/api/mcp/prompts/brief/toggle")
    assert r1.status_code == 200
    state_after_first = r1.json()["enabled"]
    r2 = client.post("/api/mcp/prompts/brief/toggle")
    assert r2.json()["enabled"] is not state_after_first


@pytest.mark.anyio
async def test_watch_prompts_file_reloads_on_mtime_change(tmp_path, monkeypatch):
    """Touching the prompts file makes the watcher reregister + (try to) notify."""
    import anyio

    from podcodex.mcp import prompts as p
    from podcodex.mcp.server import mcp

    store = tmp_path / "mcp_prompts.json"
    monkeypatch.setattr(p, "_prompts_path", lambda: store)
    # Seed with built-ins only
    save_prompts(load_prompts())
    # Clear registry so we can observe re-registration
    mcp._prompt_manager._prompts.clear()
    assert "brief" not in mcp._prompt_manager._prompts

    # Run watcher briefly
    async with anyio.create_task_group() as tg:
        tg.start_soon(p._watch_prompts_file, mcp, 0.05)
        await anyio.sleep(0.1)  # establish baseline mtime
        # Add a new prompt on disk
        prompts = load_prompts()
        prompts.append(
            PromptDef(
                id="live_added",
                name="live_added",
                title="Added",
                description="",
                template="Hi {x}",
                slots=[SlotDef(name="x")],
            )
        )
        save_prompts(prompts)
        await anyio.sleep(0.2)  # give watcher a tick
        tg.cancel_scope.cancel()

    assert "live_added" in mcp._prompt_manager._prompts
    assert "brief" in mcp._prompt_manager._prompts  # builtin still present


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_disabled_prompt_is_not_registered(client):
    # Disable brief, confirm it leaves the live registry
    client.post("/api/mcp/prompts/brief/toggle")  # enabled -> disabled
    from podcodex.mcp.server import mcp

    assert "brief" not in mcp._prompt_manager._prompts
    # Re-enable
    client.post("/api/mcp/prompts/brief/toggle")
    assert "brief" in mcp._prompt_manager._prompts
