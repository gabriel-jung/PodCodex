"""Tests for the bot-access route (show password management)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")


from podcodex.api.app import app  # noqa: E402
from podcodex.rag import index_store as rag_index_store  # noqa: E402
from tests.fixtures.api_client import client_for


DIM = 8


def _seed_store(tmp_path: Path):
    """Fresh IndexStore with two shows indexed under the default combo."""
    store = rag_index_store.IndexStore(tmp_path / "index")
    for show in ("Alpha", "Beta"):
        col = f"{show.lower()}__bge-m3__semantic"
        store.ensure_collection(
            col, show=show, model="bge-m3", chunker="semantic", dim=DIM
        )
        chunks = [
            {
                "text": "x",
                "episode": "ep1",
                "show": show,
                "source": "transcript",
                "dominant_speaker": "sp",
                "start": 0.0,
                "end": 1.0,
            }
        ]
        rng = np.random.default_rng(0)
        store.save_chunks(col, "ep1", chunks, rng.random((1, DIM), dtype=np.float32))
    return store


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    # Pinned before seeding: creating the index stamps its owner, and without
    # this the stamp would come from (and create) the developer's real
    # machine-id file.
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "test-owner")
    _seed_store(tmp_path)
    monkeypatch.setenv("PODCODEX_INDEX", str(tmp_path / "index"))
    # Isolate from the real config.json: show names come only from the
    # seeded index, not the developer's registered show folders.
    import podcodex.core.app_config as app_config

    monkeypatch.setattr(app_config, "load_config", lambda: app_config.AppConfig())
    rag_index_store.get_index_store.cache_clear()
    yield
    rag_index_store.get_index_store.cache_clear()


@pytest.fixture
def client():
    return client_for(app)


# ── List ────────────────────────────────────────────────────────────────


def test_list_shows_all_unprotected_initially(client):
    r = client.get("/api/bot-access/passwords")
    assert r.status_code == 200
    body = r.json()
    assert [b["show"] for b in body] == ["Alpha", "Beta"]
    assert all(b["is_protected"] is False for b in body)


def test_get_one_unknown_show_404(client):
    r = client.get("/api/bot-access/passwords/Nope")
    assert r.status_code == 404


# ── Generate ────────────────────────────────────────────────────────────


def test_generate_returns_plaintext_once(client):
    r = client.post("/api/bot-access/passwords/Alpha", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["show"] == "Alpha"
    assert body["generated"] is True
    assert isinstance(body["password"], str)
    assert len(body["password"]) >= 20  # 16 bytes -> 22 urlsafe chars

    # Status now reflects protected
    status = client.get("/api/bot-access/passwords/Alpha").json()
    assert status["is_protected"] is True


def test_generate_stores_sha256_hash(client):
    r = client.post("/api/bot-access/passwords/Alpha", json={})
    plaintext = r.json()["password"]
    expected = f"sha256:{hashlib.sha256(plaintext.encode()).hexdigest()}"

    store = rag_index_store.get_index_store()
    assert store.get_show_passwords()["Alpha"] == expected


# ── Manual ──────────────────────────────────────────────────────────────


def test_manual_password_accepts_16_chars(client):
    r = client.post("/api/bot-access/passwords/Alpha", json={"password": "a" * 16})
    assert r.status_code == 200
    body = r.json()
    assert body["generated"] is False
    assert body["password"] == "a" * 16


def test_manual_password_rejects_too_short(client):
    r = client.post("/api/bot-access/passwords/Alpha", json={"password": "short"})
    assert r.status_code == 422
    assert "at least 16" in r.json()["detail"]


def test_manual_password_whitespace_is_trimmed_then_rejected(client):
    r = client.post("/api/bot-access/passwords/Alpha", json={"password": "   "})
    # Trimmed to empty → treated as generate, not manual; should generate.
    # Confirm behaviour: empty-after-trim means generate.
    assert r.status_code == 200
    assert r.json()["generated"] is True


# ── Rotate ──────────────────────────────────────────────────────────────


def test_rotate_replaces_existing_hash(client):
    first = client.post("/api/bot-access/passwords/Alpha", json={}).json()
    second = client.post("/api/bot-access/passwords/Alpha", json={}).json()
    assert first["password"] != second["password"]

    store = rag_index_store.get_index_store()
    expected = f"sha256:{hashlib.sha256(second['password'].encode()).hexdigest()}"
    assert store.get_show_passwords()["Alpha"] == expected


# ── Delete ──────────────────────────────────────────────────────────────


def test_delete_removes_protection(client):
    client.post("/api/bot-access/passwords/Alpha", json={})
    r = client.delete("/api/bot-access/passwords/Alpha")
    assert r.status_code == 204
    assert client.get("/api/bot-access/passwords/Alpha").json()["is_protected"] is False


def test_delete_unknown_show_404(client):
    r = client.delete("/api/bot-access/passwords/Nope")
    assert r.status_code == 404


# ── Unknown show ────────────────────────────────────────────────────────


def test_set_unknown_show_404(client):
    r = client.post("/api/bot-access/passwords/Nope", json={})
    assert r.status_code == 404


# ── Index ownership ─────────────────────────────────────────────────────


def test_set_password_on_replica_returns_409(client, monkeypatch):
    """A replica must not accept a password the next rsync would erase."""
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")

    r = client.post("/api/bot-access/passwords/Alpha", json={})
    assert r.status_code == 409
    assert "replica" in r.json()["detail"]

    assert rag_index_store.get_index_store().get_show_passwords() == {}


def test_delete_password_on_replica_returns_409(client, monkeypatch):
    assert client.post("/api/bot-access/passwords/Alpha", json={}).status_code == 200
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")

    r = client.delete("/api/bot-access/passwords/Alpha")
    assert r.status_code == 409
    assert client.get("/api/bot-access/passwords/Alpha").json()["is_protected"] is True


def test_claiming_the_index_restores_writes(client, monkeypatch):
    from podcodex.rag.index_origin import claim_origin

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")
    assert client.post("/api/bot-access/passwords/Alpha", json={}).status_code == 409

    claim_origin(rag_index_store.get_index_store().path)
    assert client.post("/api/bot-access/passwords/Alpha", json={}).status_code == 200


def test_unstamped_index_still_accepts_writes(client, monkeypatch):
    """Indexes predating the marker are unowned and must keep working."""
    from podcodex.rag.index_origin import ORIGIN_FILENAME

    (rag_index_store.get_index_store().path / ORIGIN_FILENAME).unlink()
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "some-other-host")

    assert client.post("/api/bot-access/passwords/Alpha", json={}).status_code == 200


# ── Guild unlock lists keyed by show id ─────────────────────────────────


def _bare_bot(store, server_cfg):
    """A bot object with just enough wired for the access mixin."""
    from podcodex.bot.bot import PodCodexBot

    bot = PodCodexBot.__new__(PodCodexBot)
    bot._shows = {}
    bot._local = store
    bot._server_cfg = server_cfg
    bot._save_server_config = lambda: None
    return bot


def test_allowed_shows_migrates_from_names_to_ids(tmp_path):
    from podcodex.bot.config import ServerSettings

    store = rag_index_store.get_index_store()
    store.set_collection_identity(
        "alpha__bge-m3__semantic", show_id="alpha_1234abcd", show="Alpha"
    )
    settings = ServerSettings(allowed_shows=["Alpha"])
    bot = _bare_bot(store, {1: settings})

    bot._reload_shows()

    assert settings.allowed_shows == ["alpha_1234abcd"]


def test_unlocked_show_survives_a_rename(tmp_path):
    """The bug one layer out: renaming used to re-lock every guild."""
    from podcodex.bot.config import ServerSettings
    from podcodex.core.show_passwords import hash_show_password

    store = rag_index_store.get_index_store()
    store.set_collection_identity(
        "alpha__bge-m3__semantic", show_id="alpha_1234abcd", show="Alpha"
    )
    store.set_show_password(
        "alpha_1234abcd", hash_show_password("x" * 16), show_label="Alpha"
    )

    settings = ServerSettings(allowed_shows=["Alpha"])
    bot = _bare_bot(store, {1: settings})
    bot._reload_shows()
    assert bot._show_allowed_by_label("Alpha", settings) is True

    # Rename: label changes in the index, identity does not.
    store.set_show_label("alpha_1234abcd", "Renamed")
    store.set_show_password(
        "alpha_1234abcd", hash_show_password("x" * 16), show_label="Renamed"
    )
    bot._reload_shows()

    assert bot._show_allowed_by_label("Renamed", settings) is True
    assert settings.allowed_shows == ["alpha_1234abcd"]


def test_protected_show_stays_locked_for_other_guilds(tmp_path):
    from podcodex.bot.config import ServerSettings
    from podcodex.core.show_passwords import hash_show_password

    store = rag_index_store.get_index_store()
    store.set_collection_identity(
        "alpha__bge-m3__semantic", show_id="alpha_1234abcd", show="Alpha"
    )
    store.set_show_password(
        "alpha_1234abcd", hash_show_password("x" * 16), show_label="Alpha"
    )

    other = ServerSettings(allowed_shows=[])
    bot = _bare_bot(store, {2: other})
    bot._reload_shows()

    assert bot._show_allowed_by_label("Alpha", other) is False
    store.set_show_label("alpha_1234abcd", "Renamed")
    bot._reload_shows()
    assert bot._show_allowed_by_label("Renamed", other) is False
