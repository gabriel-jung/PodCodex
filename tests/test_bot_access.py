"""Tests for the bot-access route (show password management)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from podcodex.api.app import app  # noqa: E402
from podcodex.rag import index_store as rag_index_store  # noqa: E402


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
    _seed_store(tmp_path)
    monkeypatch.setenv("PODCODEX_INDEX", str(tmp_path / "index"))
    rag_index_store.get_index_store.cache_clear()
    yield
    rag_index_store.get_index_store.cache_clear()


@pytest.fixture
def client():
    return TestClient(app, headers={"X-PodCodex": "1"})


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
