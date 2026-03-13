"""Tests for podcodex.rag.localstore — all tests use :memory: database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from podcodex.rag.localstore import DEFAULT_DB_PATH, LocalStore


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _store() -> LocalStore:
    return LocalStore(db_path=":memory:")


def _rng_embeddings(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim), dtype=np.float32)


def _chunks(n: int, episode: str = "ep1") -> list[dict]:
    return [
        {
            "text": f"chunk {i}",
            "start": float(i),
            "end": float(i + 1),
            "episode": episode,
        }
        for i in range(n)
    ]


# ──────────────────────────────────────────────
# Schema / construction
# ──────────────────────────────────────────────


def test_tables_exist():
    s = _store()
    tables = {
        r[0]
        for r in s._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"collections", "chunks", "embeddings"}.issubset(tables)


def test_memory_path_accepted():
    s = LocalStore(db_path=":memory:")
    assert s._path == ":memory:"


def test_none_path_uses_default():
    LocalStore.__new__(LocalStore)
    # Just verify DEFAULT_DB_PATH is a Path with parent dirs resolvable
    assert isinstance(DEFAULT_DB_PATH, Path)


def test_podcodex_db_env_var(tmp_path, monkeypatch):
    db_file = tmp_path / "custom.db"
    monkeypatch.setenv("PODCODEX_DB", str(db_file))
    # Re-import to pick up env var at module level
    import importlib
    import podcodex.rag.localstore as mod

    importlib.reload(mod)
    assert str(mod.DEFAULT_DB_PATH) == str(db_file)
    # Restore
    importlib.reload(mod)


def test_dir_created_automatically(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "vectors.db"
    LocalStore(db_path=deep)
    assert deep.parent.exists()
    assert deep.exists()


# ──────────────────────────────────────────────
# Collection management
# ──────────────────────────────────────────────


def test_collection_not_exists_initially():
    s = _store()
    assert not s.collection_exists("my_show__bge-m3")


def test_ensure_collection_creates():
    s = _store()
    s.ensure_collection(
        "my_show__bge-m3", show="My Show", model="bge-m3", chunker="semantic", dim=1024
    )
    assert s.collection_exists("my_show__bge-m3")


def test_ensure_collection_idempotent():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.ensure_collection(
        "c", show="S", model="m", chunker="semantic", dim=8
    )  # no exception
    assert s.list_collections() == ["c"]


def test_list_collections_empty():
    s = _store()
    assert s.list_collections() == []


def test_list_collections_all():
    s = _store()
    s.ensure_collection("b__m", show="B", model="m", chunker="semantic", dim=8)
    s.ensure_collection("a__m", show="A", model="m", chunker="semantic", dim=8)
    assert s.list_collections() == ["a__m", "b__m"]


def test_list_collections_filter_show():
    s = _store()
    s.ensure_collection(
        "a__bge-m3", show="A", model="bge-m3", chunker="semantic", dim=1024
    )
    s.ensure_collection(
        "b__bge-m3", show="B", model="bge-m3", chunker="semantic", dim=1024
    )
    assert s.list_collections(show="A") == ["a__bge-m3"]


def test_list_collections_filter_model():
    s = _store()
    s.ensure_collection(
        "a__bge-m3", show="A", model="bge-m3", chunker="semantic", dim=1024
    )
    s.ensure_collection("a__e5", show="A", model="e5", chunker="semantic", dim=384)
    assert s.list_collections(model="e5") == ["a__e5"]


def test_delete_collection():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    assert s.collection_exists("c")
    s.delete_collection("c")
    assert not s.collection_exists("c")


def test_delete_collection_no_op_if_missing():
    s = _store()
    s.delete_collection("nonexistent")  # should not raise


# ──────────────────────────────────────────────
# episode_is_indexed
# ──────────────────────────────────────────────


def test_episode_not_indexed_before_save():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    assert not s.episode_is_indexed("c", "ep1")


def test_episode_indexed_after_save():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(2), _rng_embeddings(2))
    assert s.episode_is_indexed("c", "ep1")


def test_episode_not_indexed_after_delete():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(2), _rng_embeddings(2))
    s.delete_episode("c", "ep1")
    assert not s.episode_is_indexed("c", "ep1")


# ──────────────────────────────────────────────
# save_chunks
# ──────────────────────────────────────────────


def test_save_chunks_count():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(5), _rng_embeddings(5))
    count = s._conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE collection='c' AND episode='ep1'"
    ).fetchone()[0]
    assert count == 5


def test_save_chunks_text_round_trip():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    chunks = [{"text": "hello world", "start": 0.0}]
    s.save_chunks("c", "ep1", chunks, _rng_embeddings(1))
    loaded = s.load_chunks_no_embeddings("c", "ep1")
    assert loaded[0]["text"] == "hello world"


def test_save_chunks_meta_round_trip():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    chunks = [{"text": "t", "start": 1.5, "end": 3.0, "speaker": "Alice"}]
    s.save_chunks("c", "ep1", chunks, _rng_embeddings(1))
    loaded = s.load_chunks_no_embeddings("c", "ep1")
    assert loaded[0]["start"] == pytest.approx(1.5)
    assert loaded[0]["end"] == pytest.approx(3.0)
    assert loaded[0]["speaker"] == "Alice"


def test_save_chunks_embedding_round_trip():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    emb = _rng_embeddings(1, dim=8)
    s.save_chunks("c", "ep1", _chunks(1), emb)
    loaded = s.load_chunks("c", "ep1")
    assert np.allclose(loaded[0]["embedding"], emb[0], atol=1e-6)


def test_save_chunks_length_mismatch_raises():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    with pytest.raises(ValueError, match="Length mismatch"):
        s.save_chunks("c", "ep1", _chunks(3), _rng_embeddings(2))


# ──────────────────────────────────────────────
# load_chunks
# ──────────────────────────────────────────────


def test_load_chunks_ordering():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    chunks = [{"text": f"chunk {i}", "idx": i} for i in range(5)]
    s.save_chunks("c", "ep1", chunks, _rng_embeddings(5))
    loaded = s.load_chunks("c", "ep1")
    assert [ch["idx"] for ch in loaded] == list(range(5))


def test_load_chunks_embedding_shape():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=16)
    s.save_chunks("c", "ep1", _chunks(3), _rng_embeddings(3, dim=16))
    loaded = s.load_chunks("c", "ep1")
    for ch in loaded:
        assert ch["embedding"].shape == (16,)


def test_load_chunks_empty_episode():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    assert s.load_chunks("c", "ep_missing") == []


# ──────────────────────────────────────────────
# list_episodes
# ──────────────────────────────────────────────


def test_list_episodes_sorted():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep_b", _chunks(1, "ep_b"), _rng_embeddings(1))
    s.save_chunks("c", "ep_a", _chunks(1, "ep_a"), _rng_embeddings(1))
    assert s.list_episodes("c") == ["ep_a", "ep_b"]


def test_list_episodes_multi():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    for ep in ["e1", "e2", "e3"]:
        s.save_chunks("c", ep, _chunks(2, ep), _rng_embeddings(2))
    assert s.list_episodes("c") == ["e1", "e2", "e3"]


def test_list_episodes_empty():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    assert s.list_episodes("c") == []


# ──────────────────────────────────────────────
# delete_episode
# ──────────────────────────────────────────────


def test_delete_episode_removes_only_target():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(2, "ep1"), _rng_embeddings(2))
    s.save_chunks("c", "ep2", _chunks(2, "ep2"), _rng_embeddings(2))
    s.delete_episode("c", "ep1")
    assert not s.episode_is_indexed("c", "ep1")
    assert s.episode_is_indexed("c", "ep2")


def test_delete_episode_no_op_if_missing():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.delete_episode("c", "nonexistent")  # should not raise


# ──────────────────────────────────────────────
# CASCADE: delete collection removes chunks + embeddings
# ──────────────────────────────────────────────


def test_cascade_delete_collection_removes_chunks():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(3), _rng_embeddings(3))
    s.delete_collection("c")
    count = s._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == 0


def test_cascade_delete_collection_removes_embeddings():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(3), _rng_embeddings(3))
    s.delete_collection("c")
    count = s._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert count == 0


# ──────────────────────────────────────────────
# Duplicate safety
# ──────────────────────────────────────────────


def test_duplicate_save_raises_integrity_error():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(2), _rng_embeddings(2))
    with pytest.raises(sqlite3.IntegrityError):
        s.save_chunks("c", "ep1", _chunks(2), _rng_embeddings(2))


def test_delete_then_resave_succeeds():
    s = _store()
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(2), _rng_embeddings(2))
    s.delete_episode("c", "ep1")
    s.save_chunks("c", "ep1", _chunks(3), _rng_embeddings(3))
    assert len(s.load_chunks_no_embeddings("c", "ep1")) == 3
