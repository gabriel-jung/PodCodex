"""Tests for podcodex.rag.index_store — every test uses a fresh tmp_path index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from podcodex.rag.index_store import IndexStore


def _store(tmp_path: Path) -> IndexStore:
    return IndexStore(tmp_path / "index")


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
            "show": "S",
            "source": "transcript",
            "dominant_speaker": f"sp{i % 2}",
        }
        for i in range(n)
    ]


# ── Collection management ────────────────────────────────────────────────


def test_collection_not_exists_initially(tmp_path):
    s = _store(tmp_path)
    assert not s.collection_exists("my_show__bge-m3__semantic")


def test_ensure_collection_creates(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection(
        "my_show__bge-m3__semantic",
        show="My Show",
        model="bge-m3",
        chunker="semantic",
        dim=8,
    )
    assert s.collection_exists("my_show__bge-m3__semantic")


def test_ensure_collection_idempotent(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    assert s.list_collections() == ["c"]


def test_list_collections_filter_show(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("a", show="A", model="m", chunker="semantic", dim=8)
    s.ensure_collection("b", show="B", model="m", chunker="semantic", dim=8)
    assert s.list_collections(show="A") == ["a"]


def test_list_collections_filter_model(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("a__bge", show="A", model="bge-m3", chunker="semantic", dim=8)
    s.ensure_collection("a__e5", show="A", model="e5", chunker="semantic", dim=8)
    assert s.list_collections(model="e5") == ["a__e5"]


def test_get_collection_info(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="bge-m3", chunker="semantic", dim=1024)
    assert s.get_collection_info("c") == {
        "show": "S",
        "model": "bge-m3",
        "chunker": "semantic",
        "dim": 1024,
    }


def test_get_collection_info_missing(tmp_path):
    s = _store(tmp_path)
    assert s.get_collection_info("nope") is None


def test_delete_collection(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.delete_collection("c")
    assert not s.collection_exists("c")


def test_delete_collection_no_op_if_missing(tmp_path):
    _store(tmp_path).delete_collection("nope")  # should not raise


# ── Episode-level ────────────────────────────────────────────────────────


def test_episode_indexed_after_save(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(2), _rng_embeddings(2))
    assert s.episode_is_indexed("c", "ep1")
    assert s.episode_chunk_count("c", "ep1") == 2


def test_delete_episode_removes_only_target(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(2, "ep1"), _rng_embeddings(2))
    s.save_chunks("c", "ep2", _chunks(2, "ep2"), _rng_embeddings(2))
    s.delete_episode("c", "ep1")
    assert not s.episode_is_indexed("c", "ep1")
    assert s.episode_is_indexed("c", "ep2")


def test_list_episodes_sorted(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep_b", _chunks(1, "ep_b"), _rng_embeddings(1))
    s.save_chunks("c", "ep_a", _chunks(1, "ep_a"), _rng_embeddings(1))
    assert s.list_episodes("c") == ["ep_a", "ep_b"]


# ── save_chunks ──────────────────────────────────────────────────────────


def test_save_chunks_text_round_trip(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks(
        "c", "ep1", [{"text": "hello world", "start": 0.0}], _rng_embeddings(1)
    )
    loaded = s.load_chunks_no_embeddings("c", "ep1")
    assert loaded[0]["text"] == "hello world"


def test_save_chunks_meta_round_trip(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    chunk = {
        "text": "t",
        "start": 1.5,
        "end": 3.0,
        "speakers": [{"speaker": "Alice", "text": "hi"}],
    }
    s.save_chunks("c", "ep1", [chunk], _rng_embeddings(1))
    loaded = s.load_chunks_no_embeddings("c", "ep1")
    assert loaded[0]["start"] == pytest.approx(1.5)
    assert loaded[0]["end"] == pytest.approx(3.0)
    assert loaded[0]["speakers"] == [{"speaker": "Alice", "text": "hi"}]


def test_save_chunks_length_mismatch_raises(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    with pytest.raises(ValueError, match="Length mismatch"):
        s.save_chunks("c", "ep1", _chunks(3), _rng_embeddings(2))


# ── Native search ────────────────────────────────────────────────────────


def test_search_vector_returns_self_first(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=4)
    vecs = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    chunks = [
        {"text": "first", "episode": "e", "start": 0.0, "end": 1.0},
        {"text": "second", "episode": "e", "start": 1.0, "end": 2.0},
        {"text": "third", "episode": "e", "start": 2.0, "end": 3.0},
    ]
    s.save_chunks("c", "e", chunks, vecs)
    hits = s.search_vector("c", vecs[0], top_k=2)
    assert hits[0]["text"] == "first"


def test_search_vector_episode_filter(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=4)
    vecs = _rng_embeddings(2, dim=4)
    s.save_chunks(
        "c",
        "ep1",
        [{"text": "alpha", "episode": "ep1", "start": 0.0, "end": 1.0}],
        vecs[:1],
    )
    s.save_chunks(
        "c",
        "ep2",
        [{"text": "beta", "episode": "ep2", "start": 0.0, "end": 1.0}],
        vecs[1:],
    )
    hits = s.search_vector("c", vecs[0], top_k=5, episode="ep2")
    assert [h["text"] for h in hits] == ["beta"]


def test_search_fts_finds_token(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=4)
    chunks = [
        {"text": "the quick brown fox", "episode": "e", "start": 0.0, "end": 1.0},
        {"text": "lazy dogs sleep all day", "episode": "e", "start": 1.0, "end": 2.0},
    ]
    s.save_chunks("c", "e", chunks, _rng_embeddings(2, dim=4))
    hits = s.search_fts("c", "fox", top_k=5)
    assert any("fox" in h["text"] for h in hits)


# ── Stats helpers ────────────────────────────────────────────────────────


def test_collection_chunk_count(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "e1", _chunks(3, "e1"), _rng_embeddings(3))
    s.save_chunks("c", "e2", _chunks(2, "e2"), _rng_embeddings(2))
    assert s.collection_chunk_count("c") == 5


def test_list_sources_and_speakers(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "e1", _chunks(4, "e1"), _rng_embeddings(4))
    assert s.list_sources("c") == ["transcript"]
    assert s.list_speakers("c") == ["sp0", "sp1"]


def test_get_episode_stats(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "e1", _chunks(3, "e1"), _rng_embeddings(3))
    stats = s.get_episode_stats("c")
    assert len(stats) == 1
    assert stats[0]["episode"] == "e1"
    assert stats[0]["chunk_count"] == 3
    assert stats[0]["duration"] == pytest.approx(3.0)
    assert set(stats[0]["speakers"]) == {"sp0", "sp1"}


# ── Persistence ──────────────────────────────────────────────────────────


def test_reopening_preserves_data(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "e1", _chunks(2, "e1"), _rng_embeddings(2))
    s2 = IndexStore(tmp_path / "index")
    assert s2.collection_exists("c")
    assert s2.list_episodes("c") == ["e1"]
    assert s2.collection_chunk_count("c") == 2
