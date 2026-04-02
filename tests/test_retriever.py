"""Tests for podcodex.rag.retriever — uses in-memory LocalStore."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from podcodex.rag.localstore import LocalStore


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

DIM = 4


def _seed_local(episodes: dict[str, int] | None = None) -> tuple[LocalStore, str]:
    """Create an in-memory LocalStore with seeded data. Returns (store, collection_name)."""
    local = LocalStore(db_path=":memory:")
    col = "test__bge-m3__semantic"
    local.ensure_collection(
        col, show="test", model="bge-m3", chunker="semantic", dim=DIM
    )

    if episodes is None:
        episodes = {"ep1": 3, "ep2": 2}

    for ep, n in episodes.items():
        chunks = [
            {
                "episode": ep,
                "show": "test",
                "start": float(i),
                "end": float(i + 1),
                "speaker": "Alice" if i % 2 == 0 else "Bob",
                "dominant_speaker": "Alice" if i % 2 == 0 else "Bob",
                "source": "polished",
                "text": f"chunk {i} of {ep} about neural networks and podcasting",
            }
            for i in range(n)
        ]
        embeddings = np.random.rand(n, DIM).astype(np.float32)
        local.save_chunks(col, ep, chunks, embeddings)
    return local, col


def _make_retriever(local: LocalStore | None = None, col: str = ""):
    """Return a retriever with mocked embedder (to avoid loading torch)."""
    if local is None:
        local, col = _seed_local()

    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = np.random.rand(DIM).astype(np.float32)

    with patch("podcodex.rag.embedder.get_embedder", return_value=mock_emb):
        from podcodex.rag.retriever import Retriever

        retriever = Retriever(model="bge-m3", local=local)

    return retriever, mock_emb, col


# ──────────────────────────────────────────────
# Constructor
# ──────────────────────────────────────────────


def test_retriever_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        from podcodex.rag.retriever import Retriever

        Retriever(model="bad_model")


def test_retriever_uses_get_embedder():
    mock_emb = MagicMock()
    mock_factory = MagicMock(return_value=mock_emb)
    local = LocalStore(db_path=":memory:")

    with patch("podcodex.rag.embedder.get_embedder", mock_factory):
        from podcodex.rag.retriever import Retriever

        Retriever(model="e5-small", local=local)

    mock_factory.assert_called_once_with("e5-small", device="cpu")


def test_retriever_accepts_local_store():
    local = LocalStore(db_path=":memory:")
    mock_emb = MagicMock()

    with patch("podcodex.rag.embedder.get_embedder", return_value=mock_emb):
        from podcodex.rag.retriever import Retriever

        r = Retriever(model="bge-m3", local=local)

    assert r._local is local


# ──────────────────────────────────────────────
# Dense search (alpha=1.0)
# ──────────────────────────────────────────────


def test_dense_search_returns_results():
    retriever, _, col = _make_retriever()
    results = retriever.retrieve("neural networks", col, top_k=3, alpha=1.0)
    assert len(results) > 0
    assert all("score" in r for r in results)
    assert all("text" in r for r in results)


def test_dense_search_respects_top_k():
    retriever, _, col = _make_retriever()
    results = retriever.retrieve("q", col, top_k=2, alpha=1.0)
    assert len(results) <= 2


def test_dense_search_empty_collection():
    local = LocalStore(db_path=":memory:")
    retriever, _, _ = _make_retriever(local=local)
    results = retriever.retrieve("q", "nonexistent", top_k=5, alpha=1.0)
    assert results == []


# ──────────────────────────────────────────────
# BM25 search (alpha=0.0)
# ──────────────────────────────────────────────


def test_bm25_search_returns_results():
    retriever, _, col = _make_retriever()
    results = retriever.retrieve("neural", col, top_k=5, alpha=0.0)
    assert len(results) > 0


def test_bm25_search_empty_collection():
    local = LocalStore(db_path=":memory:")
    retriever, _, _ = _make_retriever(local=local)
    results = retriever.retrieve("q", "nonexistent", alpha=0.0)
    assert results == []


# ──────────────────────────────────────────────
# Hybrid search (default alpha=0.5)
# ──────────────────────────────────────────────


def test_weighted_search_returns_results():
    retriever, _, col = _make_retriever()
    results = retriever.retrieve("podcasting neural", col, top_k=5, alpha=0.5)
    assert len(results) > 0


def test_weighted_search_blends_scores():
    retriever, _, col = _make_retriever()
    results = retriever.retrieve("neural", col, top_k=5, alpha=0.5)
    # Results should have scores between 0 and ~2 (dense + BM25 combined)
    assert all(r["score"] >= 0 for r in results)


# ──────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────


def test_dense_search_episode_filter():
    retriever, _, col = _make_retriever()
    results = retriever.retrieve("q", col, top_k=10, alpha=1.0, episode="ep1")
    assert all(r.get("episode") == "ep1" for r in results)


def test_dense_search_speaker_filter():
    retriever, _, col = _make_retriever()
    results = retriever.retrieve("q", col, top_k=10, alpha=1.0, speaker="Alice")
    assert all(r.get("dominant_speaker") == "Alice" for r in results)


# ──────────────────────────────────────────────
# Exact search (find)
# ──────────────────────────────────────────────


def test_find_substring_search():
    retriever, _, col = _make_retriever()
    results = retriever.find("neural", col, top_k=25)
    assert len(results) > 0
    assert all("neural" in r["text"].lower() for r in results)
    assert all(r["score"] == 1.0 for r in results)


def test_find_no_match():
    retriever, _, col = _make_retriever()
    results = retriever.find("xyznonexistent", col, top_k=25)
    assert results == []


# ──────────────────────────────────────────────
# Random
# ──────────────────────────────────────────────


def test_random_returns_chunk():
    retriever, _, col = _make_retriever()
    result = retriever.random(col)
    assert result is not None
    assert "text" in result


def test_random_empty_collection():
    local = LocalStore(db_path=":memory:")
    retriever, _, _ = _make_retriever(local=local)
    result = retriever.random("nonexistent")
    assert result is None


# ──────────────────────────────────────────────
# Cache invalidation
# ──────────────────────────────────────────────


def test_invalidate_clears_caches():
    retriever, _, col = _make_retriever()
    # Warm caches
    retriever.retrieve("q", col, top_k=1, alpha=1.0)
    retriever.retrieve("q", col, top_k=1, alpha=0.0)
    assert col in retriever._vector_cache
    assert col in retriever._bm25_cache

    retriever.invalidate(col)
    assert col not in retriever._vector_cache
    assert col not in retriever._bm25_cache


def test_invalidate_all():
    retriever, _, col = _make_retriever()
    retriever.retrieve("q", col, top_k=1, alpha=1.0)
    retriever._bm25_cache["other"] = MagicMock()

    retriever.invalidate()
    assert len(retriever._vector_cache) == 0
    assert len(retriever._bm25_cache) == 0


# ──────────────────────────────────────────────
# _normalize
# ──────────────────────────────────────────────


def test_normalize_empty():
    from podcodex.rag.retriever import _normalize

    assert _normalize([]) == []


def test_normalize_single_result():
    from podcodex.rag.retriever import _normalize

    result = _normalize([{"score": 0.3, "text": "a"}])
    assert result[0]["score"] == pytest.approx(1.0)


def test_normalize_rank_based_scores():
    from podcodex.rag.retriever import _normalize

    results = [{"score": 0.0, "text": "a"}, {"score": 0.0, "text": "b"}]
    normed = _normalize(results)
    assert normed[0]["score"] == pytest.approx(1.0)
    assert normed[1]["score"] == pytest.approx(0.5)
