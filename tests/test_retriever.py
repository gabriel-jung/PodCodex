"""Tests for podcodex.rag.retriever — all heavy deps mocked."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _dense_vec(dim: int = 1024) -> np.ndarray:
    return np.random.rand(dim).astype(np.float32)


def _make_scored_point(payload: dict, score: float = 0.9) -> MagicMock:
    pt = MagicMock()
    pt.payload = payload
    pt.score = score
    return pt


def _make_retriever(model: str = "bge-m3"):
    """Return (retriever, mock_embedder, mock_store)."""
    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = _dense_vec()
    mock_store = MagicMock()

    with (
        patch("podcodex.rag.embedder.get_embedder", return_value=mock_emb),
        patch("podcodex.rag.retriever.QdrantStore", return_value=mock_store),
    ):
        from podcodex.rag.retriever import Retriever

        retriever = Retriever(model=model)

    return retriever, mock_emb, mock_store


# ──────────────────────────────────────────────
# Retriever constructor
# ──────────────────────────────────────────────


def test_retriever_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        from podcodex.rag.retriever import Retriever

        Retriever(model="bad_model")


def test_retriever_uses_get_embedder():
    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = _dense_vec()
    mock_store = MagicMock()

    mock_factory = MagicMock(return_value=mock_emb)
    with (
        patch("podcodex.rag.embedder.get_embedder", mock_factory),
        patch("podcodex.rag.retriever.QdrantStore", return_value=mock_store),
    ):
        from podcodex.rag.retriever import Retriever

        Retriever(model="e5-small")

    mock_factory.assert_called_once_with("e5-small", device="cpu")


def test_retriever_accepts_prebuilt_store():
    mock_emb = MagicMock()
    mock_store = MagicMock()

    with patch("podcodex.rag.embedder.get_embedder", return_value=mock_emb):
        from podcodex.rag.retriever import Retriever

        r = Retriever(model="bge-m3", store=mock_store)

    assert r._store is mock_store


# ──────────────────────────────────────────────
# Dense search (alpha=1.0)
# ──────────────────────────────────────────────


def test_dense_search_calls_search_points():
    retriever, _, mock_store = _make_retriever()
    mock_store.search_points.return_value = [
        {"text": "hello", "episode": "E1", "start": 0.0, "score": 0.85}
    ]

    results = retriever.retrieve("my query", "col", top_k=3, alpha=1.0)

    mock_store.search_points.assert_called_once()
    args = mock_store.search_points.call_args
    assert args[0][0] == "col"  # collection
    assert args[0][2] == 3  # top_k
    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(0.85)
    assert results[0]["text"] == "hello"


def test_dense_search_result_has_payload_fields():
    retriever, _, mock_store = _make_retriever()
    mock_store.search_points.return_value = [
        {"episode": "E1", "start": 5.0, "score": 0.77}
    ]

    results = retriever.retrieve("q", "col", alpha=1.0)

    assert results[0]["score"] == pytest.approx(0.77)
    assert results[0]["episode"] == "E1"


# ──────────────────────────────────────────────
# BM25 search (alpha=0.0)
# ──────────────────────────────────────────────


def test_bm25_search_uses_bm25s():
    retriever, _, mock_store = _make_retriever()

    chunk = {"text": "hello podcast world", "episode": "E1", "start": 0.0}
    mock_store.scroll_payloads.return_value = [chunk]
    mock_store.collection_point_count.return_value = 1

    mock_bm25_inst = MagicMock()
    mock_bm25_inst.retrieve.return_value = ([[0]], [[0.8]])

    mock_bm25s = MagicMock()
    mock_bm25s.BM25.return_value = mock_bm25_inst

    with patch("podcodex.rag.retriever.bm25s", mock_bm25s):
        results = retriever.retrieve("podcast", "col", top_k=5, alpha=0.0)

    mock_store.scroll_payloads.assert_called_once()
    assert len(results) == 1
    assert results[0]["text"] == "hello podcast world"


def test_bm25_search_empty_collection():
    retriever, _, mock_store = _make_retriever()
    mock_store.scroll_payloads.return_value = []
    mock_store.collection_point_count.return_value = 0

    with patch("podcodex.rag.retriever.bm25s"):
        results = retriever.retrieve("q", "col", alpha=0.0)

    assert results == []


# ──────────────────────────────────────────────
# Hybrid / alpha-weighted search (default alpha=0.5)
# ──────────────────────────────────────────────


def test_weighted_search_calls_both_dense_and_bm25():
    """alpha=0.5 → calls search_points (dense) and scroll_payloads (BM25)."""
    retriever, _, mock_store = _make_retriever()

    chunk = {"text": "hello", "episode": "E1", "start": 0.0}
    mock_store.search_points.return_value = [{**chunk, "score": 0.6}]
    mock_store.scroll_payloads.return_value = [chunk]
    mock_store.collection_point_count.return_value = 1

    mock_bm25_inst = MagicMock()
    mock_bm25_inst.retrieve.return_value = ([[0]], [[0.5]])

    mock_bm25s = MagicMock()
    mock_bm25s.BM25.return_value = mock_bm25_inst

    with patch("podcodex.rag.retriever.bm25s", mock_bm25s):
        results = retriever.retrieve("hello", "col", top_k=5)

    mock_store.search_points.assert_called()
    mock_store.scroll_payloads.assert_called()
    assert len(results) >= 1


def test_weighted_search_alpha_1_uses_dense_only():
    """alpha=1.0 → only search_points, no scroll."""
    retriever, _, mock_store = _make_retriever()
    mock_store.search_points.return_value = [
        {"text": "hi", "episode": "E1", "start": 0.0, "score": 0.9}
    ]

    results = retriever.retrieve("q", "col", alpha=1.0)

    mock_store.search_points.assert_called_once()
    mock_store.scroll_payloads.assert_not_called()
    assert results[0]["score"] == pytest.approx(0.9)


def test_weighted_search_alpha_0_uses_bm25_only():
    """alpha=0.0 → only scroll, no search_points."""
    retriever, _, mock_store = _make_retriever()
    chunk = {"text": "kw", "episode": "E1", "start": 0.0}
    mock_store.scroll_payloads.return_value = [chunk]
    mock_store.collection_point_count.return_value = 1

    mock_bm25_inst = MagicMock()
    mock_bm25_inst.retrieve.return_value = ([[0]], [[0.5]])

    mock_bm25s = MagicMock()
    mock_bm25s.BM25.return_value = mock_bm25_inst

    with patch("podcodex.rag.retriever.bm25s", mock_bm25s):
        results = retriever.retrieve("q", "col", alpha=0.0)

    mock_store.search_points.assert_not_called()
    assert results[0]["text"] == "kw"


# ──────────────────────────────────────────────
# _result_to_dict
# ──────────────────────────────────────────────


def test_result_to_dict_injects_score():
    from podcodex.rag.store import _result_to_dict

    pt = _make_scored_point({"text": "hi", "start": 1.0}, score=0.99)
    d = _result_to_dict(pt)
    assert d["score"] == pytest.approx(0.99)
    assert d["text"] == "hi"
    assert d["start"] == 1.0


def test_result_to_dict_empty_payload():
    from podcodex.rag.store import _result_to_dict

    pt = MagicMock()
    pt.payload = None
    pt.score = 0.5
    d = _result_to_dict(pt)
    assert d == {"score": 0.5}


# ──────────────────────────────────────────────
# _normalize
# ──────────────────────────────────────────────


def test_normalize_empty():
    from podcodex.rag.retriever import _normalize

    assert _normalize([]) == []


def test_normalize_single_nonzero_result_scores_to_one():
    from podcodex.rag.retriever import _normalize

    result = _normalize([{"score": 0.3, "text": "a"}])
    assert result[0]["score"] == pytest.approx(1.0)


def test_normalize_rank_based_scores():
    """Rank-based normalization: first gets 1.0, last gets 1/n."""
    from podcodex.rag.retriever import _normalize

    results = [{"score": 0.0, "text": "a"}, {"score": 0.0, "text": "b"}]
    normed = _normalize(results)
    # Rank-based: 1.0 - (0/2) = 1.0, 1.0 - (1/2) = 0.5
    assert normed[0]["score"] == pytest.approx(1.0)
    assert normed[1]["score"] == pytest.approx(0.5)


def test_normalize_preserves_order_and_fields():
    from podcodex.rag.retriever import _normalize

    results = [{"score": 0.2, "x": 1}, {"score": 0.8, "x": 2}]
    normed = _normalize(results)
    # Rank-based: position 0 → 1.0, position 1 → 0.5
    assert normed[0]["score"] == pytest.approx(1.0)
    assert normed[1]["score"] == pytest.approx(0.5)
    assert normed[0]["x"] == 1
    assert normed[1]["x"] == 2


# ──────────────────────────────────────────────
# BM25 caching
# ──────────────────────────────────────────────


def _setup_bm25_mocks(mock_store):
    """Configure mock_store for BM25 scroll + collection_point_count."""
    chunk = {"text": "hello podcast world", "episode": "E1", "start": 0.0}
    mock_store.scroll_payloads.return_value = [chunk]
    mock_store.collection_point_count.return_value = 1


def test_bm25_cache_hit_skips_scroll():
    """Two full-collection BM25 queries — scroll should be called only once."""
    retriever, _, mock_store = _make_retriever()
    _setup_bm25_mocks(mock_store)

    mock_bm25_inst = MagicMock()
    mock_bm25_inst.retrieve.return_value = ([[0]], [[0.8]])

    mock_bm25s = MagicMock()
    mock_bm25s.BM25.return_value = mock_bm25_inst

    with patch("podcodex.rag.retriever.bm25s", mock_bm25s):
        retriever.retrieve("podcast", "col", top_k=5, alpha=0.0)
        retriever.retrieve("another query", "col", top_k=5, alpha=0.0)

    # scroll_payloads called once to build cache, not again for second query
    assert mock_store.scroll_payloads.call_count == 1


def test_bm25_cache_miss_on_count_change():
    """When point_count changes, cache is rebuilt."""
    retriever, _, mock_store = _make_retriever()
    _setup_bm25_mocks(mock_store)

    mock_bm25_inst = MagicMock()
    mock_bm25_inst.retrieve.return_value = ([[0]], [[0.8]])

    mock_bm25s = MagicMock()
    mock_bm25s.BM25.return_value = mock_bm25_inst

    with patch("podcodex.rag.retriever.bm25s", mock_bm25s):
        retriever.retrieve("q1", "col", top_k=5, alpha=0.0)

        # Change point count → cache stale
        mock_store.collection_point_count.return_value = 2
        retriever.retrieve("q2", "col", top_k=5, alpha=0.0)

    # scroll_payloads called twice: initial build + rebuild after count change
    assert mock_store.scroll_payloads.call_count == 2


def test_bm25_episode_scoped_bypasses_cache():
    """Episode-filtered BM25 queries should not use or populate the cache."""
    retriever, _, mock_store = _make_retriever()
    _setup_bm25_mocks(mock_store)

    mock_bm25_inst = MagicMock()
    mock_bm25_inst.retrieve.return_value = ([[0]], [[0.8]])

    mock_bm25s = MagicMock()
    mock_bm25s.BM25.return_value = mock_bm25_inst

    with patch("podcodex.rag.retriever.bm25s", mock_bm25s):
        retriever.retrieve("q", "col", top_k=5, alpha=0.0, episode="E1")
        retriever.retrieve("q", "col", top_k=5, alpha=0.0, episode="E1")

    # Each episode-scoped query scrolls independently (no cache)
    assert mock_store.scroll_payloads.call_count == 2
    assert "col" not in retriever._bm25_cache


def test_invalidate_bm25_cache():
    retriever, _, mock_store = _make_retriever()
    _setup_bm25_mocks(mock_store)

    mock_bm25_inst = MagicMock()
    mock_bm25_inst.retrieve.return_value = ([[0]], [[0.8]])

    mock_bm25s = MagicMock()
    mock_bm25s.BM25.return_value = mock_bm25_inst

    with patch("podcodex.rag.retriever.bm25s", mock_bm25s):
        retriever.retrieve("q", "col", top_k=5, alpha=0.0)
        assert "col" in retriever._bm25_cache

        retriever.invalidate_bm25_cache("col")
        assert "col" not in retriever._bm25_cache

    # invalidate_bm25_cache(None) clears all
    retriever._bm25_cache["a"] = MagicMock()
    retriever._bm25_cache["b"] = MagicMock()
    retriever.invalidate_bm25_cache()
    assert len(retriever._bm25_cache) == 0
