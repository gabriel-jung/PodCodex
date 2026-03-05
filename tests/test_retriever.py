"""Tests for podcodex.rag.retriever — all heavy deps mocked."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _dense_query_vec(dim: int = 1024) -> np.ndarray:
    return np.random.rand(dim).astype(np.float32)


def _sparse_query_vec() -> dict:
    return {"indices": [1, 5, 10], "values": [0.8, 0.4, 0.2]}


def _make_scored_point(payload: dict, score: float = 0.9) -> MagicMock:
    pt = MagicMock()
    pt.payload = payload
    pt.score = score
    return pt


def _query_points_return(points: list) -> MagicMock:
    """Build the mock return value for client.query_points()."""
    rv = MagicMock()
    rv.points = points
    return rv


def _patch_store_and_qdrant():
    mock_store = MagicMock()
    mock_client = MagicMock()
    mock_store._client = mock_client
    return (
        patch("podcodex.rag.retriever.QdrantStore", return_value=mock_store),
        mock_store,
        mock_client,
    )


# ──────────────────────────────────────────────
# _load_embedder dispatch
# ──────────────────────────────────────────────


def test_load_embedder_pplx_context():
    mock_cls = MagicMock()
    with patch("podcodex.rag.embedder.PplxEmbedder", mock_cls):
        from importlib import reload
        import podcodex.rag.retriever as ret_mod

        reload(ret_mod)
        with patch("podcodex.rag.embedder.PplxEmbedder", mock_cls):
            ret_mod._load_embedder("pplx_context")
    mock_cls.assert_called_once()


def test_load_embedder_e5_semantic():
    mock_cls = MagicMock()
    with patch("podcodex.rag.embedder.E5Embedder", mock_cls):
        from podcodex.rag.retriever import _load_embedder

        _load_embedder("e5_semantic")
    mock_cls.assert_called_once()


def test_load_embedder_bge_speaker():
    mock_cls = MagicMock()
    with patch("podcodex.rag.embedder.BGEEmbedder", mock_cls):
        from podcodex.rag.retriever import _load_embedder

        _load_embedder("bge_speaker")
    mock_cls.assert_called_once()


def test_load_embedder_bge_semantic():
    mock_cls = MagicMock()
    with patch("podcodex.rag.embedder.BGEEmbedder", mock_cls):
        from podcodex.rag.retriever import _load_embedder

        _load_embedder("bge_semantic")
    mock_cls.assert_called_once()


def test_load_embedder_unknown_raises():
    from podcodex.rag.retriever import _load_embedder

    with pytest.raises(ValueError, match="Unknown strategy"):
        _load_embedder("bad_strategy")


# ──────────────────────────────────────────────
# Retriever constructor
# ──────────────────────────────────────────────


def test_retriever_unknown_strategy_raises():
    store_patch, _, _ = _patch_store_and_qdrant()
    with store_patch:
        with patch("podcodex.rag.retriever._load_embedder"):
            from podcodex.rag.retriever import Retriever

            with pytest.raises(ValueError, match="Unknown strategy"):
                Retriever("bad_strategy")


# ──────────────────────────────────────────────
# Dense search (dense-only strategies)
# ──────────────────────────────────────────────


@pytest.mark.parametrize("strategy,dim", [("pplx_context", 1024), ("e5_semantic", 384)])
def test_dense_search_calls_query_points(strategy, dim):
    vec = _dense_query_vec(dim)
    scored = _make_scored_point(
        {"text": "hello", "episode": "E1", "start": 0.0}, score=0.85
    )

    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = vec  # plain ndarray for dense-only

    mock_store = MagicMock()
    mock_client = MagicMock()
    mock_store._client = mock_client
    mock_client.query_points.return_value = _query_points_return([scored])

    with (
        patch("podcodex.rag.retriever.QdrantStore", return_value=mock_store),
        patch("podcodex.rag.retriever._load_embedder", return_value=mock_emb),
    ):
        from podcodex.rag.retriever import Retriever

        retriever = Retriever(strategy)
        results = retriever.retrieve("my query", "col", top_k=3)

    mock_client.query_points.assert_called_once_with(
        collection_name="col",
        query=vec.tolist(),
        using=None,
        limit=3,
    )
    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(0.85)
    assert results[0]["text"] == "hello"


def test_dense_search_result_has_payload_fields():
    vec = _dense_query_vec()
    pt = _make_scored_point({"episode": "E1", "start": 5.0}, score=0.77)

    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = vec

    store_patch, mock_store, mock_client = _patch_store_and_qdrant()
    mock_client.query_points.return_value = _query_points_return([pt])

    with (
        store_patch,
        patch("podcodex.rag.retriever._load_embedder", return_value=mock_emb),
    ):
        from podcodex.rag.retriever import Retriever

        r = Retriever("pplx_context")
        r._store = mock_store
        results = r.retrieve("q", "col")

    assert results[0]["score"] == pytest.approx(0.77)
    assert results[0]["episode"] == "E1"


# ──────────────────────────────────────────────
# Hybrid / alpha-weighted search (BGE strategies)
# ──────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["bge_speaker", "bge_semantic"])
def test_weighted_search_calls_query_points_twice(strategy):
    """alpha=0.5 (default) → one dense call + one BM25 call."""
    dense_vec = _dense_query_vec()
    sparse_vec = _sparse_query_vec()
    query_emb = {"dense": dense_vec, "sparse": sparse_vec}

    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = query_emb

    chunk = {"text": "hello", "episode": "E1", "start": 0.0}
    scored = _make_scored_point(chunk, score=0.6)

    mock_store = MagicMock()
    mock_client = MagicMock()
    mock_store._client = mock_client
    # Both dense and BM25 calls return the same point
    mock_client.query_points.return_value = _query_points_return([scored])

    mock_qdrant_models = MagicMock()

    with (
        patch("podcodex.rag.retriever.QdrantStore", return_value=mock_store),
        patch("podcodex.rag.retriever._load_embedder", return_value=mock_emb),
        patch.dict("sys.modules", {"qdrant_client.models": mock_qdrant_models}),
    ):
        from podcodex.rag.retriever import Retriever

        retriever = Retriever(strategy)
        results = retriever.retrieve("my query", "col", top_k=5)

    assert mock_client.query_points.call_count == 2
    calls = mock_client.query_points.call_args_list
    # First call: dense (using="dense")
    assert calls[0].kwargs["using"] == "dense"
    assert calls[0].kwargs["collection_name"] == "col"
    # Second call: sparse (using="sparse")
    assert calls[1].kwargs["using"] == "sparse"
    assert calls[1].kwargs["collection_name"] == "col"

    assert len(results) == 1
    assert results[0]["text"] == "hello"


def test_weighted_search_alpha_1_uses_dense_only():
    """alpha=1.0 → pure dense, only one query_points call."""
    vec = _dense_query_vec()
    query_emb = {"dense": vec, "sparse": _sparse_query_vec()}

    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = query_emb

    scored = _make_scored_point(
        {"text": "hi", "episode": "E1", "start": 0.0}, score=0.9
    )
    mock_store = MagicMock()
    mock_client = MagicMock()
    mock_store._client = mock_client
    mock_client.query_points.return_value = _query_points_return([scored])

    with (
        patch("podcodex.rag.retriever.QdrantStore", return_value=mock_store),
        patch("podcodex.rag.retriever._load_embedder", return_value=mock_emb),
    ):
        from podcodex.rag.retriever import Retriever

        r = Retriever("bge_speaker")
        r._store = mock_store
        results = r.retrieve("q", "col", alpha=1.0)

    mock_client.query_points.assert_called_once()
    assert results[0]["score"] == pytest.approx(0.9)


def test_weighted_search_alpha_0_uses_bm25_only():
    """alpha=0.0 → pure BM25, sparse vector used."""
    query_emb = {"dense": _dense_query_vec(), "sparse": _sparse_query_vec()}

    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = query_emb

    scored = _make_scored_point(
        {"text": "kw", "episode": "E1", "start": 0.0}, score=0.5
    )
    mock_store = MagicMock()
    mock_client = MagicMock()
    mock_store._client = mock_client
    mock_client.query_points.return_value = _query_points_return([scored])

    mock_qdrant_models = MagicMock()
    with (
        patch("podcodex.rag.retriever.QdrantStore", return_value=mock_store),
        patch("podcodex.rag.retriever._load_embedder", return_value=mock_emb),
        patch.dict("sys.modules", {"qdrant_client.models": mock_qdrant_models}),
    ):
        from podcodex.rag.retriever import Retriever

        r = Retriever("bge_speaker")
        r._store = mock_store
        results = r.retrieve("q", "col", alpha=0.0)

    mock_client.query_points.assert_called_once()
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["using"] == "sparse"
    assert results[0]["text"] == "kw"


# ──────────────────────────────────────────────
# _result_to_dict
# ──────────────────────────────────────────────


def test_result_to_dict_injects_score():
    from podcodex.rag.retriever import _result_to_dict

    pt = _make_scored_point({"text": "hi", "start": 1.0}, score=0.99)
    d = _result_to_dict(pt)
    assert d["score"] == pytest.approx(0.99)
    assert d["text"] == "hi"
    assert d["start"] == 1.0


def test_result_to_dict_empty_payload():
    from podcodex.rag.retriever import _result_to_dict

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


def test_normalize_single_result_scores_to_one():
    from podcodex.rag.retriever import _normalize

    result = _normalize([{"score": 0.3, "text": "a"}])
    assert result[0]["score"] == pytest.approx(1.0)


def test_normalize_preserves_order_and_fields():
    from podcodex.rag.retriever import _normalize

    results = [{"score": 0.2, "x": 1}, {"score": 0.8, "x": 2}]
    normed = _normalize(results)
    assert normed[0]["score"] == pytest.approx(0.0)
    assert normed[1]["score"] == pytest.approx(1.0)
    assert normed[0]["x"] == 1
    assert normed[1]["x"] == 2
