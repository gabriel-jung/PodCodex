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


def _patch_store_and_qdrant():
    """
    Return a patch context that replaces QdrantStore with a mock so that
    Retriever.__init__ never actually connects to Qdrant.
    """
    mock_store = MagicMock()
    mock_client = MagicMock()
    mock_store._client = mock_client
    return (
        patch("podcodex.rag.retriever.QdrantStore", return_value=mock_store),
        mock_store,
        mock_client,
    )


def _patch_embedder(class_name: str, encode_query_return):
    """Patch an embedder class so loading it doesn't import ML libs."""
    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = encode_query_return
    return patch(f"podcodex.rag.embedder.{class_name}", return_value=mock_emb), mock_emb


# ──────────────────────────────────────────────
# _load_embedder dispatch
# ──────────────────────────────────────────────


def test_load_embedder_pplx_context():
    from unittest.mock import patch, MagicMock

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
# Dense search
# ──────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["pplx_context", "e5_semantic"])
def test_dense_search_calls_client_search(strategy):
    vec = _dense_query_vec(1024 if strategy == "pplx_context" else 384)
    scored = _make_scored_point({"text": "hello", "episode": "E1"}, score=0.85)

    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = vec

    mock_store = MagicMock()
    mock_client = MagicMock()
    mock_store._client = mock_client
    mock_client.search.return_value = [scored]

    with (
        patch("podcodex.rag.retriever.QdrantStore", return_value=mock_store),
        patch("podcodex.rag.retriever._load_embedder", return_value=mock_emb),
    ):
        from podcodex.rag.retriever import Retriever

        retriever = Retriever(strategy)
        results = retriever.retrieve("my query", "col", top_k=3)

    mock_client.search.assert_called_once_with(
        collection_name="col",
        query_vector=vec.tolist(),
        limit=3,
    )
    assert len(results) == 1
    assert results[0]["score"] == 0.85
    assert results[0]["text"] == "hello"


def test_dense_search_result_has_score_and_payload():
    vec = _dense_query_vec()
    pt = _make_scored_point({"episode": "E1", "start": 0.0}, score=0.77)

    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = vec

    store_patch, mock_store, mock_client = _patch_store_and_qdrant()
    mock_client.search.return_value = [pt]

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
# Hybrid search
# ──────────────────────────────────────────────


def _mock_qdrant_models():
    """Return a sys.modules mock for qdrant_client.models."""
    mock_mod = MagicMock()
    return mock_mod


@pytest.mark.parametrize("strategy", ["bge_speaker", "bge_semantic"])
def test_hybrid_search_calls_query_points(strategy):
    dense_vec = _dense_query_vec()
    sparse_vec = _sparse_query_vec()
    query_emb = {"dense": dense_vec, "sparse": sparse_vec}

    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = query_emb

    scored = _make_scored_point({"text": "hello"}, score=0.6)

    mock_store = MagicMock()
    mock_client = MagicMock()
    mock_store._client = mock_client
    mock_client.query_points.return_value.points = [scored]

    mock_qdrant_models = _mock_qdrant_models()

    with (
        patch("podcodex.rag.retriever.QdrantStore", return_value=mock_store),
        patch("podcodex.rag.retriever._load_embedder", return_value=mock_emb),
        patch.dict("sys.modules", {"qdrant_client.models": mock_qdrant_models}),
    ):
        from podcodex.rag.retriever import Retriever

        retriever = Retriever(strategy)
        results = retriever.retrieve("my query", "col", top_k=5)

    mock_client.query_points.assert_called_once()
    call_kwargs = mock_client.query_points.call_args[1]
    assert call_kwargs["collection_name"] == "col"
    assert call_kwargs["limit"] == 5

    assert len(results) == 1
    assert results[0]["score"] == 0.6
    assert results[0]["text"] == "hello"


def test_hybrid_search_result_has_score():
    dense_vec = _dense_query_vec()
    sparse_vec = _sparse_query_vec()

    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = {"dense": dense_vec, "sparse": sparse_vec}

    scored = _make_scored_point({"episode": "E2"}, score=0.42)

    store_patch, mock_store, mock_client = _patch_store_and_qdrant()
    mock_client.query_points.return_value.points = [scored]

    mock_qdrant_models = _mock_qdrant_models()

    with (
        store_patch,
        patch("podcodex.rag.retriever._load_embedder", return_value=mock_emb),
        patch.dict("sys.modules", {"qdrant_client.models": mock_qdrant_models}),
    ):
        from podcodex.rag.retriever import Retriever

        r = Retriever("bge_semantic")
        r._store = mock_store
        results = r.retrieve("q", "col")

    assert results[0]["score"] == pytest.approx(0.42)
    assert results[0]["episode"] == "E2"


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
