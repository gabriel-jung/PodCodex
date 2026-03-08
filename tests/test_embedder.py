"""Tests for podcodex.rag.embedder — all model I/O is mocked."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _mock_modules(*names):
    return patch.dict("sys.modules", {name: MagicMock() for name in names})


def _chunks(texts: list[str], episode: str = "E1") -> list[dict]:
    return [{"episode": episode, "text": t} for t in texts]


# ──────────────────────────────────────────────
# PplxEmbedder
# ──────────────────────────────────────────────


def _make_pplx_mocks():
    transformers_mock = MagicMock()
    st_mock = MagicMock()

    def ctx_encode(episodes, **kwargs):
        n = len(episodes[0])
        return [np.zeros((n, 1024), dtype=np.float32)]

    transformers_mock.AutoModel.from_pretrained.return_value.encode = ctx_encode
    st_mock.SentenceTransformer.return_value.encode.return_value = np.zeros(
        1024, dtype=np.float32
    )
    return transformers_mock, st_mock


def test_pplx_encode_passages_shape():
    transformers_mock, st_mock = _make_pplx_mocks()
    with _mock_modules("transformers", "sentence_transformers"):
        sys.modules["transformers"] = transformers_mock
        sys.modules["sentence_transformers"] = st_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.PplxEmbedder()
        result = embedder.encode_passages(_chunks(["Hello", "World", "Foo"]))
    assert result.shape == (3, 1024)
    assert result.dtype == np.float32


def test_pplx_encode_passages_episode_grouping():
    """Chunks from different episodes are encoded separately; output order preserved."""
    transformers_mock, st_mock = _make_pplx_mocks()

    call_log = []

    def ctx_encode(episodes, **kwargs):
        call_log.append(len(episodes[0]))
        n = len(episodes[0])
        return [np.ones((n, 1024), dtype=np.float32) * len(call_log)]

    transformers_mock.AutoModel.from_pretrained.return_value.encode = ctx_encode

    chunks = [
        {"episode": "E1", "text": "a"},
        {"episode": "E2", "text": "b"},
        {"episode": "E1", "text": "c"},
    ]
    with _mock_modules("transformers", "sentence_transformers"):
        sys.modules["transformers"] = transformers_mock
        sys.modules["sentence_transformers"] = st_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.PplxEmbedder()
        result = embedder.encode_passages(chunks)

    assert len(call_log) == 2
    assert sorted(call_log) == [1, 2]
    assert result.shape == (3, 1024)


def test_pplx_encode_query_shape():
    transformers_mock, st_mock = _make_pplx_mocks()
    with _mock_modules("transformers", "sentence_transformers"):
        sys.modules["transformers"] = transformers_mock
        sys.modules["sentence_transformers"] = st_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.PplxEmbedder()
        result = embedder.encode_query("test query")
    assert result.shape == (1024,)
    assert result.dtype == np.float32


# ──────────────────────────────────────────────
# E5Embedder
# ──────────────────────────────────────────────


def _make_e5_mock():
    st_mock = MagicMock()

    def encode(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        return np.zeros((n, 384), dtype=np.float32)

    st_mock.SentenceTransformer.return_value.encode.side_effect = encode
    return st_mock


def test_e5_encode_passages_shape():
    st_mock = _make_e5_mock()
    with _mock_modules("sentence_transformers"):
        sys.modules["sentence_transformers"] = st_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.E5Embedder(model_key="e5-small")
        result = embedder.encode_passages(_chunks(["a", "b"]))
    assert result.shape == (2, 384)
    assert result.dtype == np.float32


def test_e5_encode_passages_uses_passage_prefix():
    st_mock = _make_e5_mock()
    with _mock_modules("sentence_transformers"):
        sys.modules["sentence_transformers"] = st_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.E5Embedder(model_key="e5-small")
        embedder.encode_passages(_chunks(["Hello"]))
    call_args = st_mock.SentenceTransformer.return_value.encode.call_args
    texts_passed = call_args[0][0]
    assert texts_passed[0].startswith("passage: ")


def test_e5_encode_query_uses_query_prefix():
    st_mock = _make_e5_mock()
    with _mock_modules("sentence_transformers"):
        sys.modules["sentence_transformers"] = st_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.E5Embedder(model_key="e5-small")
        embedder.encode_query("my question")
    call_args = st_mock.SentenceTransformer.return_value.encode.call_args
    text_passed = call_args[0][0]
    assert text_passed.startswith("query: ")


def test_e5_encode_query_shape():
    st_mock = _make_e5_mock()
    with _mock_modules("sentence_transformers"):
        sys.modules["sentence_transformers"] = st_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.E5Embedder(model_key="e5-small")
        result = embedder.encode_query("test")
    assert result.shape == (384,)
    assert result.dtype == np.float32


# ──────────────────────────────────────────────
# BGEEmbedder
# ──────────────────────────────────────────────


def _make_bge_mock():
    flag_mock = MagicMock()

    def bge_encode(texts, **kwargs):
        n = len(texts)
        return {"dense_vecs": np.zeros((n, 1024), dtype=np.float32)}

    flag_mock.BGEM3FlagModel.return_value.encode.side_effect = bge_encode
    return flag_mock


def test_bge_encode_passages_shape():
    flag_mock = _make_bge_mock()
    with _mock_modules("FlagEmbedding"):
        sys.modules["FlagEmbedding"] = flag_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.BGEEmbedder()
        result = embedder.encode_passages(_chunks(["a", "b", "c"]))
    assert result.shape == (3, 1024)
    assert result.dtype == np.float32


def test_bge_encode_query_shape():
    flag_mock = _make_bge_mock()
    with _mock_modules("FlagEmbedding"):
        sys.modules["FlagEmbedding"] = flag_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.BGEEmbedder()
        result = embedder.encode_query("test")
    assert result.shape == (1024,)
    assert result.dtype == np.float32


# ──────────────────────────────────────────────
# get_embedder factory
# ──────────────────────────────────────────────


def test_get_embedder_unknown_raises():
    import pytest
    from podcodex.rag.embedder import get_embedder

    with pytest.raises(ValueError, match="Unknown model"):
        get_embedder("bad_model")
