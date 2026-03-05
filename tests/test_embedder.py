"""Tests for podcodex.rag.embedder — all model I/O is mocked."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _mock_modules(*names):
    """Patch sys.modules with MagicMocks for the given module names."""
    return patch.dict("sys.modules", {name: MagicMock() for name in names})


def _chunks(texts: list[str], episode: str = "E1") -> list[dict]:
    return [{"episode": episode, "text": t} for t in texts]


# ──────────────────────────────────────────────
# _to_qdrant_sparse
# ──────────────────────────────────────────────


def test_to_qdrant_sparse_basic():
    from podcodex.rag.embedder import _to_qdrant_sparse

    result = _to_qdrant_sparse({"10": 0.5, "200": 0.3})
    assert set(result["indices"]) == {10, 200}
    assert len(result["values"]) == 2
    assert all(isinstance(v, float) for v in result["values"])


def test_to_qdrant_sparse_empty():
    from podcodex.rag.embedder import _to_qdrant_sparse

    result = _to_qdrant_sparse({})
    assert result == {"indices": [], "values": []}


# ──────────────────────────────────────────────
# PplxEmbedder
# ──────────────────────────────────────────────


def _make_pplx_mocks():
    """Return (transformers_mock, st_mock) with sensible defaults."""
    transformers_mock = MagicMock()
    st_mock = MagicMock()

    # ctx model: encode([[text1, text2]]) → [np.array shape (2, 1024)]
    def ctx_encode(episodes, **kwargs):
        n = len(episodes[0])
        return [np.zeros((n, 1024), dtype=np.float32)]

    transformers_mock.AutoModel.from_pretrained.return_value.encode = ctx_encode
    # query model: encode(text) → (1024,)
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

    # Two episodes → two encode() calls
    assert len(call_log) == 2
    # E1 had 2 chunks, E2 had 1
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
        dim = 384
        return np.zeros((n, dim) if isinstance(texts, list) else dim, dtype=np.float32)

    st_mock.SentenceTransformer.return_value.encode.side_effect = encode
    return st_mock


def test_e5_encode_passages_shape():
    st_mock = _make_e5_mock()
    with _mock_modules("sentence_transformers"):
        sys.modules["sentence_transformers"] = st_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.E5Embedder()
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
        embedder = emb_mod.E5Embedder()
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
        embedder = emb_mod.E5Embedder()
        embedder.encode_query("my question")
    call_args = st_mock.SentenceTransformer.return_value.encode.call_args
    text_passed = call_args[0][0]
    assert text_passed.startswith("query: ")


# ──────────────────────────────────────────────
# BGEEmbedder
# ──────────────────────────────────────────────


def _make_bge_mock():
    flag_mock = MagicMock()

    def bge_encode(texts, **kwargs):
        n = len(texts)
        return {
            "dense_vecs": np.zeros((n, 1024), dtype=np.float32),
            "lexical_weights": [{"10": 0.5, "20": 0.3}] * n,
        }

    flag_mock.BGEM3FlagModel.return_value.encode.side_effect = bge_encode
    return flag_mock


def test_bge_encode_passages_dense_shape():
    flag_mock = _make_bge_mock()
    with _mock_modules("FlagEmbedding"):
        sys.modules["FlagEmbedding"] = flag_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.BGEEmbedder()
        result = embedder.encode_passages(_chunks(["a", "b", "c"]))
    assert result["dense"].shape == (3, 1024)
    assert result["dense"].dtype == np.float32


def test_bge_encode_passages_sparse_format():
    flag_mock = _make_bge_mock()
    with _mock_modules("FlagEmbedding"):
        sys.modules["FlagEmbedding"] = flag_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.BGEEmbedder()
        result = embedder.encode_passages(_chunks(["a", "b"]))
    assert len(result["sparse"]) == 2
    for sv in result["sparse"]:
        assert "indices" in sv and "values" in sv
        assert all(isinstance(i, int) for i in sv["indices"])
        assert all(isinstance(v, float) for v in sv["values"])


def test_bge_encode_query_structure():
    flag_mock = _make_bge_mock()
    with _mock_modules("FlagEmbedding"):
        sys.modules["FlagEmbedding"] = flag_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.BGEEmbedder()
        result = embedder.encode_query("test")
    assert "dense" in result and "sparse" in result
    assert result["dense"].shape == (1024,)
    assert "indices" in result["sparse"] and "values" in result["sparse"]
