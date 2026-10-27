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

    ctx_model = transformers_mock.AutoModel.from_pretrained.return_value
    ctx_model.encode = ctx_encode
    ctx_model.to.return_value = ctx_model  # .to(device) returns same mock
    st_mock.SentenceTransformer.return_value.encode.return_value = np.zeros(
        1024, dtype=np.float32
    )
    return transformers_mock, st_mock


def test_pplx_encode_passages_episode_grouping():
    """Chunks from different episodes are encoded separately; output order preserved."""
    transformers_mock, st_mock = _make_pplx_mocks()

    call_log = []

    def ctx_encode(episodes, **kwargs):
        call_log.append(len(episodes[0]))
        n = len(episodes[0])
        return [np.ones((n, 1024), dtype=np.float32) * len(call_log)]

    ctx_model = transformers_mock.AutoModel.from_pretrained.return_value
    ctx_model.encode = ctx_encode
    ctx_model.to.return_value = ctx_model  # .to(device) returns same mock

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

    # One forward pass per episode, E1's two chunks together...
    assert call_log == [2, 1]
    # ...and each row lands back at its chunk's position (the mock fills a
    # pass's rows with the pass number).
    assert result.dtype == np.float32
    assert [row[0] for row in result] == [1.0, 2.0, 1.0]


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


def test_pplx_loads_each_model_only_when_used():
    """A search encodes queries only, an index job passages only."""
    transformers_mock, st_mock = _make_pplx_mocks()
    with _mock_modules("transformers", "sentence_transformers"):
        sys.modules["transformers"] = transformers_mock
        sys.modules["sentence_transformers"] = st_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.PplxEmbedder()
        st_mock.SentenceTransformer.assert_not_called()  # nothing at init
        embedder.encode_query("q")
        transformers_mock.AutoModel.from_pretrained.assert_not_called()
        embedder.encode_passages([{"episode": "E", "text": "a"}])
    st_mock.SentenceTransformer.assert_called_once()  # indexing loaded no more


def test_bge_precision_follows_the_device_facility(monkeypatch):
    """FP16 only on a CUDA device the facility gives a half dtype; CPU and
    Pascal (float32) run in full precision."""
    import torch

    import podcodex.core.device as device
    from podcodex.rag.embedder import _half_precision

    assert _half_precision("cpu") is False
    monkeypatch.setattr(device, "torch_dtype", lambda: torch.float32)
    assert _half_precision("cuda") is False
    monkeypatch.setattr(device, "torch_dtype", lambda: torch.float16)
    assert _half_precision("cuda") is True


def test_every_remote_code_model_is_pinned_to_a_commit():
    """Perplexity models run repo code (trust_remote_code): an unpinned load
    executes whatever the repo holds on the day it downloads."""
    import re

    from podcodex.rag.defaults import MODELS

    sha = re.compile(r"^[0-9a-f]{40}$")
    for key, spec in MODELS.items():
        if key.startswith("pplx"):
            assert spec.revision and sha.match(spec.revision), key
            assert spec.query_revision and sha.match(spec.query_revision), key


def test_pplx_loaders_pass_the_pinned_revisions():
    transformers_mock, st_mock = _make_pplx_mocks()
    with _mock_modules("transformers", "sentence_transformers"):
        sys.modules["transformers"] = transformers_mock
        sys.modules["sentence_transformers"] = st_mock
        from importlib import reload
        import podcodex.rag.embedder as emb_mod

        reload(emb_mod)
        embedder = emb_mod.PplxEmbedder()
        embedder._ctx_model  # noqa: B018 - force the lazy loads
        embedder._query_model  # noqa: B018
    spec = emb_mod.MODELS["pplx-0.6B"]
    assert (
        st_mock.SentenceTransformer.call_args.kwargs["revision"] == spec.query_revision
    )
    assert (
        transformers_mock.AutoModel.from_pretrained.call_args.kwargs["revision"]
        == spec.revision
    )


def test_the_chunker_reuses_the_e5_embedders_model(monkeypatch):
    """One copy of E5 Small in memory when it both chunks and indexes, and
    clearing the embedder cache frees the chunker's wrapper too."""
    from unittest.mock import MagicMock

    import podcodex.rag.embedder as emb

    e5 = MagicMock()
    monkeypatch.setattr(emb, "get_embedder", lambda key, device="cpu": e5)
    chonkie = MagicMock()
    with _mock_modules("chonkie"):
        sys.modules["chonkie"] = chonkie
        first = emb.get_chunker_embeddings("cpu")
        again = emb.get_chunker_embeddings("cpu")
    assert first is again
    chonkie.SentenceTransformerEmbeddings.assert_called_once_with(e5._model)
    emb.clear_embedder_cache()
    assert "chunker|cpu" not in emb._embedder_cache
