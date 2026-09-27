"""Tests for podcodex.rag.indexing — all heavy deps mocked."""

from unittest.mock import MagicMock, patch

import numpy as np


# ──────────────────────────────────────────────
# vectorize_episode: skip / upgrade / overwrite, against a real tmp index
# ──────────────────────────────────────────────


def _vectorize(store, text, source, *, overwrite=False):
    """Index one chunk of episode E1 with a stub embedder; return the count."""
    from podcodex.rag.indexing import vectorize_episode

    embedder = MagicMock()
    embedder.encode_passages.side_effect = lambda chunks: np.zeros(
        (len(chunks), 384), dtype=np.float32
    )
    chunk = {
        "text": text,
        "episode": "E1",
        "show": "S",
        "source": source,
        "start": 0.0,
        "end": 1.0,
    }
    transcript = {"meta": {"show": "S", "episode": "E1", "source": source}}
    with patch("podcodex.rag.indexing.get_embedder", return_value=embedder):
        _, n = vectorize_episode(
            transcript,
            "S",
            "E1",
            "e5-small",
            "semantic",
            store,
            show_id="s_1",
            chunks=[chunk],
            overwrite=overwrite,
        )
    return n


def _stored(store):
    col = store.resolve_collection("s_1", "e5-small", "semantic")
    return [(h.text, h.source) for h in store.load_chunks_no_embeddings(col, "E1")]


def test_an_episode_indexed_from_the_same_source_is_skipped(tmp_path):
    from podcodex.rag.index_store import IndexStore

    store = IndexStore(tmp_path / "index")
    assert _vectorize(store, "first", "transcript") == 1

    assert _vectorize(store, "second", "transcript") == 0
    assert _stored(store) == [("first", "transcript")]


def test_a_better_source_replaces_the_stored_rows(tmp_path):
    """A corrected transcript supersedes the raw one it was indexed from."""
    from podcodex.rag.index_store import IndexStore

    store = IndexStore(tmp_path / "index")
    _vectorize(store, "first", "transcript")

    assert _vectorize(store, "second", "corrected") == 1
    assert _stored(store) == [("second", "corrected")]


def test_overwrite_reindexes_even_the_same_source(tmp_path):
    from podcodex.rag.index_store import IndexStore

    store = IndexStore(tmp_path / "index")
    _vectorize(store, "first", "transcript")

    assert _vectorize(store, "second", "transcript", overwrite=True) == 1
    assert _stored(store) == [("second", "transcript")]


# ──────────────────────────────────────────────
# vectorize_batch: a failed combination is not a success
# ──────────────────────────────────────────────


def test_vectorize_batch_raises_after_writing_the_combinations_that_work():
    import pytest

    from podcodex.rag import indexing

    written: list[str] = []

    def _episode(_t, _show, _ep, model_key, _chunking, _local, **_kw):
        if model_key == "e5-small":
            raise RuntimeError("embedder failed to load")
        written.append(model_key)
        return [{"text": "c"}], 1

    with (
        patch.object(indexing, "vectorize_episode", _episode),
        pytest.raises(indexing.IndexingError, match="embedder failed to load"),
    ):
        indexing.vectorize_batch(
            {"segments": []},
            "Show",
            "ep1",
            ["e5-small", "bge-m3"],
            ["semantic"],
            MagicMock(),
        )
    assert written == ["bge-m3"]


def test_vectorize_batch_counts_an_unexpected_value_error_as_a_failure():
    """Only NoChunksError means "nothing to index"; any other ValueError failed."""
    import pytest

    from podcodex.rag import indexing

    def _episode(*_a, **_kw):
        raise ValueError("embedding dim mismatch")

    with (
        patch.object(indexing, "vectorize_episode", _episode),
        pytest.raises(indexing.IndexingError, match="dim mismatch"),
    ):
        indexing.vectorize_batch(
            {"segments": []}, "Show", "ep1", ["bge-m3"], ["semantic"], MagicMock()
        )
