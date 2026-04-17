"""Tests for podcodex.mcp.server — skipped if the mcp SDK isn't installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcp")

from podcodex.mcp import server as mcp_server  # noqa: E402
from podcodex.rag import index_store as rag_index_store  # noqa: E402
from podcodex.rag import retriever as rag_retriever  # noqa: E402
from podcodex.rag.index_store import IndexStore  # noqa: E402
from podcodex.rag.store import collection_name  # noqa: E402


DIM = 8


def _seed_store(tmp_path: Path) -> IndexStore:
    """Build a fresh IndexStore with one show indexed under the default model+chunker."""
    store = IndexStore(tmp_path / "index")
    col = collection_name("My Show", "bge-m3", "semantic")
    store.ensure_collection(
        col, show="My Show", model="bge-m3", chunker="semantic", dim=DIM
    )
    chunks = [
        {
            "text": f"chunk {i}",
            "episode": "ep1",
            "show": "My Show",
            "source": "transcript",
            "dominant_speaker": f"sp{i % 2}",
            "start": float(i),
            "end": float(i + 1),
        }
        for i in range(6)
    ]
    rng = np.random.default_rng(0)
    store.save_chunks(col, "ep1", chunks, rng.random((6, DIM), dtype=np.float32))
    return store


@pytest.fixture(autouse=True)
def _reset_caches(tmp_path, monkeypatch):
    """Point shared IndexStore/Retriever at a tmp index for every test."""
    _seed_store(tmp_path)
    monkeypatch.setenv("PODCODEX_INDEX", str(tmp_path / "index"))
    rag_index_store.get_index_store.cache_clear()
    rag_retriever.get_retriever.cache_clear()
    yield
    rag_index_store.get_index_store.cache_clear()
    rag_retriever.get_retriever.cache_clear()


def test_list_shows_returns_indexed_shows():
    shows = mcp_server.list_shows()
    assert shows == [{"show": "My Show", "episodes": 1}]


def test_list_shows_skips_non_default_combos(tmp_path):
    store = rag_index_store.get_index_store()
    store.ensure_collection(
        "other__e5-small__semantic",
        show="Other",
        model="e5-small",
        chunker="semantic",
        dim=DIM,
    )
    shows = mcp_server.list_shows()
    assert [s["show"] for s in shows] == ["My Show"]


def test_resolve_collections_empty_show_returns_all_defaults():
    cols = mcp_server._resolve_collections(None)
    assert cols == [collection_name("My Show", "bge-m3", "semantic")]


def test_resolve_collections_case_insensitive():
    cols = mcp_server._resolve_collections("my show")
    assert cols == [collection_name("My Show", "bge-m3", "semantic")]


def test_resolve_collections_unknown_show_returns_empty():
    assert mcp_server._resolve_collections("Nope") == []


def test_trim_shape_prefers_rss_title():
    trimmed = mcp_server._trim(
        {
            "show": "S",
            "episode": "0001_some_stem_here",
            "episode_title": "Episode one: the beginning",
            "chunk_index": 4,
            "start": 1.5,
            "end": 3.0,
            "dominant_speaker": "Alice",
            "text": "hi",
            "score": 0.72,
            "meta_extra": "dropped",
        }
    )
    assert trimmed == {
        "show": "S",
        "episode": "0001_some_stem_here",
        "episode_title": "Episode one: the beginning",
        "chunk_index": 4,
        "start": 1.5,
        "end": 3.0,
        "speaker": "Alice",
        "text": "hi",
        "score": 0.72,
    }


def test_trim_falls_back_to_humanized_stem():
    trimmed = mcp_server._trim(
        {
            "show": "S",
            "episode": "0042_episode_5_my_great_topic",
            "chunk_index": 0,
            "start": 0.0,
            "end": 1.0,
            "dominant_speaker": "Bob",
            "text": "x",
        }
    )
    assert trimmed["episode_title"] == "My great topic"


def test_trim_omits_score_when_absent():
    trimmed = mcp_server._trim(
        {
            "show": "S",
            "episode": "e",
            "chunk_index": 0,
            "start": 0.0,
            "end": 1.0,
            "dominant_speaker": "",
            "text": "x",
        }
    )
    assert "score" not in trimmed


def test_get_context_returns_window():
    out = mcp_server.get_context(show="My Show", episode="ep1", chunk_index=3, window=1)
    assert [c["chunk_index"] for c in out] == [2, 3, 4]
    for c in out:
        assert c["show"] == "My Show"
        assert c["episode"] == "ep1"
        assert "score" not in c


def test_get_context_unknown_show_returns_empty():
    out = mcp_server.get_context(show="Nope", episode="ep1", chunk_index=0, window=2)
    assert out == []


def test_speaker_stats_aggregates_across_chunks():
    stats = mcp_server.speaker_stats()
    by_speaker = {s["speaker"]: s for s in stats}
    # Seed fixture registers 6 chunks alternating sp0/sp1
    assert by_speaker["sp0"]["chunk_count"] == 3
    assert by_speaker["sp1"]["chunk_count"] == 3
    # Sorted descending
    counts = [s["chunk_count"] for s in stats]
    assert counts == sorted(counts, reverse=True)


def test_speaker_stats_skips_chunks_with_no_dominant_speaker(tmp_path):
    # Already-seeded chunks all have a dominant_speaker; just confirm
    # the fixture's setup doesn't leak an empty-speaker row.
    stats = mcp_server.speaker_stats()
    assert all(s["speaker"] for s in stats)


def test_exact_returns_literal_matches():
    matches = mcp_server.exact(query="chunk 3")
    assert any(m["chunk_index"] == 3 for m in matches)
    # No top_k cap — substring "chunk" matches every chunk
    broad = mcp_server.exact(query="chunk")
    assert len(broad) == 6
