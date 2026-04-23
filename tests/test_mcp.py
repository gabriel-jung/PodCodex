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
    # Pristine class-level resolver + list_shows date cache for each test so
    # one test's setup can't leak into another.
    IndexStore._show_folder_resolver = None
    mcp_server._SHOW_DATE_CACHE.clear()
    yield
    rag_index_store.get_index_store.cache_clear()
    rag_retriever.get_retriever.cache_clear()
    IndexStore._show_folder_resolver = None
    mcp_server._SHOW_DATE_CACHE.clear()


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


# ──────────────────────────────────────────────
# list_episodes / get_episode / search filters
# ──────────────────────────────────────────────


def _seed_multi_episode(store: IndexStore) -> str:
    """Add three dated episodes on top of the autouse fixture's ``ep1``.

    Returns the collection name so tests can re-use it.
    """
    col = collection_name("My Show", "bge-m3", "semantic")
    rng = np.random.default_rng(1)
    dated = [
        ("ep-jan", "2024-01-15", "Episode one: January deep dive", 101),
        ("ep-mar", "2024-03-10", "Episode two: March special", 102),
        ("ep-jun", "2024-06-01", "Episode three: June finale", 103),
    ]
    for stem, pd, title, number in dated:
        chunks = [
            {
                "text": f"content for {stem}",
                "episode": stem,
                "show": "My Show",
                "source": "transcript",
                "dominant_speaker": "host",
                "start": 0.0,
                "end": 10.0,
                "pub_date": pd,
                "episode_title": title,
                "episode_number": number,
                "description": f"desc for {stem}",
            }
        ]
        store.save_chunks(col, stem, chunks, rng.random((1, DIM), dtype=np.float32))
    return col


def test_list_episodes_returns_all_when_no_filter():
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    out = mcp_server.list_episodes()
    stems = {e["episode"] for e in out}
    assert stems == {"ep1", "ep-jan", "ep-mar", "ep-jun"}
    expected_keys = {
        "show",
        "episode",
        "episode_title",
        "pub_date",
        "episode_number",
        "chunk_count",
        "duration",
        "speakers",
        "description",
    }
    for entry in out:
        assert expected_keys <= entry.keys()
    # Dated entries carry real speakers + description from the seed chunks.
    by_stem = {e["episode"]: e for e in out}
    assert by_stem["ep-mar"]["speakers"] == ["host"]
    assert by_stem["ep-mar"]["description"] == "desc for ep-mar"


def test_list_episodes_restricts_to_show_case_insensitive():
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    # Matching show yields rows
    assert mcp_server.list_episodes(show="my show")
    # Unknown show yields empty list (not an error)
    assert mcp_server.list_episodes(show="nope") == []


def test_list_episodes_filters_by_pub_date_range():
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    out = mcp_server.list_episodes(pub_date_min="2024-02-01", pub_date_max="2024-04-30")
    stems = {e["episode"] for e in out}
    assert stems == {"ep-mar"}


def test_list_episodes_filters_by_title_contains_case_insensitive():
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    out = mcp_server.list_episodes(title_contains="MARCH")
    assert [e["episode"] for e in out] == ["ep-mar"]


def test_get_episode_returns_full_metadata():
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    rec = mcp_server.get_episode(show="My Show", episode="ep-mar")
    assert rec is not None
    assert rec["episode"] == "ep-mar"
    assert rec["episode_title"] == "Episode two: March special"
    assert rec["pub_date"] == "2024-03-10"
    assert rec["episode_number"] == 102
    assert rec["description"] == "desc for ep-mar"
    assert rec["duration"] == 10.0
    assert "host" in rec["speakers"]


def test_get_episode_unknown_stem_returns_none():
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    assert mcp_server.get_episode(show="My Show", episode="no-such") is None


def _stub_retriever_encoder(monkeypatch):
    """Replace the shared retriever's ``encode_query`` with an 8-dim zero vector
    so MCP ``search`` can run against the test fixture without pulling the
    live 1024-dim BGE-M3 weights."""
    retriever = rag_retriever.get_retriever()
    monkeypatch.setattr(
        retriever,
        "encode_query",
        lambda _q: np.zeros(DIM, dtype=np.float32),
    )


def test_exact_restricts_to_episodes_list():
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    results = mcp_server.exact(query="content", episodes=["ep-jan", "ep-jun"])
    stems = {r["episode"] for r in results}
    assert stems and stems <= {"ep-jan", "ep-jun"}


def test_exact_respects_pub_date_range():
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    results = mcp_server.exact(
        query="content",
        pub_date_min="2024-02-01",
        pub_date_max="2024-04-30",
    )
    stems = {r["episode"] for r in results}
    assert stems == {"ep-mar"}


def test_exact_chunks_carry_pub_date():
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    results = mcp_server.exact(query="content", episodes=["ep-jun"])
    assert results and all(r.get("pub_date") == "2024-06-01" for r in results)


def test_list_shows_adds_date_range_when_available():
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    shows = mcp_server.list_shows()
    # Single indexed show; fixture's ep1 has no date, but ep-jan/ep-mar/ep-jun do
    entry = shows[0]
    assert entry["show"] == "My Show"
    assert entry["first_pub_date"] == "2024-01-15"
    assert entry["last_pub_date"] == "2024-06-01"


def test_search_restricts_to_episodes_list(monkeypatch):
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    _stub_retriever_encoder(monkeypatch)
    results = mcp_server.search(query="content", episodes=["ep-jan", "ep-jun"])
    stems = {r["episode"] for r in results}
    assert stems and stems <= {"ep-jan", "ep-jun"}


def test_search_respects_pub_date_range(monkeypatch):
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    _stub_retriever_encoder(monkeypatch)
    results = mcp_server.search(
        query="content",
        pub_date_min="2024-02-01",
        pub_date_max="2024-04-30",
    )
    stems = {r["episode"] for r in results}
    assert stems == {"ep-mar"}


def test_search_chunks_carry_pub_date(monkeypatch):
    store = rag_index_store.get_index_store()
    _seed_multi_episode(store)
    _stub_retriever_encoder(monkeypatch)
    results = mcp_server.search(query="content", episodes=["ep-jun"])
    assert results and all(r.get("pub_date") == "2024-06-01" for r in results)
