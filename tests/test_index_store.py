"""Tests for podcodex.rag.index_store — every test uses a fresh tmp_path index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from podcodex.rag.index_store import IndexStore


def _store(tmp_path: Path) -> IndexStore:
    return IndexStore(tmp_path / "index")


def _rng_embeddings(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim), dtype=np.float32)


def _chunks(n: int, episode: str = "ep1") -> list[dict]:
    return [
        {
            "text": f"chunk {i}",
            "start": float(i),
            "end": float(i + 1),
            "episode": episode,
            "show": "S",
            "source": "transcript",
            "dominant_speaker": f"sp{i % 2}",
        }
        for i in range(n)
    ]


# ── Collection management ────────────────────────────────────────────────


def test_collection_not_exists_initially(tmp_path):
    s = _store(tmp_path)
    assert not s.collection_exists("my_show__bge-m3__semantic")


def test_ensure_collection_creates(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection(
        "my_show__bge-m3__semantic",
        show="My Show",
        model="bge-m3",
        chunker="semantic",
        dim=8,
    )
    assert s.collection_exists("my_show__bge-m3__semantic")


def test_ensure_collection_idempotent(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    assert s.list_collections() == ["c"]


def test_list_collections_filter_show(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("a", show="A", model="m", chunker="semantic", dim=8)
    s.ensure_collection("b", show="B", model="m", chunker="semantic", dim=8)
    assert s.list_collections(show="A") == ["a"]


def test_list_collections_filter_model(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("a__bge", show="A", model="bge-m3", chunker="semantic", dim=8)
    s.ensure_collection("a__e5", show="A", model="e5", chunker="semantic", dim=8)
    assert s.list_collections(model="e5") == ["a__e5"]


def test_get_collection_info(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="bge-m3", chunker="semantic", dim=1024)
    assert s.get_collection_info("c") == {
        "show": "S",
        "model": "bge-m3",
        "chunker": "semantic",
        "dim": 1024,
    }


def test_get_collection_info_missing(tmp_path):
    s = _store(tmp_path)
    assert s.get_collection_info("nope") is None


def test_delete_collection(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.delete_collection("c")
    assert not s.collection_exists("c")


def test_delete_collection_no_op_if_missing(tmp_path):
    _store(tmp_path).delete_collection("nope")  # should not raise


# ── Episode-level ────────────────────────────────────────────────────────


def test_episode_indexed_after_save(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(2), _rng_embeddings(2))
    assert s.episode_is_indexed("c", "ep1")
    assert s.episode_chunk_count("c", "ep1") == 2


def test_delete_episode_removes_only_target(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(2, "ep1"), _rng_embeddings(2))
    s.save_chunks("c", "ep2", _chunks(2, "ep2"), _rng_embeddings(2))
    s.delete_episode("c", "ep1")
    assert not s.episode_is_indexed("c", "ep1")
    assert s.episode_is_indexed("c", "ep2")


def test_list_episodes_sorted(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep_b", _chunks(1, "ep_b"), _rng_embeddings(1))
    s.save_chunks("c", "ep_a", _chunks(1, "ep_a"), _rng_embeddings(1))
    assert s.list_episodes("c") == ["ep_a", "ep_b"]


# ── save_chunks ──────────────────────────────────────────────────────────


def test_save_chunks_text_round_trip(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks(
        "c", "ep1", [{"text": "hello world", "start": 0.0}], _rng_embeddings(1)
    )
    loaded = s.load_chunks_no_embeddings("c", "ep1")
    assert loaded[0]["text"] == "hello world"


def test_save_chunks_meta_round_trip(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    chunk = {
        "text": "t",
        "start": 1.5,
        "end": 3.0,
        "speakers": [{"speaker": "Alice", "text": "hi"}],
    }
    s.save_chunks("c", "ep1", [chunk], _rng_embeddings(1))
    loaded = s.load_chunks_no_embeddings("c", "ep1")
    assert loaded[0]["start"] == pytest.approx(1.5)
    assert loaded[0]["end"] == pytest.approx(3.0)
    assert loaded[0]["speakers"] == [{"speaker": "Alice", "text": "hi"}]


def test_save_chunks_length_mismatch_raises(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    with pytest.raises(ValueError, match="Length mismatch"):
        s.save_chunks("c", "ep1", _chunks(3), _rng_embeddings(2))


# ── get_chunk_window ─────────────────────────────────────────────────────


def _prepare_episode(tmp_path: Path, n: int = 10) -> IndexStore:
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "ep1", _chunks(n, "ep1"), _rng_embeddings(n))
    return s


def test_get_chunk_window_returns_center_plus_neighbors(tmp_path):
    s = _prepare_episode(tmp_path, n=10)
    window = s.get_chunk_window("c", "ep1", chunk_index=5, window=2)
    assert [c["chunk_index"] for c in window] == [3, 4, 5, 6, 7]


def test_get_chunk_window_clamps_at_start(tmp_path):
    s = _prepare_episode(tmp_path, n=10)
    window = s.get_chunk_window("c", "ep1", chunk_index=1, window=3)
    assert [c["chunk_index"] for c in window] == [0, 1, 2, 3, 4]


def test_get_chunk_window_clamps_at_end(tmp_path):
    s = _prepare_episode(tmp_path, n=10)
    window = s.get_chunk_window("c", "ep1", chunk_index=9, window=3)
    assert [c["chunk_index"] for c in window] == [6, 7, 8, 9]


def test_get_chunk_window_zero_width_returns_center_only(tmp_path):
    s = _prepare_episode(tmp_path, n=5)
    window = s.get_chunk_window("c", "ep1", chunk_index=2, window=0)
    assert [c["chunk_index"] for c in window] == [2]


def test_get_chunk_window_missing_center_returns_empty(tmp_path):
    s = _prepare_episode(tmp_path, n=5)
    assert s.get_chunk_window("c", "ep1", chunk_index=99) == []


def test_get_chunk_window_missing_episode_returns_empty(tmp_path):
    s = _prepare_episode(tmp_path, n=3)
    assert s.get_chunk_window("c", "missing", chunk_index=0) == []


def test_get_chunk_window_negative_window_treated_as_zero(tmp_path):
    s = _prepare_episode(tmp_path, n=5)
    window = s.get_chunk_window("c", "ep1", chunk_index=2, window=-3)
    assert [c["chunk_index"] for c in window] == [2]


# ── Native search ────────────────────────────────────────────────────────


def test_search_vector_returns_self_first(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=4)
    vecs = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    chunks = [
        {"text": "first", "episode": "e", "start": 0.0, "end": 1.0},
        {"text": "second", "episode": "e", "start": 1.0, "end": 2.0},
        {"text": "third", "episode": "e", "start": 2.0, "end": 3.0},
    ]
    s.save_chunks("c", "e", chunks, vecs)
    hits = s.search_vector("c", vecs[0], top_k=2)
    assert hits[0]["text"] == "first"


def test_search_vector_episode_filter(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=4)
    vecs = _rng_embeddings(2, dim=4)
    s.save_chunks(
        "c",
        "ep1",
        [{"text": "alpha", "episode": "ep1", "start": 0.0, "end": 1.0}],
        vecs[:1],
    )
    s.save_chunks(
        "c",
        "ep2",
        [{"text": "beta", "episode": "ep2", "start": 0.0, "end": 1.0}],
        vecs[1:],
    )
    hits = s.search_vector("c", vecs[0], top_k=5, episode="ep2")
    assert [h["text"] for h in hits] == ["beta"]


def test_search_fts_finds_token(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=4)
    chunks = [
        {"text": "the quick brown fox", "episode": "e", "start": 0.0, "end": 1.0},
        {"text": "lazy dogs sleep all day", "episode": "e", "start": 1.0, "end": 2.0},
    ]
    s.save_chunks("c", "e", chunks, _rng_embeddings(2, dim=4))
    hits = s.search_fts("c", "fox", top_k=5)
    assert any("fox" in h["text"] for h in hits)


# ── Stats helpers ────────────────────────────────────────────────────────


def test_collection_chunk_count(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "e1", _chunks(3, "e1"), _rng_embeddings(3))
    s.save_chunks("c", "e2", _chunks(2, "e2"), _rng_embeddings(2))
    assert s.collection_chunk_count("c") == 5


def test_list_sources_and_speakers(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "e1", _chunks(4, "e1"), _rng_embeddings(4))
    assert s.list_sources("c") == ["transcript"]
    assert s.list_speakers("c") == ["sp0", "sp1"]


def test_get_episode_stats(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "e1", _chunks(3, "e1"), _rng_embeddings(3))
    stats = s.get_episode_stats("c")
    assert len(stats) == 1
    assert stats[0]["episode"] == "e1"
    assert stats[0]["chunk_count"] == 3
    assert stats[0]["duration"] == pytest.approx(3.0)
    assert set(stats[0]["speakers"]) == {"sp0", "sp1"}


# ── Persistence ──────────────────────────────────────────────────────────


def test_reopening_preserves_data(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    s.save_chunks("c", "e1", _chunks(2, "e1"), _rng_embeddings(2))
    s2 = IndexStore(tmp_path / "index")
    assert s2.collection_exists("c")
    assert s2.list_episodes("c") == ["e1"]
    assert s2.collection_chunk_count("c") == 2


# ── _normalize_pub_date ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024-01-15", "2024-01-15"),
        ("2024-01-15T12:34:56Z", "2024-01-15"),
        ("2024-01-15T12:34:56+02:00", "2024-01-15"),
        ("Mon, 15 Jan 2024 12:00:00 GMT", "2024-01-15"),
        ("Mon, 15 Jan 2024 12:00:00 +0200", "2024-01-15"),
        ("20240115", "2024-01-15"),
        ("", None),
        (None, None),
        ("not a date", None),
        ("2024", None),
    ],
)
def test_normalize_pub_date(raw, expected):
    from podcodex.rag.index_store import _normalize_pub_date

    assert _normalize_pub_date(raw) == expected


def test_normalize_pub_date_idempotent():
    from podcodex.rag.index_store import _normalize_pub_date

    assert _normalize_pub_date(
        _normalize_pub_date("Mon, 15 Jan 2024 12:00:00 GMT")
    ) == ("2024-01-15")


# ── pub_date column + migration ──────────────────────────────────────────


def test_save_chunks_writes_normalized_pub_date(tmp_path):
    s = _store(tmp_path)
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    chunk = {
        "text": "t",
        "start": 0.0,
        "end": 1.0,
        "episode": "ep1",
        "show": "S",
        "source": "transcript",
        "dominant_speaker": "A",
        "pub_date": "Mon, 15 Jan 2024 12:00:00 GMT",
    }
    s.save_chunks("c", "ep1", [chunk], _rng_embeddings(1))
    rows = s._table("c").search().select(["pub_date"]).limit(5).to_list()
    assert rows[0]["pub_date"] == "2024-01-15"


# ── _build_where ─────────────────────────────────────────────────────────


def test_build_where_empty():
    from podcodex.rag.index_store import _build_where

    assert _build_where() == ""


def test_build_where_episode_equality():
    from podcodex.rag.index_store import _build_where

    assert _build_where(episode="ep1") == "episode = 'ep1'"


def test_build_where_episodes_list_overrides_single():
    from podcodex.rag.index_store import _build_where

    assert (
        _build_where(episode="ignored", episodes=["a", "b"]) == "episode IN ('a', 'b')"
    )


def test_build_where_episodes_empty_list_falls_through_to_single():
    from podcodex.rag.index_store import _build_where

    assert _build_where(episode="ep1", episodes=[]) == "episode = 'ep1'"


def test_build_where_pub_date_range():
    from podcodex.rag.index_store import _build_where

    assert (
        _build_where(
            pub_date_min="2024-01-01", pub_date_max="Mon, 15 Jan 2024 12:00:00 GMT"
        )
        == "pub_date >= '2024-01-01' AND pub_date <= '2024-01-15'"
    )


def test_build_where_invalid_pub_date_min_raises():
    from podcodex.rag.index_store import _build_where

    with pytest.raises(ValueError, match="pub_date_min"):
        _build_where(pub_date_min="not a date")


def test_build_where_sql_injection_escaped():
    from podcodex.rag.index_store import _build_where

    w = _build_where(episode="o'brien", speaker="eve'--")
    assert "o''brien" in w
    assert "eve''--" in w


def test_build_where_combined():
    from podcodex.rag.index_store import _build_where

    w = _build_where(
        episodes=["ep1", "ep2"],
        source="transcript",
        speaker="A",
        pub_date_min="2024-01-01",
        pub_date_max="2024-12-31",
    )
    assert w == (
        "episode IN ('ep1', 'ep2') AND source = 'transcript' AND "
        "dominant_speaker = 'A' AND pub_date >= '2024-01-01' AND "
        "pub_date <= '2024-12-31'"
    )


# ── get_episode / list_episodes_filtered ─────────────────────────────────


def _seed_with_pub_dates(s: IndexStore):
    s.ensure_collection("c", show="S", model="m", chunker="semantic", dim=8)
    eps = [
        ("ep1", "2024-01-15", "First ep", 1, "desc one"),
        ("ep2", "2024-03-10", "Second ep", 2, "desc two"),
        ("ep3", "2024-06-20", "Third ep", 3, ""),
    ]
    for ep, pd, title, num, desc in eps:
        chunk = {
            "text": f"text {ep}",
            "start": 0.0,
            "end": 42.0,
            "episode": ep,
            "show": "S",
            "source": "transcript",
            "dominant_speaker": "Alice",
            "pub_date": pd,
            "episode_title": title,
            "episode_number": num,
        }
        if desc:
            chunk["description"] = desc
        s.save_chunks("c", ep, [chunk], _rng_embeddings(1))


def test_get_episode_returns_metadata(tmp_path):
    s = _store(tmp_path)
    _seed_with_pub_dates(s)
    ep = s.get_episode("c", "ep2")
    assert ep is not None
    assert ep["episode"] == "ep2"
    assert ep["pub_date"] == "2024-03-10"
    assert ep["episode_title"] == "Second ep"
    assert ep["episode_number"] == 2
    assert ep["description"] == "desc two"
    assert ep["source"] == "transcript"
    assert ep["chunk_count"] == 1
    assert ep["duration"] == pytest.approx(42.0)
    assert ep["speakers"] == ["Alice"]


def test_get_episode_missing_returns_none(tmp_path):
    s = _store(tmp_path)
    _seed_with_pub_dates(s)
    assert s.get_episode("c", "nope") is None


def test_list_episodes_filtered_by_date_range(tmp_path):
    s = _store(tmp_path)
    _seed_with_pub_dates(s)
    res = s.list_episodes_filtered(
        "c", pub_date_min="2024-02-01", pub_date_max="2024-05-01"
    )
    assert [r["episode"] for r in res] == ["ep2"]


def test_list_episodes_filtered_by_title(tmp_path):
    s = _store(tmp_path)
    _seed_with_pub_dates(s)
    res = s.list_episodes_filtered("c", title_contains="third")
    assert [r["episode"] for r in res] == ["ep3"]


def test_list_episodes_filtered_invalid_date_raises(tmp_path):
    s = _store(tmp_path)
    _seed_with_pub_dates(s)
    with pytest.raises(ValueError, match="pub_date_min"):
        s.list_episodes_filtered("c", pub_date_min="garbage")


def test_pub_date_migration_backfills_from_meta(tmp_path):
    """Opening a legacy table (no pub_date column) adds + backfills it."""
    import json
    import lancedb
    import pyarrow as pa

    db_path = tmp_path / "index"
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))

    legacy_schema = pa.schema(
        [
            pa.field("chunk_index", pa.int32()),
            pa.field("show", pa.string()),
            pa.field("episode", pa.string()),
            pa.field("source", pa.string()),
            pa.field("dominant_speaker", pa.string()),
            pa.field("start", pa.float64()),
            pa.field("end", pa.float64()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 8)),
            pa.field("meta", pa.string()),
        ]
    )
    t = db.create_table("c", schema=legacy_schema)
    # Also seed the _collections sidecar so collection_exists() is happy.
    from podcodex.rag.index_store import _COLLECTIONS_SCHEMA

    coll = db.create_table("_collections", schema=_COLLECTIONS_SCHEMA)
    coll.add(
        [
            {
                "name": "c",
                "show": "S",
                "model": "m",
                "chunker": "semantic",
                "dim": 8,
                "created_at": "",
            }
        ]
    )
    rng = np.random.default_rng(0).random((2, 8), dtype=np.float32)
    t.add(
        [
            {
                "chunk_index": 0,
                "show": "S",
                "episode": "ep1",
                "source": "transcript",
                "dominant_speaker": "A",
                "start": 0.0,
                "end": 1.0,
                "text": "alpha",
                "vector": rng[0].tolist(),
                "meta": json.dumps({"pub_date": "Mon, 15 Jan 2024 12:00:00 GMT"}),
            },
            {
                "chunk_index": 0,
                "show": "S",
                "episode": "ep2",
                "source": "transcript",
                "dominant_speaker": "B",
                "start": 0.0,
                "end": 1.0,
                "text": "beta",
                "vector": rng[1].tolist(),
                "meta": json.dumps({"rss_pub_date": "2024-02-20T10:00:00Z"}),
            },
        ]
    )

    store = IndexStore(db_path)
    # Triggering _table() via any read path runs the migration.
    rows = (
        store._table("c").search().select(["episode", "pub_date"]).limit(10).to_list()
    )
    by_ep = {r["episode"]: r["pub_date"] for r in rows}
    assert by_ep["ep1"] == "2024-01-15"
    assert by_ep["ep2"] == "2024-02-20"
    assert (db_path / "c.pub_date_col_v1").exists()
