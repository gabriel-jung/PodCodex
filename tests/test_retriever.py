"""Tests for podcodex.rag.retriever — backed by a real on-disk IndexStore."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from podcodex.rag.index_store import IndexStore


DIM = 4


def _seed_index(
    tmp_path: Path, episodes: dict[str, int] | None = None
) -> tuple[IndexStore, str]:
    """Create an IndexStore at tmp_path/index with seeded episodes.

    Returns ``(store, collection_name)``.
    """
    local = IndexStore(tmp_path / "index")
    col = "test__bge-m3__semantic"
    local.ensure_collection(
        col, show="test", model="bge-m3", chunker="semantic", dim=DIM
    )

    if episodes is None:
        episodes = {"ep1": 3, "ep2": 2}

    rng = np.random.default_rng(0)
    for ep, n in episodes.items():
        chunks = [
            {
                "episode": ep,
                "show": "test",
                "start": float(i),
                "end": float(i + 1),
                "dominant_speaker": "Alice" if i % 2 == 0 else "Bob",
                "source": "corrected",
                "text": f"chunk {i} of {ep} about neural networks and podcasting",
            }
            for i in range(n)
        ]
        embeddings = rng.random((n, DIM)).astype(np.float32)
        local.save_chunks(col, ep, chunks, embeddings)
    return local, col


def _make_retriever(
    tmp_path: Path,
    local: IndexStore | None = None,
    col: str = "",
):
    """Return ``(retriever, mock_embedder, collection_name)`` with the embedder mocked."""
    if local is None:
        local, col = _seed_index(tmp_path)

    mock_emb = MagicMock()
    mock_emb.encode_query.return_value = np.random.rand(DIM).astype(np.float32)

    from podcodex.rag.retriever import Retriever

    retriever = Retriever(model="bge-m3", local=local)
    retriever._embedder = mock_emb  # bypass lazy load of real BGE-M3

    return retriever, mock_emb, col


# ── Constructor ──────────────────────────────────────────────────────────


def test_retriever_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        from podcodex.rag.retriever import Retriever

        Retriever(model="bad_model")


# ── Dense search (alpha=1.0) ─────────────────────────────────────────────


def test_dense_search_returns_results(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    results = retriever.retrieve("neural networks", col, top_k=3, alpha=1.0)
    assert all("score" in r for r in results)
    assert all("text" in r for r in results)


def test_dense_search_respects_top_k(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    results = retriever.retrieve("q", col, top_k=2, alpha=1.0)
    assert len(results) <= 2


def test_dense_search_empty_collection(tmp_path):
    local = IndexStore(tmp_path / "empty")
    retriever, _, _ = _make_retriever(tmp_path, local=local, col="missing")
    assert retriever.retrieve("q", "nonexistent", top_k=5, alpha=1.0) == []


# ── FTS search (alpha=0.0) ───────────────────────────────────────────────


def test_fts_search_returns_results(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    results = retriever.retrieve("neural", col, top_k=5, alpha=0.0)
    assert len(results) > 0


def test_fts_search_empty_collection(tmp_path):
    local = IndexStore(tmp_path / "empty")
    retriever, _, _ = _make_retriever(tmp_path, local=local, col="missing")
    assert retriever.retrieve("q", "nonexistent", alpha=0.0) == []


# ── Hybrid (default alpha=0.5) ───────────────────────────────────────────


def test_weighted_search_returns_results(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    results = retriever.retrieve("podcasting neural", col, top_k=5, alpha=0.5)
    assert len(results) > 0


def test_weighted_search_blends_scores(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    results = retriever.retrieve("neural", col, top_k=5, alpha=0.5)
    assert all(r["score"] >= 0 for r in results)


# ── Filters ──────────────────────────────────────────────────────────────


def test_dense_search_episode_filter(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    results = retriever.retrieve("q", col, top_k=10, alpha=1.0, episode="ep1")
    assert all(r.get("episode") == "ep1" for r in results)


def test_dense_search_speaker_filter(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    results = retriever.retrieve("q", col, top_k=10, alpha=1.0, speaker="Alice")
    assert all(r.get("dominant_speaker") == "Alice" for r in results)


def test_dense_search_episodes_list_filter(tmp_path):
    """episodes=[...] restricts to the given stems."""
    local, col = _seed_index(tmp_path, {"ep1": 2, "ep2": 2, "ep3": 2})
    retriever, _, _ = _make_retriever(tmp_path, local=local, col=col)
    results = retriever.retrieve("q", col, top_k=10, alpha=1.0, episodes=["ep1", "ep3"])
    eps = {r.get("episode") for r in results}
    assert eps <= {"ep1", "ep3"}
    assert eps  # non-empty


def test_dense_search_pub_date_range_filter(tmp_path):
    """pub_date_min/max restricts by date."""
    local = IndexStore(tmp_path / "index")
    col = "test__bge-m3__semantic"
    local.ensure_collection(
        col, show="test", model="bge-m3", chunker="semantic", dim=DIM
    )
    rng = np.random.default_rng(0)
    for ep, pd in [("ep1", "2024-01-15"), ("ep2", "2024-03-10"), ("ep3", "2024-06-01")]:
        chunks = [
            {
                "episode": ep,
                "show": "test",
                "start": 0.0,
                "end": 1.0,
                "dominant_speaker": "A",
                "source": "corrected",
                "text": f"chunk {ep}",
                "pub_date": pd,
            }
        ]
        local.save_chunks(col, ep, chunks, rng.random((1, DIM)).astype(np.float32))
    retriever, _, _ = _make_retriever(tmp_path, local=local, col=col)
    results = retriever.retrieve(
        "q",
        col,
        top_k=10,
        alpha=1.0,
        pub_date_min="2024-02-01",
        pub_date_max="2024-04-30",
    )
    assert {r.get("episode") for r in results} == {"ep2"}


# ── exact / random ────────────────────────────────────────────────────────


def test_exact_returns_token_match(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    results = retriever.exact("neural", col)
    assert len(results) > 0
    assert all(r["score"] == 1.0 for r in results)


def test_exact_no_match(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    assert retriever.exact("xyznonexistent", col) == []


def test_exact_speaker_filter_is_turn_level(tmp_path):
    """speaker=X on /exact keeps chunks where X utters the phrase, not just
    chunks dominated by X."""
    local = IndexStore(tmp_path / "index")
    col = "test__bge-m3__semantic"
    local.ensure_collection(
        col, show="test", model="bge-m3", chunker="semantic", dim=DIM
    )
    rng = np.random.default_rng(0)
    # Alice dominates, Bob says "neural networks" in a turn.
    chunk1 = {
        "episode": "ep1",
        "show": "test",
        "start": 0.0,
        "end": 10.0,
        "dominant_speaker": "Alice",
        "source": "corrected",
        "text": "alice talks a lot. bob mentions neural networks briefly.",
        "speakers": [
            {
                "speaker": "Alice",
                "text": "alice talks a lot.",
                "start": 0.0,
                "end": 6.0,
            },
            {
                "speaker": "Bob",
                "text": "bob mentions neural networks briefly.",
                "start": 6.0,
                "end": 10.0,
            },
        ],
    }
    # Alice dominates, only Alice speaks — no "neural networks".
    chunk2 = {
        "episode": "ep2",
        "show": "test",
        "start": 0.0,
        "end": 5.0,
        "dominant_speaker": "Alice",
        "source": "corrected",
        "text": "alice only here.",
        "speakers": [
            {"speaker": "Alice", "text": "alice only here.", "start": 0.0, "end": 5.0},
        ],
    }
    local.save_chunks(col, "ep1", [chunk1], rng.random((1, DIM)).astype(np.float32))
    local.save_chunks(col, "ep2", [chunk2], rng.random((1, DIM)).astype(np.float32))
    retriever, _, _ = _make_retriever(tmp_path, local=local, col=col)

    # Turn-level: Bob is the speaker, he utters the phrase → one hit.
    hits = retriever.exact("neural networks", col, speaker="Bob")
    assert {h["episode"] for h in hits} == {"ep1"}

    # Without speaker filter: chunk with the phrase matches regardless.
    hits = retriever.exact("neural networks", col)
    assert {h["episode"] for h in hits} == {"ep1"}

    # Alice doesn't utter the phrase → no hits, even though she dominates ep1.
    hits = retriever.exact("neural networks", col, speaker="Alice")
    assert hits == []


def test_random_returns_chunk(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    result = retriever.random(col)
    assert result is not None
    assert "text" in result


def test_random_empty_collection(tmp_path):
    local = IndexStore(tmp_path / "empty")
    retriever, _, _ = _make_retriever(tmp_path, local=local, col="missing")
    assert retriever.random("nonexistent") is None


# ── _rank_normalize ──────────────────────────────────────────────────────


def test_rank_normalize_empty():
    from podcodex.rag.retriever import _rank_normalize

    assert _rank_normalize([]) == []


def test_rank_normalize_single_result():
    from podcodex.rag.retriever import _rank_normalize

    result = _rank_normalize([{"score": 0.3, "text": "a"}])
    assert result[0]["score"] == pytest.approx(1.0)


def test_rank_normalize_rank_based_scores():
    from podcodex.rag.retriever import _rank_normalize

    results = [{"score": 0.0, "text": "a"}, {"score": 0.0, "text": "b"}]
    normed = _rank_normalize(results)
    assert normed[0]["score"] == pytest.approx(1.0)
    assert normed[1]["score"] == pytest.approx(0.5)
