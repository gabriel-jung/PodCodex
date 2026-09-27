"""Tests for podcodex.rag.retriever — backed by a real on-disk IndexStore."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from podcodex.rag.index_store import IndexStore
from podcodex.rag.hit import Hit
from podcodex.rag.retriever import _chunk_key, merge_results
from tests.fixtures.index import add_show, chunk


DIM = 4


def _seed_index(
    tmp_path: Path, episodes: dict[str, int] | None = None
) -> tuple[IndexStore, str]:
    """An index with show "test": ``{stem: chunk count}``, speakers alternating
    Alice / Bob. Returns ``(store, collection_name)``."""
    local = IndexStore(tmp_path / "index")
    col = add_show(
        local,
        "test",
        {
            ep: [
                chunk(
                    f"chunk {i} of {ep} about neural networks and podcasting",
                    episode=ep,
                    show="test",
                    speaker="Alice" if i % 2 == 0 else "Bob",
                    start=float(i),
                    end=float(i + 1),
                    source="corrected",
                )
                for i in range(n)
            ]
            for ep, n in (episodes or {"ep1": 3, "ep2": 2}).items()
        },
        dim=DIM,
    )
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


# ── Dense, FTS and the blend between them ────────────────────────────────


def _split_corpus(tmp_path: Path):
    """Two chunks each only one retriever can find, plus close-ish filler.

    "vec" sits on the query vector but lacks the query word; "word" has the
    word but points elsewhere. So which one ranks first says which side of
    the blend alpha favoured.
    """
    local = IndexStore(tmp_path / "index")
    col = "split__bge-m3__semantic"
    local.ensure_collection(col, show="t", model="bge-m3", chunker="semantic", dim=DIM)
    rows = {
        "vec": ("the weather is mild today", [1.0, 0.0, 0.0, 0.0]),
        "word": ("neural networks explained", [0.1, 1.0, 0.0, 0.0]),
        **{
            f"filler{i}": (f"filler talk number {i}", [0.3, 0.2, 1.0, 0.1 * i])
            for i in range(4)
        },
    }
    for ep, (text, vec) in rows.items():
        chunk = {"episode": ep, "show": "t", "start": 0.0, "end": 1.0, "text": text}
        local.save_chunks(col, ep, [chunk], np.array([vec], dtype=np.float32))
    retriever, _, _ = _make_retriever(tmp_path, local=local, col=col)
    return retriever, col


@pytest.mark.parametrize(
    "alpha, top, count",
    [(1.0, "vec", 3), (0.9, "vec", 3), (0.1, "word", 3), (0.0, "word", 1)],
)
def test_alpha_decides_which_retriever_ranks_first(tmp_path, alpha, top, count):
    retriever, col = _split_corpus(tmp_path)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    results = retriever.retrieve(
        "neural", col, top_k=3, alpha=alpha, query_vector=query
    )

    assert results[0].episode == top
    # Dense and the blend fill top_k; FTS alone only finds the one match.
    assert len(results) == count


def test_dense_search_empty_collection(tmp_path):
    local = IndexStore(tmp_path / "empty")
    retriever, _, _ = _make_retriever(tmp_path, local=local, col="missing")
    assert retriever.retrieve("q", "nonexistent", top_k=5, alpha=1.0) == []


def test_fts_search_empty_collection(tmp_path):
    local = IndexStore(tmp_path / "empty")
    retriever, _, _ = _make_retriever(tmp_path, local=local, col="missing")
    assert retriever.retrieve("q", "nonexistent", alpha=0.0) == []


# ── Filters ──────────────────────────────────────────────────────────────


def test_dense_search_episode_filter(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    results = retriever.retrieve("q", col, top_k=10, alpha=1.0, episode="ep1")
    assert results
    assert all(r.episode == "ep1" for r in results)


def test_dense_search_speaker_filter(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    results = retriever.retrieve("q", col, top_k=10, alpha=1.0, speaker="Alice")
    assert results
    assert all(r.dominant_speaker == "Alice" for r in results)


def test_dense_search_episodes_list_filter(tmp_path):
    """episodes=[...] restricts to the given stems."""
    local, col = _seed_index(tmp_path, {"ep1": 2, "ep2": 2, "ep3": 2})
    retriever, _, _ = _make_retriever(tmp_path, local=local, col=col)
    results = retriever.retrieve("q", col, top_k=10, alpha=1.0, episodes=["ep1", "ep3"])
    eps = {r.episode for r in results}
    assert eps <= {"ep1", "ep3"}
    assert eps  # non-empty


def test_dense_search_pub_date_range_filter(tmp_path):
    """pub_date_min/max restricts by date."""
    local = IndexStore(tmp_path / "index")
    dates = {"ep1": "2024-01-15", "ep2": "2024-03-10", "ep3": "2024-06-01"}
    col = add_show(
        local,
        "test",
        {
            ep: [chunk(f"chunk {ep}", episode=ep, show="test", pub_date=pd)]
            for ep, pd in dates.items()
        },
        dim=DIM,
    )
    retriever, _, _ = _make_retriever(tmp_path, local=local, col=col)
    results = retriever.retrieve(
        "q",
        col,
        top_k=10,
        alpha=1.0,
        pub_date_min="2024-02-01",
        pub_date_max="2024-04-30",
    )
    assert {r.episode for r in results} == {"ep2"}


# ── exact / random ────────────────────────────────────────────────────────


def test_exact_returns_token_match(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    results = retriever.exact("neural", col)
    assert len(results) > 0
    assert all(r.score == 1.0 for r in results)


def test_exact_no_match(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    assert retriever.exact("xyznonexistent", col) == []


def test_exact_speaker_filter_is_turn_level(tmp_path):
    """speaker=X on /exact keeps chunks where X utters the phrase, not just
    chunks dominated by X."""
    local = IndexStore(tmp_path / "index")
    alice = "alice talks a lot."
    bob = "bob mentions neural networks briefly."
    turn = {"speaker": "Alice", "text": "alice only here.", "start": 0.0, "end": 5.0}
    col = add_show(
        local,
        "test",
        {
            # Alice dominates, Bob says "neural networks" in a turn.
            "ep1": [
                chunk(
                    f"{alice} {bob}",
                    show="test",
                    end=10.0,
                    speakers=[
                        {"speaker": "Alice", "text": alice, "start": 0.0, "end": 6.0},
                        {"speaker": "Bob", "text": bob, "start": 6.0, "end": 10.0},
                    ],
                )
            ],
            # Alice dominates, only Alice speaks: no "neural networks".
            "ep2": [chunk(turn["text"], episode="ep2", show="test", speakers=[turn])],
        },
        dim=DIM,
    )
    retriever, _, _ = _make_retriever(tmp_path, local=local, col=col)

    # Turn-level: Bob is the speaker, he utters the phrase → one hit.
    hits = retriever.exact("neural networks", col, speaker="Bob")
    assert {h.episode for h in hits} == {"ep1"}

    # Without speaker filter: chunk with the phrase matches regardless.
    hits = retriever.exact("neural networks", col)
    assert {h.episode for h in hits} == {"ep1"}

    # Alice doesn't utter the phrase → no hits, even though she dominates ep1.
    hits = retriever.exact("neural networks", col, speaker="Alice")
    assert hits == []


def test_random_returns_chunk(tmp_path):
    retriever, _, col = _make_retriever(tmp_path)
    result = retriever.random(col)
    assert result is not None
    assert result.text


def test_random_empty_collection(tmp_path):
    local = IndexStore(tmp_path / "empty")
    retriever, _, _ = _make_retriever(tmp_path, local=local, col="missing")
    assert retriever.random("nonexistent") is None


# ── rank_normalize ──────────────────────────────────────────────────────


def test_rank_normalize_empty():
    from podcodex.rag.retriever import rank_normalize

    assert rank_normalize([]) == []


def test_rank_normalize_single_result():
    from podcodex.rag.retriever import rank_normalize

    result = rank_normalize([Hit(score=0.3, text="a")])
    assert result[0].score == pytest.approx(1.0)


def test_rank_normalize_rank_based_scores():
    from podcodex.rag.retriever import rank_normalize

    results = [Hit(score=0.0, text="a"), Hit(score=0.0, text="b")]
    normed = rank_normalize(results)
    assert normed[0].score == pytest.approx(1.0)
    assert normed[1].score == pytest.approx(0.5)


# ── exact_counts (count/batch mode) ──────────────────────────────────────


def test_exact_counts_group_by_episode(tmp_path):
    ret, _emb, col = _make_retriever(tmp_path)  # ep1: 3 chunks, ep2: 2 chunks
    out = ret.exact_counts(["chunk 0", "chunk"], col, group_by="episode")
    assert out["chunk 0"] == {"ep1": 1, "ep2": 1}
    assert out["chunk"]["ep1"] == 3
    assert out["chunk"]["ep2"] == 2


def test_exact_counts_empty_group_omitted(tmp_path):
    ret, _emb, col = _make_retriever(tmp_path)
    out = ret.exact_counts(["nonexistent phrase zzz"], col)
    assert out["nonexistent phrase zzz"] == {}


def test_exact_counts_first_hit(tmp_path):
    ret, _emb, col = _make_retriever(tmp_path)
    out = ret.exact_counts(["chunk 0"], col, group_by="episode", first_hit=True)
    entry = out["chunk 0"]["ep1"]
    assert entry["count"] == 1
    assert entry["first"]["chunk_index"] == 0
    assert entry["first"]["start_hms"] == "0m00"


# ──────────────────────────────────────────────
# merge_results
# ──────────────────────────────────────────────


def _hits(scores: list[float]) -> list[Hit]:
    return [Hit(text=f"t{i}", score=s) for i, s in enumerate(scores)]


def test_merge_score_strategy_sorts_globally():
    hits_by_col = {
        "a": _hits([0.9, 0.5]),
        "b": _hits([0.8, 0.6]),
    }
    merged = merge_results(hits_by_col, top_k=4, strategy="score")
    scores = [c.score for c, _ in merged]
    assert scores == [0.9, 0.8, 0.6, 0.5]


def test_merge_score_strategy_respects_top_k():
    hits_by_col = {"a": _hits([0.9, 0.8, 0.7])}
    merged = merge_results(hits_by_col, top_k=2, strategy="score")
    assert len(merged) == 2


def test_merge_roundrobin_interleaves():
    hits_by_col = {
        "a": _hits([0.9, 0.7]),
        "b": _hits([0.8, 0.6]),
    }
    merged = merge_results(hits_by_col, top_k=4, strategy="roundrobin")
    collections = [col for _, col in merged]
    # Round-robin alternates between collections
    assert collections[0] != collections[1]


def test_merge_roundrobin_respects_top_k():
    hits_by_col = {
        "a": _hits([0.9, 0.7, 0.5]),
        "b": _hits([0.8, 0.6, 0.4]),
    }
    merged = merge_results(hits_by_col, top_k=3, strategy="roundrobin")
    assert len(merged) == 3


def test_merge_roundrobin_uneven_collections():
    hits_by_col = {
        "a": _hits([0.9]),
        "b": _hits([0.8, 0.6, 0.4]),
    }
    merged = merge_results(hits_by_col, top_k=4, strategy="roundrobin")
    assert len(merged) == 4
    # "a" exhausted after 1, remaining come from "b"
    assert sum(1 for _, col in merged if col == "a") == 1
    assert sum(1 for _, col in merged if col == "b") == 3


def test_merge_empty_input():
    assert merge_results({}, top_k=5, strategy="score") == []
    assert merge_results({}, top_k=5, strategy="roundrobin") == []


def test_merge_single_collection():
    hits_by_col = {"a": _hits([0.9, 0.5])}
    merged = merge_results(hits_by_col, top_k=5, strategy="roundrobin")
    assert len(merged) == 2
    assert all(col == "a" for _, col in merged)


def test_random_turn_flatten_keeps_unnamed_speaker_empty(tmp_path, monkeypatch):
    """An unnamed turn must not be given an invented "Unknown" name: the
    display layer (display_speaker) owns how a blank speaker is rendered."""
    from podcodex.rag.hit import Hit, SpeakerTurn
    from podcodex.rag.retriever import Retriever

    chunk = Hit(
        episode="ep1",
        show="S",
        start=10.0,
        end=20.0,
        text="a b",
        speakers=[
            SpeakerTurn(speaker="", text="a", start=10.0, end=15.0),
            SpeakerTurn(speaker="", text="b", start=15.0, end=20.0),
        ],
    )

    class _Store:
        def count_chunks(self, *a, **kw):
            return 1

        def chunk_at(self, *a, **kw):
            return chunk.model_copy(deep=True)

    r = Retriever.__new__(Retriever)
    r._local = _Store()
    picked = r.random("col")
    assert picked is not None
    assert picked.speaker == ""
    assert picked.speaker_label == ""


def test_random_retries_when_the_index_shrinks_mid_pick():
    """count_chunks and chunk_at are two reads; an out-of-process reindex
    between them must not surface as "no excerpts" on a populated collection."""
    from podcodex.rag.hit import Hit
    from podcodex.rag.retriever import Retriever

    class _ShrinkingStore:
        def __init__(self):
            self.counts = iter([1200, 900])  # first count is stale
            self.calls = 0

        def count_chunks(self, *a, **kw):
            return next(self.counts)

        def chunk_at(self, collection, offset, **kw):
            self.calls += 1
            # The stale offset misses; the re-counted one lands.
            return None if self.calls == 1 else Hit(text="ok", episode="ep1")

    r = Retriever.__new__(Retriever)
    r._local = _ShrinkingStore()
    picked = r.random("col")
    assert picked is not None and picked.text == "ok"


def test_chunk_key_distinguishes_chunks_with_the_same_start():
    """Chunks that begin in one turn, or in an untimed transcript, share a start."""
    a = Hit(show="S", episode="ep1", chunk_index=0, start=0.0)
    b = Hit(show="S", episode="ep1", chunk_index=1, start=0.0)
    assert _chunk_key(a) != _chunk_key(b)
