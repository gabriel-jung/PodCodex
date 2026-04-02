"""Tests for podcodex.rag.store (collection_name) and new LocalStore bulk methods."""

import numpy as np

from podcodex.rag.localstore import LocalStore
from podcodex.rag.store import collection_name


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_local() -> LocalStore:
    return LocalStore(db_path=":memory:")


def _seed_collection(
    local: LocalStore,
    col: str = "test__bge-m3__semantic",
    show: str = "test",
    episodes: dict[str, int] | None = None,
    dim: int = 4,
) -> None:
    """Create a collection and seed it with dummy chunks."""
    local.ensure_collection(col, show=show, model="bge-m3", chunker="semantic", dim=dim)
    if episodes is None:
        episodes = {"ep1": 3, "ep2": 2}
    for ep, n in episodes.items():
        chunks = [
            {
                "episode": ep,
                "show": show,
                "start": float(i),
                "end": float(i + 1),
                "speaker": "Alice" if i % 2 == 0 else "Bob",
                "dominant_speaker": "Alice" if i % 2 == 0 else "Bob",
                "source": "polished" if i == 0 else "transcript",
                "text": f"chunk {i} of {ep}",
            }
            for i in range(n)
        ]
        embeddings = np.random.rand(n, dim).astype(np.float32)
        local.save_chunks(col, ep, chunks, embeddings)


# ──────────────────────────────────────────────
# collection_name
# ──────────────────────────────────────────────


def test_collection_name_basic():
    assert collection_name("my_podcast", "bge-m3") == "my_podcast__bge-m3__semantic"


def test_collection_name_with_chunker():
    assert (
        collection_name("my_podcast", "bge-m3", "speaker")
        == "my_podcast__bge-m3__speaker"
    )


def test_collection_name_normalizes_spaces():
    assert collection_name("My Podcast", "e5-small") == "my_podcast__e5-small__semantic"


def test_collection_name_normalizes_mixed_case():
    assert collection_name("MyPodcast", "e5-large") == "mypodcast__e5-large__semantic"


def test_collection_name_normalizes_special_chars():
    assert collection_name("My Podcast!", "bge-m3") == "my_podcast__bge-m3__semantic"


def test_collection_name_idempotent():
    assert collection_name("my_podcast", "bge-m3") == collection_name(
        "My Podcast", "bge-m3"
    )


# ──────────────────────────────────────────────
# LocalStore.collection_chunk_count
# ──────────────────────────────────────────────


def test_collection_chunk_count():
    local = _make_local()
    col = "test__bge-m3__semantic"
    _seed_collection(local, col, episodes={"ep1": 3, "ep2": 2})
    assert local.collection_chunk_count(col) == 5


def test_collection_chunk_count_empty():
    local = _make_local()
    assert local.collection_chunk_count("nonexistent") == 0


# ──────────────────────────────────────────────
# LocalStore.load_all_chunks
# ──────────────────────────────────────────────


def test_load_all_chunks():
    local = _make_local()
    col = "test__bge-m3__semantic"
    _seed_collection(local, col, episodes={"ep1": 3, "ep2": 2})
    chunks = local.load_all_chunks(col)
    assert len(chunks) == 5
    assert all("text" in c for c in chunks)
    assert all("embedding" not in c for c in chunks)


def test_load_all_chunks_with_episode_filter():
    local = _make_local()
    col = "test__bge-m3__semantic"
    _seed_collection(local, col, episodes={"ep1": 3, "ep2": 2})
    chunks = local.load_all_chunks(col, episode="ep1")
    assert len(chunks) == 3
    assert all(c.get("episode") == "ep1" for c in chunks)


def test_load_all_chunks_empty():
    local = _make_local()
    assert local.load_all_chunks("nonexistent") == []


# ──────────────────────────────────────────────
# LocalStore.load_all_vectors
# ──────────────────────────────────────────────


def test_load_all_vectors():
    local = _make_local()
    col = "test__bge-m3__semantic"
    dim = 4
    _seed_collection(local, col, episodes={"ep1": 3, "ep2": 2}, dim=dim)
    matrix, chunks = local.load_all_vectors(col)
    assert matrix.shape == (5, dim)
    assert matrix.dtype == np.float32
    assert len(chunks) == 5
    assert all("text" in c for c in chunks)


def test_load_all_vectors_empty():
    local = _make_local()
    matrix, chunks = local.load_all_vectors("nonexistent")
    assert matrix.shape == (0, 0)
    assert chunks == []


# ──────────────────────────────────────────────
# LocalStore.list_sources
# ──────────────────────────────────────────────


def test_list_sources():
    local = _make_local()
    col = "test__bge-m3__semantic"
    _seed_collection(local, col)
    sources = local.list_sources(col)
    assert "polished" in sources
    assert "transcript" in sources


def test_list_sources_empty():
    local = _make_local()
    assert local.list_sources("nonexistent") == []


# ──────────────────────────────────────────────
# LocalStore.list_speakers
# ──────────────────────────────────────────────


def test_list_speakers():
    local = _make_local()
    col = "test__bge-m3__semantic"
    _seed_collection(local, col)
    speakers = local.list_speakers(col)
    assert "Alice" in speakers
    assert "Bob" in speakers


def test_list_speakers_empty():
    local = _make_local()
    assert local.list_speakers("nonexistent") == []


# ──────────────────────────────────────────────
# LocalStore.get_episode_stats
# ──────────────────────────────────────────────


def test_get_episode_stats():
    local = _make_local()
    col = "test__bge-m3__semantic"
    _seed_collection(local, col, episodes={"ep1": 3, "ep2": 2})
    stats = local.get_episode_stats(col)
    assert len(stats) == 2
    ep1 = next(s for s in stats if s["episode"] == "ep1")
    assert ep1["chunk_count"] == 3
    assert "Alice" in ep1["speakers"]


def test_get_episode_stats_empty():
    local = _make_local()
    assert local.get_episode_stats("nonexistent") == []
