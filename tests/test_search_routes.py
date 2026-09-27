"""Search routes over a real tmp index: the model fallback chain, exact
search and random quotes."""

import numpy as np
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from tests.fixtures.api_client import make_client

    return make_client(tmp_path, monkeypatch)


SEARCH_DIM = 8


ALPHA_ID = "alpha_0000abcd"


@pytest.fixture
def seeded_index(tmp_path, monkeypatch):
    """IndexStore with show "Alpha" indexed only under e5-small/semantic.

    The API's request default is bge-m3/semantic, which this fixture never
    creates for Alpha. Route tests use this gap to confirm the resolver
    chain falls through to Alpha's actual collection instead of querying a
    collection that doesn't exist.
    """
    from podcodex.rag import index_store as rag_index_store
    from podcodex.rag import retriever as rag_retriever
    from podcodex.rag.index_store import IndexStore

    from tests.fixtures.index import add_show, chunk

    index_path = tmp_path / "search-index"
    store = IndexStore(index_path)
    add_show(
        store,
        "Alpha",
        {
            "ep1": [
                chunk(
                    f"hello world chunk {i}", show="Alpha", start=float(i), end=i + 1.0
                )
                for i in range(3)
            ]
        },
        show_id=ALPHA_ID,
        model="e5-small",
        dim=SEARCH_DIM,
    )

    monkeypatch.setenv("PODCODEX_INDEX", str(index_path))
    rag_index_store.get_index_store.cache_clear()
    rag_retriever.get_retriever.cache_clear()
    # Stub the embedder so the fallback resolves against e5-small without
    # pulling live model weights; only encode_query is on the query path.
    retriever = rag_retriever.get_retriever("e5-small")
    monkeypatch.setattr(
        retriever, "encode_query", lambda _q: np.zeros(SEARCH_DIM, dtype=np.float32)
    )
    yield store
    rag_index_store.get_index_store.cache_clear()
    rag_retriever.get_retriever.cache_clear()


def test_search_falls_back_when_requested_combo_missing(client, seeded_index):
    """Requesting a model/chunking combo a show doesn't have must still
    return the show's actual results, not an empty/404-ish response."""
    resp = client.post(
        "/api/search/query",
        json={"query": "hello", "show_id": ALPHA_ID, "model": "bge-m3"},
        headers={"X-PodCodex": "1"},
    )
    assert resp.status_code == 200
    assert resp.json()  # empty if it queries a nonexistent collection


def test_exact_endpoint_falls_back_when_requested_combo_missing(client, seeded_index):
    """/exact must resolve through the same fallback chain as /query: a
    show indexed only under a non-default model must still be searchable."""
    resp = client.post(
        "/api/search/exact",
        json={"query": "hello world chunk 1", "show_id": ALPHA_ID, "model": "bge-m3"},
        headers={"X-PodCodex": "1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body
    assert any("hello world chunk 1" in r["text"] for r in body)
    result = body[0]
    assert "episode" in result
    assert "episode_stem" in result
    assert "score" in result
    assert "match_text" in result


def test_exact_search_honours_top_k(client, seeded_index):
    """The palette asks for a few hits per show, not every match."""
    resp = client.post(
        "/api/search/exact",
        json={"query": "hello world", "show_id": ALPHA_ID, "top_k": 2},
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()) == 2  # three chunks match


def test_random_endpoint_falls_back_when_requested_combo_missing(client, seeded_index):
    """/random must resolve through the same fallback chain as the other
    search routes instead of silently returning None for an indexed show."""
    resp = client.post(
        "/api/search/random",
        json={"show_id": ALPHA_ID, "model": "bge-m3"},
        headers={"X-PodCodex": "1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body is not None
    assert body["score"] == 1.0
    assert body["text"]


def test_speakers_falls_back_when_requested_combo_missing(client, seeded_index):
    """/speakers must resolve through the same fallback chain as the other
    search routes: a wrong model param must not silently return []."""
    resp = client.get(
        "/api/search/speakers",
        params={"show_id": ALPHA_ID, "model": "bge-m3"},
        headers={"X-PodCodex": "1"},
    )
    assert resp.status_code == 200
    assert resp.json() == ["Alice"]


def test_is_flagged_break_not_flagged():
    from podcodex.api.routes._helpers import is_flagged

    assert is_flagged({"speaker": "[BREAK]", "text": "", "start": 0, "end": 5}) is False


def test_is_flagged_unknown_speaker():
    from podcodex.api.routes._helpers import is_flagged

    assert (
        is_flagged({"speaker": "UNKNOWN", "text": "hi", "start": 0, "end": 1}) is True
    )


def test_is_flagged_low_density():
    from podcodex.api.routes._helpers import is_flagged

    # 3 chars over 5s = 0.6 chars/s, below threshold of 2
    assert is_flagged({"speaker": "A", "text": "hmm", "start": 0, "end": 5}) is True


def test_is_flagged_normal_segment():
    from podcodex.api.routes._helpers import is_flagged

    assert (
        is_flagged(
            {"speaker": "A", "text": "This is a normal sentence.", "start": 0, "end": 2}
        )
        is False
    )
