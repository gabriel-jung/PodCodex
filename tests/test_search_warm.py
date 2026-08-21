"""Opening the search panel warms the embedder in the background.

A cold first search costs ~6 s: torch and sentence_transformers import, then
the embedding model loads from disk. Every search after it is ~60 ms. The
frontend's SearchPanel fetches ``/api/search/stats`` when it mounts, which
is the earliest signal that a search is coming, so the load happens while
the user types instead of after they hit enter.

``/api/search/config`` is fetched on the same mount and deliberately does
*not* warm: it carries no show, so it could only guess at the model, and
guessing wrong loads a second one (bge-m3 alone is 4.3 GB) that no search
will use.
"""

from __future__ import annotations

import threading

import pytest

from podcodex.api.routes import search as search_routes
from tests.fixtures.api_client import make_client


@pytest.fixture(autouse=True)
def _reset_warm_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard is process-wide and once-only by design."""
    monkeypatch.setattr(search_routes, "_warm_started", False)


def _join_warm_threads() -> None:
    for thread in threading.enumerate():
        if thread.name == "search-warm":
            thread.join(timeout=5)


class _FakeStore:
    """Enough of the store surface for /stats to answer. The assertions are
    about which show gets warmed, not about the statistics themselves."""

    _INFO = {
        "beta__e5-small__semantic": {
            "show": "beta",
            "model": "e5-small",
            "chunker": "semantic",
        },
    }

    def get_all_collection_info(self) -> dict:
        return dict(self._INFO)

    def list_collections(self, show: str = "") -> list[str]:
        slug = show.strip().lower()
        return [n for n, i in self._INFO.items() if not slug or i["show"] == slug]

    def get_collection_info(self, collection: str) -> dict | None:
        return self._INFO.get(collection)

    def collection_summary(self, collection: str) -> dict:
        return {"episodes": 1, "chunks": 10, "sources": []}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(search_routes, "get_index_store", lambda: _FakeStore())
    return make_client(tmp_path, monkeypatch)


@pytest.fixture
def warmed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record which show gets warmed, without loading anything."""
    shows: list[str] = []
    monkeypatch.setattr(search_routes, "_warm_show_sync", shows.append)
    return shows


def test_fetching_stats_warms_that_show(client, warmed) -> None:
    response = client.get("/api/search/stats", params={"show": "Beta"})
    _join_warm_threads()

    assert response.status_code == 200
    assert warmed == ["Beta"]


def test_fetching_search_config_warms_nothing(client, warmed) -> None:
    response = client.get("/api/search/config")
    _join_warm_threads()

    assert response.status_code == 200
    assert warmed == []


def test_warming_happens_once_per_process(client, warmed) -> None:
    """The expensive half is the shared import; repeating it would only
    reload models the retriever cache already bounds."""
    for _ in range(3):
        client.get("/api/search/stats", params={"show": "Beta"})
    _join_warm_threads()

    assert warmed == ["Beta"]


def test_the_all_shows_scope_warms_nothing(client, warmed) -> None:
    """No show means no model to resolve; /stats is also called with an
    empty show by the index overview."""
    response = client.get("/api/search/stats")
    _join_warm_threads()

    assert response.status_code == 200
    assert warmed == []


def test_warm_resolves_the_model_the_way_a_search_does(monkeypatch) -> None:
    """Regression guard. A show can pin a non-default model through its RAG
    prefs, so the model has to come from the same resolver the handlers use.
    An earlier version read the raw collection list instead — and an even
    earlier one filtered it by comparing the request's show string to the
    stored ``show`` field, which are different keys, so it silently warmed
    nothing at all."""
    loaded: list[str] = []

    class _Col:
        model = "e5-large"

    import podcodex.rag.retriever as retriever_mod

    class _Retriever:
        @property
        def embedder(self):
            loaded.append("e5-large")
            return object()

    monkeypatch.setattr(search_routes, "_resolve_req_cols", lambda *a: [_Col()])
    monkeypatch.setattr(retriever_mod, "get_retriever", lambda m: _Retriever())

    search_routes._warm_show_sync("Beta")

    assert loaded == ["e5-large"]


def test_an_unindexed_show_loads_nothing(monkeypatch) -> None:
    loaded: list[str] = []
    import podcodex.rag.retriever as retriever_mod

    monkeypatch.setattr(search_routes, "_resolve_req_cols", lambda *a: [])
    monkeypatch.setattr(retriever_mod, "get_retriever", lambda m: loaded.append(m))

    search_routes._warm_show_sync("Beta")

    assert loaded == []


def test_a_failing_warm_never_escapes(monkeypatch) -> None:
    """It runs on a daemon thread and the search path rebuilds on demand, so
    a failure must cost latency and nothing else."""

    def boom(*args):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(search_routes, "_resolve_req_cols", boom)

    search_routes._warm_show_sync("Beta")  # must not raise
