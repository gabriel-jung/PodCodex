"""Tests for podcodex.rag.search_service."""

import pytest

from podcodex.rag import search_service
from podcodex.rag.hit import Hit
from podcodex.rag.search_service import (
    SearchCollection,
    exact_search,
    hybrid_search,
    load_show_rag_prefs,
    random_quote,
    resolve_collections,
)


@pytest.fixture(autouse=True)
def _reset_prefs_cache():
    """load_show_rag_prefs caches by TTL; tests that monkeypatch load_config
    must not see a cached result left over from another test."""
    search_service._prefs_cache = None
    yield
    search_service._prefs_cache = None


COL_INFO = {
    "alpha__bge-m3__semantic": {
        "show": "Alpha",
        "show_id": "alpha_1",
        "model": "bge-m3",
        "chunker": "semantic",
        "dim": 1024,
        "artwork_url": "https://img/a.png",
    },
    "alpha__e5-small__speaker": {
        "show": "Alpha",
        "show_id": "alpha_1",
        "model": "e5-small",
        "chunker": "speaker",
        "dim": 384,
        "artwork_url": "https://img/a.png",
    },
    "beta__e5-small__semantic": {
        "show": "Beta",
        "show_id": "beta_2",
        "model": "e5-small",
        "chunker": "semantic",
        "dim": 384,
        "artwork_url": "",
    },
}


def test_resolve_defaults_pick_default_model():
    cols = resolve_collections(COL_INFO)
    assert [c.name for c in cols] == [
        "alpha__bge-m3__semantic",  # default model+chunker exists
        "beta__e5-small__semantic",  # default missing, first by name
    ]
    assert cols[0] == SearchCollection(
        name="alpha__bge-m3__semantic",
        model="bge-m3",
        show="Alpha",
        artwork_url="https://img/a.png",
        show_id="alpha_1",
    )


def test_resolve_show_filter_case_insensitive():
    cols = resolve_collections(COL_INFO, shows=["ALPHA"])
    assert [c.show for c in cols] == ["Alpha"]


def test_resolve_show_filter_strips_whitespace():
    cols = resolve_collections(COL_INFO, shows=["  alpha  "])
    assert [c.show for c in cols] == ["Alpha"]


def test_resolve_show_pref_beats_default():
    prefs = {"alpha_1": ("e5-small", "speaker")}
    cols = resolve_collections(COL_INFO, show_prefs=prefs)
    assert cols[0].name == "alpha__e5-small__speaker"


def test_resolve_override_beats_show_pref():
    prefs = {"alpha_1": ("e5-small", "speaker")}
    cols = resolve_collections(
        COL_INFO, show_prefs=prefs, override=("bge-m3", "semantic")
    )
    assert cols[0].name == "alpha__bge-m3__semantic"


def test_resolve_override_falls_through_when_absent():
    # Beta has no bge-m3 collection: override misses, pref misses,
    # default misses, first-by-name wins. Show stays reachable.
    cols = resolve_collections(
        COL_INFO, shows=["Beta"], override=("bge-m3", "semantic")
    )
    assert [c.name for c in cols] == ["beta__e5-small__semantic"]


def test_resolve_global_default_beats_first_by_name():
    # A mismatched caller default must fall to the global
    # DEFAULT_MODEL/DEFAULT_CHUNKING rung, not to alphabetical-first.
    col_info = {
        "s__aardvark__semantic": {
            "show": "S",
            "model": "aardvark",
            "chunker": "semantic",
        },
        "s__bge-m3__semantic": {
            "show": "S",
            "model": "bge-m3",
            "chunker": "semantic",
        },
    }
    cols = resolve_collections(col_info, default=("nope", "nope"))
    assert [c.name for c in cols] == ["s__bge-m3__semantic"]


def test_resolve_unknown_show_empty():
    assert resolve_collections(COL_INFO, shows=["Nope"]) == []


def test_resolve_empty_index():
    assert resolve_collections({}) == []


def _write_show(
    tmp_path, folder, name=None, rag_model="", rag_chunker="", show_id=None
):
    """A show.toml with id ``<folder>_1234abcd`` unless *show_id* says otherwise
    (``""`` is a show from before ids)."""
    d = tmp_path / folder
    d.mkdir()
    sid = f"{folder}_1234abcd" if show_id is None else show_id
    lines = [f'id = "{sid}"'] if sid else []
    lines += [f'name = "{name}"'] if name is not None else []
    if rag_model or rag_chunker:
        lines += ["", "[pipeline]"]
        if rag_model:
            lines.append(f'rag_model = "{rag_model}"')
        if rag_chunker:
            lines.append(f'rag_chunker = "{rag_chunker}"')
    (d / "show.toml").write_text("\n".join(lines) + "\n")
    return d


def test_rag_prefs_only_shows_with_prefs(tmp_path, monkeypatch):
    a = _write_show(tmp_path, "a", "Alpha", rag_model="e5-small")
    b = _write_show(tmp_path, "b", "Beta")  # no pref: omitted

    class Cfg:
        show_folders = [str(a), str(b)]

    monkeypatch.setattr("podcodex.rag.search_service.load_config", lambda: Cfg())
    prefs = load_show_rag_prefs()
    assert prefs == {"a_1234abcd": ("e5-small", "semantic")}  # blank half filled


@pytest.mark.legacy("show-id")
def test_rag_prefs_nameless_show_keyed_by_folder(tmp_path, monkeypatch):
    # show.toml with prefs but no name line: key falls back to folder basename
    d = _write_show(tmp_path, "MyFolder", rag_model="e5-small", show_id="")

    class Cfg:
        show_folders = [str(d)]

    monkeypatch.setattr("podcodex.rag.search_service.load_config", lambda: Cfg())
    prefs = load_show_rag_prefs()
    assert prefs == {"myfolder": ("e5-small", "semantic")}


def test_rag_prefs_bad_folder_swallowed(tmp_path, monkeypatch):
    class Cfg:
        show_folders = [str(tmp_path / "missing")]

    monkeypatch.setattr("podcodex.rag.search_service.load_config", lambda: Cfg())
    assert load_show_rag_prefs() == {}


def test_rag_prefs_cached_within_ttl(tmp_path, monkeypatch):
    a = _write_show(tmp_path, "a", "Alpha", rag_model="e5-small")

    class Cfg:
        show_folders = [str(a)]

    calls = {"n": 0}

    def fake_load_config():
        calls["n"] += 1
        return Cfg()

    monkeypatch.setattr("podcodex.rag.search_service.load_config", fake_load_config)

    first = load_show_rag_prefs()
    second = load_show_rag_prefs()
    assert first == second == {"a_1234abcd": ("e5-small", "semantic")}
    assert calls["n"] == 1  # second call served from cache, no reload

    monkeypatch.setattr(search_service, "_prefs_cache", None)
    load_show_rag_prefs()
    assert calls["n"] == 2  # cache invalidated: recomputes


# Orchestration functions

ALPHA_COL = SearchCollection("alpha__bge-m3__semantic", "bge-m3", "Alpha")
BETA_COL = SearchCollection("beta__e5-small__semantic", "e5-small", "Beta")


class StubRetriever:
    def __init__(self, hits_by_col, calls):
        self._hits = hits_by_col
        self.calls = calls

    def encode_query(self, query):
        self.calls.append(("encode", query))
        return "QV"

    def retrieve(self, query, collection, *, query_vector=None, **kw):
        self.calls.append(("retrieve", collection, query_vector))
        return list(self._hits.get(collection, []))

    def exact(self, query, collection, **kw):
        return list(self._hits.get(collection, []))

    def random(self, collection, **kw):
        hits = self._hits.get(collection, [])
        return hits[0] if hits else None


def _factory(hits_by_col, calls):
    retrievers = {}

    def factory(model):
        if model not in retrievers:
            retrievers[model] = StubRetriever(hits_by_col, calls)
        return retrievers[model]

    return factory


def test_hybrid_encodes_once_per_model_and_merges():
    calls = []
    hits = {
        "alpha__bge-m3__semantic": [Hit(text="a", score=0.9)],
        "beta__e5-small__semantic": [Hit(text="b", score=0.8)],
    }
    out = hybrid_search(
        "q",
        [ALPHA_COL, BETA_COL],
        top_k=5,
        retriever_factory=_factory(hits, calls),
    )
    assert [c.text for c, _ in out] == ["a", "b"]  # roundrobin, score order
    assert calls.count(("encode", "q")) == 2  # one per model, not per collection
    assert ("retrieve", "alpha__bge-m3__semantic", "QV") in calls


def test_hybrid_score_floor_drops_noise():
    hits = {
        "alpha__bge-m3__semantic": [
            Hit(text="keep", score=0.5),
            Hit(text="noise", score=0.01),
        ]
    }
    out = hybrid_search(
        "q",
        [ALPHA_COL],
        top_k=5,
        score_floor=0.05,
        retriever_factory=_factory(hits, []),
    )
    assert [c.text for c, _ in out] == ["keep"]


def test_hybrid_reraises_value_error():
    class Boom(StubRetriever):
        def retrieve(self, *a, **k):
            raise ValueError("dim mismatch")

    def factory(model):
        return Boom({}, [])

    with pytest.raises(ValueError):
        hybrid_search("q", [ALPHA_COL], top_k=5, retriever_factory=factory)


def test_hybrid_skips_broken_collection():
    class Flaky(StubRetriever):
        def retrieve(self, query, collection, **kw):
            if collection.startswith("alpha"):
                raise RuntimeError("lance io")
            return [Hit(text="b", score=0.8)]

    def factory(model):
        return Flaky({}, [])

    out = hybrid_search("q", [ALPHA_COL, BETA_COL], top_k=5, retriever_factory=factory)
    assert [c.text for c, _ in out] == ["b"]


def test_hybrid_skips_a_model_that_cannot_load():
    hits = {"beta__e5-small__semantic": [Hit(text="b", score=0.8)]}
    good = _factory(hits, [])

    def factory(model):
        if model == "bge-m3":
            raise OSError("weights missing")
        return good(model)

    out = hybrid_search("q", [ALPHA_COL, BETA_COL], top_k=5, retriever_factory=factory)
    assert [c.text for c, _ in out] == ["b"]


def test_hybrid_raises_when_every_collection_failed():
    def factory(model):
        raise OSError("weights missing")

    with pytest.raises(OSError):
        hybrid_search("q", [ALPHA_COL], top_k=5, retriever_factory=factory)


def test_exact_chronological_order_bot_semantics():
    hits = {
        "alpha__bge-m3__semantic": [
            Hit(text="t3", score=0.6, fuzzy_match=True),
            Hit(text="t1", score=1.0, episode="ep1", start=5.0),
            Hit(text="t2", score=0.8, fuzzy_match=True),
            Hit(text="t0", score=1.0, episode="ep0", start=9.0),
        ]
    }
    out = exact_search(
        "q",
        [ALPHA_COL],
        order="chronological",
        retriever_factory=_factory(hits, []),
    )
    # phrase hits chronological by (episode, start), fuzzy after, by score desc
    assert [c.text for c, _ in out] == ["t0", "t1", "t2", "t3"]


def test_exact_positional_keeps_retriever_order():
    hits = {"alpha__bge-m3__semantic": [Hit(text="x"), Hit(text="y")]}
    out = exact_search("q", [ALPHA_COL], retriever_factory=_factory(hits, []))
    assert [c.text for c, _ in out] == ["x", "y"]


def test_random_returns_chunk_and_collection():
    hits = {"alpha__bge-m3__semantic": [Hit(text="r")]}
    got = random_quote([ALPHA_COL], retriever_factory=_factory(hits, []))
    assert got is not None
    assert (got[0].text, got[1]) == ("r", "alpha__bge-m3__semantic")
    assert random_quote([], retriever_factory=_factory({}, [])) is None


def test_random_raises_when_every_collection_failed():
    """An index fault is not "no excerpts": with nothing answered, it raises."""

    class Broken(StubRetriever):
        def random(self, collection, **kw):
            raise RuntimeError("lance io")

    def factory(model):
        return Broken({}, [])

    with pytest.raises(RuntimeError):
        random_quote([ALPHA_COL], retriever_factory=factory)


def test_random_tries_every_collection_until_one_matches():
    """A filter that matches only one show still finds it, whichever
    collection the shuffle tries first."""
    hits = {"beta__e5-small__semantic": [Hit(text="b")]}
    for _ in range(10):
        out = random_quote([ALPHA_COL, BETA_COL], retriever_factory=_factory(hits, []))
        assert out is not None and out[1] == "beta__e5-small__semantic"


def test_random_skips_a_broken_collection():
    class Flaky(StubRetriever):
        def random(self, collection, **kw):
            if collection.startswith("alpha"):
                raise RuntimeError("lance io")
            return Hit(text="b")

    def factory(model):
        return Flaky({}, [])

    for _ in range(10):
        out = random_quote([ALPHA_COL, BETA_COL], retriever_factory=factory)
        assert out is not None and out[0].text == "b"


def test_random_reraises_value_error():
    class Boom(StubRetriever):
        def random(self, collection, **kw):
            raise ValueError("bad filter")

    def factory(model):
        return Boom({}, [])

    with pytest.raises(ValueError):
        random_quote([ALPHA_COL], retriever_factory=factory)


def test_hybrid_same_model_encodes_once():
    calls = []
    alpha2 = SearchCollection("gamma__bge-m3__semantic", "bge-m3", "Gamma")
    hits = {
        "alpha__bge-m3__semantic": [Hit(text="a", score=0.9)],
        "gamma__bge-m3__semantic": [Hit(text="g", score=0.7)],
    }
    hybrid_search(
        "q",
        [ALPHA_COL, alpha2],
        top_k=5,
        retriever_factory=_factory(hits, calls),
    )
    assert calls.count(("encode", "q")) == 1  # shared model, one encoding
    assert ("retrieve", "alpha__bge-m3__semantic", "QV") in calls
    assert ("retrieve", "gamma__bge-m3__semantic", "QV") in calls


def test_exact_skips_broken_collection():
    class Flaky(StubRetriever):
        def exact(self, query, collection, **kw):
            if collection.startswith("alpha"):
                raise RuntimeError("lance io")
            return [Hit(text="b")]

    def factory(model):
        return Flaky({}, [])

    out = exact_search("q", [ALPHA_COL, BETA_COL], retriever_factory=factory)
    assert [c.text for c, _ in out] == ["b"]


def test_exact_positional_preserves_input_collection_order():
    # Interleaved models (A, B, A): grouping by model buckets the two A
    # collections together, pulling the middle B collection to the end.
    # Positional order must match resolve order (input order) instead.
    x_col = SearchCollection("x__a-model__semantic", "a-model", "X")
    y_col = SearchCollection("y__b-model__semantic", "b-model", "Y")
    z_col = SearchCollection("z__a-model__semantic", "a-model", "Z")
    hits = {
        "x__a-model__semantic": [Hit(text="x")],
        "y__b-model__semantic": [Hit(text="y")],
        "z__a-model__semantic": [Hit(text="z")],
    }
    out = exact_search("q", [x_col, y_col, z_col], retriever_factory=_factory(hits, []))
    assert [c.text for c, _ in out] == ["x", "y", "z"]


def test_exact_reraises_value_error():
    class Boom(StubRetriever):
        def exact(self, *a, **k):
            raise ValueError("bad filter")

    def factory(model):
        return Boom({}, [])

    with pytest.raises(ValueError):
        exact_search("q", [ALPHA_COL], retriever_factory=factory)


def test_exact_raises_when_every_collection_failed():
    """A broken FTS index must not read as "the phrase is absent"."""

    class Broken(StubRetriever):
        def exact(self, *a, **k):
            raise RuntimeError("no INVERTED index")

    def factory(model):
        return Broken({}, [])

    with pytest.raises(RuntimeError):
        exact_search("q", [ALPHA_COL], retriever_factory=factory)


def test_exact_chronological_word_matches_before_superstring():
    hits = {
        "alpha__bge-m3__semantic": [
            Hit(text="sup", score=0.99, episode="ep0", start=1.0),
            Hit(text="word", score=1.0, episode="ep9", start=9.0),
        ]
    }
    out = exact_search(
        "q",
        [ALPHA_COL],
        order="chronological",
        retriever_factory=_factory(hits, []),
    )
    # Word match ranks first despite the later episode.
    assert [c.text for c, _ in out] == ["word", "sup"]


def test_exact_search_limit_stops_at_the_cap():
    from podcodex.rag.search_service import SearchCollection, exact_search

    searched: list[str] = []

    class _Ret:
        def exact(self, _q, name, **_k):
            searched.append(name)
            return [f"{name}-{i}" for i in range(3)]

    cols = [SearchCollection(name=n, model="m", show="S") for n in ("a", "b", "c")]
    out = exact_search("q", cols, limit=4, retriever_factory=lambda _m: _Ret())
    assert [h for h, _c in out] == ["a-0", "a-1", "a-2", "b-0"]
    assert searched == ["a", "b"]


_FILTERS = dict(
    episode="ep1",
    episodes=["ep1", "ep2"],
    source="corrected",
    speaker="Alice",
    pub_date_min="2024-01-01",
    pub_date_max="2024-12-31",
)


@pytest.mark.parametrize("surface", ["hybrid", "exact", "random"])
def test_every_filter_reaches_the_retriever(surface):
    """A filter dropped on the way leaves results the user excluded."""
    seen: list[dict] = []

    class Spy(StubRetriever):
        def retrieve(self, query, collection, *, query_vector=None, **kw):
            seen.append(kw)
            return []

        def exact(self, query, collection, **kw):
            seen.append(kw)
            return []

        def random(self, collection, **kw):
            seen.append(kw)
            return None

    def factory(model):
        return Spy({}, [])

    if surface == "hybrid":
        hybrid_search("q", [ALPHA_COL], top_k=5, retriever_factory=factory, **_FILTERS)
    elif surface == "exact":
        exact_search("q", [ALPHA_COL], retriever_factory=factory, **_FILTERS)
    else:
        random_quote([ALPHA_COL], retriever_factory=factory, **_FILTERS)

    assert seen and all(kw.items() >= _FILTERS.items() for kw in seen)


def test_a_fan_out_reads_the_index_metadata_once(tmp_path, monkeypatch):
    """Per-collection table listings and directory stats made a search's
    fixed cost grow with the square of the collection count."""
    import numpy as np

    from podcodex.rag.index_store import IndexStore
    from podcodex.rag.retriever import Retriever
    from tests.fixtures.index import add_show, chunk

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "test")
    store = IndexStore(tmp_path / "index")
    cols = [
        SearchCollection(
            add_show(
                store, f"S{i}", {"ep": [chunk("giraffes talk", show=f"S{i}")]}, dim=8
            ),
            "bge-m3",
            f"S{i}",
        )
        for i in range(6)
    ]

    class _Emb:
        def encode_query(self, _q):
            return np.ones(8, dtype=np.float32)

    ret = Retriever("bge-m3", local=store)
    ret._embedder = _Emb()
    hybrid_search("giraffes", cols, top_k=5, retriever_factory=lambda _m: ret)  # warm

    listings = []
    real = store._db.list_tables
    monkeypatch.setattr(
        store._db, "list_tables", lambda *a: listings.append(1) or real(*a)
    )
    stats = []
    real_mtime = store.index_mtime
    monkeypatch.setattr(store, "index_mtime", lambda: stats.append(1) or real_mtime())

    out = hybrid_search("giraffes", cols, top_k=5, retriever_factory=lambda _m: ret)
    exact_search("giraffes", cols, retriever_factory=lambda _m: ret)

    assert len({c for _, c in out}) == 5  # top_k across shows, still answered
    assert len(listings) <= 2 and len(stats) <= 2  # once per fan-out


def test_a_collection_of_an_unregistered_model_is_skipped():
    """The index can outlive the model registry: a stale model key skips its
    collections instead of failing every show's search."""
    good = _factory({"alpha__bge-m3__semantic": [Hit(text="a", score=0.9)]}, [])

    def factory(model):
        if model == "e5-small":
            raise ValueError("Unknown model 'e5-small'")
        return good(model)

    cols = [ALPHA_COL, BETA_COL]
    assert [
        c.text for c, _ in hybrid_search("q", cols, top_k=5, retriever_factory=factory)
    ] == ["a"]
    assert [c.text for c, _ in exact_search("q", cols, retriever_factory=factory)] == [
        "a"
    ]
    assert random_quote([BETA_COL, ALPHA_COL], retriever_factory=factory)[0].text == "a"
