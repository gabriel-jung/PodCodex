"""The app's search, episode and index routes resolve a show by its id.

Two registered shows share one display name here, which is exactly what a
name-keyed route could not tell apart: it resolved the oldest folder and
answered from that show's collections for both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from podcodex.core.app_config import AppConfig  # noqa: E402
from podcodex.rag import index_store as rag_index_store  # noqa: E402
from podcodex.rag import retriever as rag_retriever  # noqa: E402
from podcodex.rag.index_store import IndexStore  # noqa: E402
from tests.fixtures.api_client import make_client  # noqa: E402

DIM = 8
ID_A = "twin_aaaaaaaa"
ID_B = "twin_bbbbbbbb"


def _show(root: Path, folder: str, show_id: str, name: str, stem: str) -> Path:
    path = root / folder
    path.mkdir(parents=True)
    (path / "show.toml").write_text(
        f'id = "{show_id}"\nname = "{name}"\n', encoding="utf-8"
    )
    (path / f"{stem}.mp3").write_bytes(b"fake")
    return path


def _index(store: IndexStore, col: str, show_id: str, stem: str, text: str) -> None:
    store.ensure_collection(
        col,
        show="Twin",
        model="bge-m3",
        chunker="semantic",
        dim=DIM,
        show_id=show_id,
    )
    chunk = {
        "text": text,
        "episode": stem,
        "show": "Twin",
        "source": "transcript",
        "dominant_speaker": "Alice",
        "start": 0.0,
        "end": 1.0,
    }
    rng = np.random.default_rng(0)
    store.save_chunks(col, stem, [chunk], rng.random((1, DIM), dtype=np.float32))


@pytest.fixture
def twins(tmp_path, monkeypatch):
    """Two registered shows called "Twin", each indexed with its own episode."""
    shows = tmp_path / "shows"
    a = _show(shows, "a", ID_A, "Twin", "ep_a")
    b = _show(shows, "b", ID_B, "Twin", "ep_b")

    index_path = tmp_path / "index"
    store = IndexStore(index_path)
    _index(store, "twin_a__bge-m3__semantic", ID_A, "ep_a", "apple pie recipe")
    _index(store, "twin_b__bge-m3__semantic", ID_B, "ep_b", "banana split recipe")
    monkeypatch.setenv("PODCODEX_INDEX", str(index_path))
    client = make_client(
        tmp_path, monkeypatch, config=AppConfig(show_folders=[str(a), str(b)])
    )
    return client, a, b


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Retrievers are cached per model and hold the store they were built
    with, so one left over from another test searches that test's index."""
    rag_index_store.get_index_store.cache_clear()
    rag_retriever.get_retriever.cache_clear()
    yield
    rag_index_store.get_index_store.cache_clear()
    rag_retriever.get_retriever.cache_clear()


def _exact(client, show_id: str) -> list[dict]:
    r = client.post("/api/search/exact", json={"query": "recipe", "show_id": show_id})
    assert r.status_code == 200, r.text
    return r.json()


def test_search_answers_from_the_requested_show_only(twins):
    client, a, b = twins

    hits_a = _exact(client, ID_A)
    hits_b = _exact(client, ID_B)

    assert [h["text"] for h in hits_a] == ["apple pie recipe"]
    assert [h["text"] for h in hits_b] == ["banana split recipe"]
    # Hits map back to files through the searched show's own folder.
    assert hits_a[0]["audio_path"] == str(a / "ep_a.mp3")
    assert hits_b[0]["output_dir"] == str(b / "ep_b")


def test_search_survives_a_rename(twins):
    client, a, _b = twins
    (a / "show.toml").write_text(f'id = "{ID_A}"\nname = "Renamed"\n', encoding="utf-8")

    assert [h["text"] for h in _exact(client, ID_A)] == ["apple pie recipe"]


def test_an_unknown_id_finds_nothing(twins):
    client, _a, _b = twins

    assert _exact(client, "nobody_00000000") == []


def test_episode_list_is_per_show(twins):
    client, _a, _b = twins

    r = client.get("/api/episodes/list", params={"show_id": ID_B})

    assert r.status_code == 200
    assert [e["episode"] for e in r.json()] == ["ep_b"]


def test_index_routes_take_the_show_from_the_episode_folder(twins):
    client, a, _b = twins

    r = client.get(
        "/api/index/episode-collections", params={"audio_path": str(a / "ep_a.mp3")}
    )

    assert r.status_code == 200
    assert [c["collection"] for c in r.json()] == ["twin_a__bge-m3__semantic"]


def test_a_collection_without_an_id_is_reached_through_its_label(tmp_path, monkeypatch):
    """Rows written before ids existed carry only the display name."""
    folder = _show(tmp_path / "shows", "legacy", "legacy_11111111", "Legacy", "ep1")
    index_path = tmp_path / "index"
    store = IndexStore(index_path)
    store.ensure_collection(
        "legacy__bge-m3__semantic",
        show="Legacy",
        model="bge-m3",
        chunker="semantic",
        dim=DIM,
    )
    chunk = {
        "text": "old recipe",
        "episode": "ep1",
        "show": "Legacy",
        "source": "transcript",
        "dominant_speaker": "Alice",
        "start": 0.0,
        "end": 1.0,
    }
    rng = np.random.default_rng(0)
    store.save_chunks(
        "legacy__bge-m3__semantic",
        "ep1",
        [chunk],
        rng.random((1, DIM), dtype=np.float32),
    )
    monkeypatch.setenv("PODCODEX_INDEX", str(index_path))
    client = make_client(
        tmp_path, monkeypatch, config=AppConfig(show_folders=[str(folder)])
    )

    assert [h["text"] for h in _exact(client, "legacy_11111111")] == ["old recipe"]


def test_a_folder_without_an_id_does_not_claim_a_same_named_show(twins, tmp_path):
    """An id-less folder matches rows by label, but a row with an id belongs
    to the show that has that id, whatever it is called."""
    client, _a, _b = twins
    loose = tmp_path / "shows" / "loose"
    loose.mkdir()
    (loose / "show.toml").write_text('name = "Twin"\n', encoding="utf-8")
    (loose / "ep_a.mp3").write_bytes(b"fake")

    r = client.get(
        "/api/index/episode-collections", params={"audio_path": str(loose / "ep_a.mp3")}
    )

    assert r.status_code == 200
    assert r.json() == []


def test_startup_gives_every_registered_show_an_id(tmp_path, monkeypatch):
    """Old shows that were never indexed had no id, and each "no id" case
    needed its own handling in search and bot access."""
    import podcodex.core.app_config as app_config
    import podcodex.ingest.show as show_mod
    from podcodex.ingest.show import load_show_meta
    from podcodex.ingest.show_registry import mint_missing_show_ids

    old = tmp_path / "old"
    old.mkdir()
    (old / "show.toml").write_text('name = "Old"\n', encoding="utf-8")
    minted = _show(tmp_path, "new", "new_12345678", "New", "ep")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "show.toml").write_text('name = "Locked"\n', encoding="utf-8")
    cfg = AppConfig(show_folders=[str(old), str(minted), str(locked)])
    monkeypatch.setattr(app_config, "load_config", lambda: cfg)

    real = show_mod.ensure_show_id

    def read_only(folder):
        if folder == locked:
            raise PermissionError("read-only")
        return real(folder)

    monkeypatch.setattr(show_mod, "ensure_show_id", read_only)

    assert mint_missing_show_ids() == 1
    assert load_show_meta(old).id
    assert load_show_meta(minted).id == "new_12345678"
    assert not load_show_meta(locked).id  # logged and skipped, not fatal
