"""Deleting a show must take its index rows with it, not orphan them."""

from __future__ import annotations

import numpy as np
import pytest

from podcodex.ingest.show import ShowMeta, ensure_show_id, save_show_meta
from podcodex.rag import index_store as rag_index_store

DIM = 8


@pytest.fixture
def show_with_index(tmp_path, monkeypatch):
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    monkeypatch.setenv("PODCODEX_INDEX", str(tmp_path / "index"))
    rag_index_store.get_index_store.cache_clear()

    folder = tmp_path / "shows" / "My Show"
    folder.mkdir(parents=True)
    save_show_meta(folder, ShowMeta(name="My Show"))
    sid = ensure_show_id(folder)

    store = rag_index_store.get_index_store()
    col = store.ensure_collection_for_show(sid, "My Show", "bge-m3", "semantic", DIM)
    store.save_chunks(
        col,
        "ep1",
        [
            {
                "text": "hi",
                "episode": "ep1",
                "show": "My Show",
                "source": "transcript",
                "dominant_speaker": "sp",
                "start": 0.0,
                "end": 1.0,
            }
        ],
        np.random.default_rng(0).random((1, DIM), dtype=np.float32),
    )
    store.set_show_password(sid, "sha256:abc", show_label="My Show")

    import podcodex.core.app_config as app_config

    cfg = app_config.AppConfig()
    cfg.show_folders = [str(folder)]
    monkeypatch.setattr(app_config, "load_config", lambda: cfg)

    yield folder, sid, store
    rag_index_store.get_index_store.cache_clear()


def test_purge_removes_collections_and_password(show_with_index):
    from podcodex.api.routes.shows import _purge_show_from_index

    folder, sid, store = show_with_index
    assert store.collections_for_show(sid)

    collections, password = _purge_show_from_index(sid)

    assert collections == 1
    assert password is True
    assert store.collections_for_show(sid) == []
    assert sid not in store.get_show_password_entries()


def test_purge_is_a_no_op_without_an_id(show_with_index):
    from podcodex.api.routes.shows import _purge_show_from_index

    _folder, sid, store = show_with_index
    assert _purge_show_from_index("") == (0, False)
    assert store.collections_for_show(sid)


def test_purge_finds_collections_that_predate_the_migration(show_with_index):
    """A show deleted before its rows were stamped must still be cleaned up."""
    from podcodex.api.routes.shows import _purge_show_from_index

    _folder, sid, store = show_with_index
    col = store.collections_for_show(sid)[0]
    store.set_collection_identity(col, show_id="", show="My Show")

    collections, _password = _purge_show_from_index(sid, "My Show")

    assert collections == 1
    assert store.collections_for_show(sid, show_label="My Show") == []


def test_purge_leaves_other_shows_alone(show_with_index):
    from podcodex.api.routes.shows import _purge_show_from_index

    _folder, sid, store = show_with_index
    store.ensure_collection_for_show(
        "other_9999abcd", "Other", "bge-m3", "semantic", DIM
    )

    _purge_show_from_index(sid)

    assert store.collections_for_show("other_9999abcd")


def test_purge_removes_a_password_that_predates_the_migration(show_with_index):
    """A name-keyed password row must not outlive the show it protected."""
    from podcodex.api.routes.shows import _purge_show_from_index

    _folder, sid, store = show_with_index
    store.delete_show_password(sid)
    store._passwords_table().add(
        [
            {
                "show_id": "",
                "show_label": "",
                "show": "My Show",
                "password_hash": "sha256:abc",
            }
        ]
    )
    assert "My Show" in store.get_show_password_entries()

    _collections, password = _purge_show_from_index(sid, "My Show")

    assert password is True
    assert store.get_show_password_entries() == {}
