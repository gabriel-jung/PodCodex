"""A show's identity in the index: its collections and password follow its
``show.toml`` id through renames, and the label the bot reads is healed from
``show.toml`` on the app side only."""

from __future__ import annotations

import pytest

from podcodex.ingest.show import ShowMeta, save_show_meta, show_id
from podcodex.rag.index_store import IndexStore
from tests.fixtures.index import DIM, add_show
from tests.fixtures.show_resolver import show_folder_resolver


@pytest.fixture
def env(tmp_path, monkeypatch):
    """One registered show, "My Show", indexed under its own id.

    Runs with no show-folder resolver (the bot's situation) unless a test
    installs one; restores whatever the rest of the suite had on the way out.
    """
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    folder = tmp_path / "shows" / "My Show"
    folder.mkdir(parents=True)
    save_show_meta(folder, ShowMeta(name="My Show"))

    store = IndexStore(tmp_path / "index")
    add_show(store, "My Show", show_id=show_id(folder))

    import podcodex.core.app_config as app_config

    cfg = app_config.AppConfig()
    cfg.show_folders = [str(folder)]
    monkeypatch.setattr(app_config, "load_config", lambda: cfg)

    with show_folder_resolver(None):
        yield store, folder


def test_indexing_again_reuses_the_shows_table(env):
    """A second table for the same show would silently split its index."""
    store, folder = env
    sid = show_id(folder)

    name = store.ensure_collection_for_show(sid, "My Show", "bge-m3", "semantic", DIM)

    assert name == "my_show__bge-m3__semantic"
    assert len(store.collections_for_show(sid)) == 1


def test_a_new_combo_is_created_under_the_id_and_survives_a_rename(env):
    """Identity is minted when show.toml is saved, so a collection created in
    the same session already carries it and a rename cannot orphan it."""
    store, folder = env
    sid = show_id(folder)
    assert sid, "saving show.toml must establish identity"

    col = store.ensure_collection_for_show(sid, "My Show", "bge-m3", "sentence", DIM)
    save_show_meta(folder, ShowMeta(id=sid, name="Renamed"))
    assert store.set_show_label(sid, "Renamed", previous_label="My Show") >= 1

    assert store.resolve_collection(sid, "bge-m3", "sentence") == col
    assert store.collection_label(col) == "Renamed"


def test_rename_propagates_the_label_into_the_index(env):
    """The bot has no show.toml, so the new name has to travel in the index."""
    store, folder = env
    sid = show_id(folder)

    assert store.set_show_label(sid, "Brand New Name") == 1
    col = store.resolve_collection(sid, "bge-m3", "semantic")
    assert store.get_collection_info(col)["show"] == "Brand New Name"
    # And a bot with only a label still finds it.
    assert (
        store.resolve_collection("", "bge-m3", "semantic", show_label="Brand New Name")
        == col
    )


def test_rename_relabels_hits_without_touching_chunks(env):
    store, folder = env
    sid = show_id(folder)
    col = store.resolve_collection(sid, "bge-m3", "semantic")

    store.set_show_label(sid, "Brand New Name")

    assert [h.show for h in store.load_chunks_no_embeddings(col, "ep1")] == [
        "Brand New Name"
    ]


def test_password_survives_a_rename(env):
    store, folder = env
    sid = show_id(folder)
    store.set_show_password(sid, "sha256:abc", show_label="My Show")

    store.set_show_label(sid, "Brand New Name")
    store.set_show_password(sid, "sha256:abc", show_label="Brand New Name")

    entries = store.get_show_password_entries()
    assert list(entries) == [sid]
    assert entries[sid]["label"] == "Brand New Name"


def test_two_shows_with_one_label_keep_their_own_passwords(env):
    store, _folder = env
    store.set_show_password("show_a", "sha256:a", show_label="Twin")
    store.set_show_password("show_b", "sha256:b", show_label="Twin")

    entries = store.get_show_password_entries()
    assert entries["show_a"]["password_hash"] == "sha256:a"
    assert entries["show_b"]["password_hash"] == "sha256:b"


def test_label_heals_on_read_after_an_offline_rename(env):
    """A rename that never went through the API (hand-edited show.toml, an
    import, the index offline) still reaches the index on the next read."""
    store, folder = env
    sid = show_id(folder)
    col = store.resolve_collection(sid, "bge-m3", "semantic")
    assert store.collection_label(col) == "My Show"

    save_show_meta(folder, ShowMeta(id=sid, name="Edited By Hand"))
    # A resolver that never matches by label: the heal must find the folder
    # by id, which is the whole point of identity-first reconciliation.
    with show_folder_resolver(lambda _name: None):
        store._collection_info_cache = None
        assert store.collection_label(col) == "Edited By Hand"


def test_heal_is_a_no_op_without_show_folders(env):
    """The bot reads an rsynced index and must serve what is stored."""
    store, folder = env
    sid = show_id(folder)
    col = store.resolve_collection(sid, "bge-m3", "semantic")

    save_show_meta(folder, ShowMeta(id=sid, name="Renamed"))
    store._collection_info_cache = None

    # No resolver registered: this is the bot, and nothing heals.
    assert store.collection_label(col) == "My Show"
