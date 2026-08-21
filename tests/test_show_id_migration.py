"""Tests for the one-time name-keyed to id-keyed index migration."""

from __future__ import annotations

import numpy as np
import pytest

from podcodex.ingest.show import ShowMeta, save_show_meta, show_id
from podcodex.rag.index_store import IndexStore
from podcodex.rag.show_id_migration import migrate_index_to_show_ids
from tests.fixtures.show_resolver import show_folder_resolver

DIM = 8


def _chunk(show: str) -> dict:
    return {
        "text": "hello",
        "episode": "ep1",
        "show": show,
        "source": "transcript",
        "dominant_speaker": "sp",
        "start": 0.0,
        "end": 1.0,
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A legacy index (name-keyed) plus one registered show folder.

    Runs with no show-folder resolver, so the read-time metadata heal cannot
    stamp the ids these tests assert the *migration* stamps. Restores
    whatever the rest of the suite had on the way out.
    """
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    folder = tmp_path / "shows" / "My Show"
    folder.mkdir(parents=True)
    save_show_meta(folder, ShowMeta(name="My Show"))

    store = IndexStore(tmp_path / "index")
    store.ensure_collection(
        "my_show__bge-m3__semantic",
        show="My Show",
        model="bge-m3",
        chunker="semantic",
        dim=DIM,
    )
    store.save_chunks(
        "my_show__bge-m3__semantic",
        "ep1",
        [_chunk("My Show")],
        np.random.default_rng(0).random((1, DIM), dtype=np.float32),
    )

    import podcodex.core.app_config as app_config

    cfg = app_config.AppConfig()
    cfg.show_folders = [str(folder)]
    monkeypatch.setattr(app_config, "load_config", lambda: cfg)

    with show_folder_resolver(None):
        yield store, folder


def test_migration_mints_id_and_keeps_the_table(env):
    """LanceDB OSS cannot rename a table, and does not need to: the name is
    internal, so a legacy collection keeps it and gains an id."""
    store, folder = env
    assert migrate_index_to_show_ids(store) == 1

    sid = show_id(folder)
    assert sid.startswith("my_show_")
    assert store.resolve_collection(sid, "bge-m3", "semantic") == (
        "my_show__bge-m3__semantic"
    )
    assert store.collection_exists("my_show__bge-m3__semantic")


def test_migration_preserves_chunks(env):
    store, folder = env
    migrate_index_to_show_ids(store)
    col = store.resolve_collection(show_id(folder), "bge-m3", "semantic")
    assert store.episode_chunk_count(col, "ep1") == 1


def test_migration_is_idempotent(env):
    store, _ = env
    assert migrate_index_to_show_ids(store) == 1
    assert migrate_index_to_show_ids(store) == 0
    assert migrate_index_to_show_ids(store) == 0


def _write_legacy_password_row(store, show: str, password_hash: str) -> None:
    """A row as written before show ids existed: keyed only by display name."""
    table = store._passwords_table()
    table.add(
        [
            {
                "show_id": "",
                "show_label": "",
                "show": show,
                "password_hash": password_hash,
            }
        ]
    )


def test_migration_rekeys_the_password(env):
    store, folder = env
    _write_legacy_password_row(store, "My Show", "sha256:abc")
    assert store.get_show_password_entries()["My Show"]["show_id"] == ""
    migrate_index_to_show_ids(store)

    sid = show_id(folder)
    entries = store.get_show_password_entries()
    assert sid in entries
    assert entries[sid]["password_hash"] == "sha256:abc"
    assert entries[sid]["label"] == "My Show"
    assert "My Show" not in entries


def test_partially_migrated_index_converges(env):
    """One collection stamped, one not: the next run finishes the job."""
    store, folder = env
    from podcodex.ingest.show import ensure_show_id

    store.ensure_collection(
        "my_show__bge-m3__sentence",
        show="My Show",
        model="bge-m3",
        chunker="sentence",
        dim=DIM,
    )
    sid = ensure_show_id(folder)
    store.set_collection_identity(
        "my_show__bge-m3__semantic", show_id=sid, show="My Show"
    )

    assert migrate_index_to_show_ids(store) == 1
    assert store.resolve_collection(sid, "bge-m3", "sentence") is not None
    assert store.resolve_collection(sid, "bge-m3", "semantic") is not None


def test_reindexing_after_migration_reuses_the_legacy_table(env):
    """A second table for the same show would silently split its index."""
    store, folder = env
    migrate_index_to_show_ids(store)
    sid = show_id(folder)

    name = store.ensure_collection_for_show(sid, "My Show", "bge-m3", "semantic", DIM)
    assert name == "my_show__bge-m3__semantic"
    assert len(store.collections_for_show(sid)) == 1


def test_orphaned_collection_is_left_alone(env):
    """A collection whose show is no longer registered must not be touched."""
    store, _ = env
    store.ensure_collection(
        "gone__bge-m3__semantic",
        show="Deleted Show",
        model="bge-m3",
        chunker="semantic",
        dim=DIM,
    )
    migrate_index_to_show_ids(store)
    assert store.collection_exists("gone__bge-m3__semantic")
    assert store.get_collection_info("gone__bge-m3__semantic")["show_id"] == ""


def test_rename_after_migration_keeps_the_collection(env):
    """The bug this all exists for."""
    store, folder = env
    migrate_index_to_show_ids(store)
    sid = show_id(folder)

    save_show_meta(folder, ShowMeta(id=sid, name="Brand New Name"))

    assert show_id(folder) == sid
    assert store.resolve_collection(sid, "bge-m3", "semantic") is not None


# ── Rename handler ───────────────────────────────────────────────────────


def test_rename_propagates_the_label_into_the_index(env):
    """The bot has no show.toml, so the new name has to travel in the index."""
    store, folder = env
    migrate_index_to_show_ids(store)
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
    migrate_index_to_show_ids(store)
    sid = show_id(folder)
    col = store.resolve_collection(sid, "bge-m3", "semantic")

    store.set_show_label(sid, "Brand New Name")
    hits = store.load_chunks_no_embeddings(col, "ep1")
    assert [h.show for h in hits] == ["Brand New Name"]


def test_password_survives_a_rename(env):
    store, folder = env
    migrate_index_to_show_ids(store)
    sid = show_id(folder)
    store.set_show_password(sid, "sha256:abc", show_label="My Show")

    store.set_show_label(sid, "Brand New Name")
    store.set_show_password(sid, "sha256:abc", show_label="Brand New Name")

    entries = store.get_show_password_entries()
    assert list(entries) == [sid]
    assert entries[sid]["label"] == "Brand New Name"


def test_show_renamed_before_upgrading_is_still_adopted(env):
    """The row says the old name, show.toml says the new one, and the table
    name is the only surviving link."""
    store, folder = env
    save_show_meta(folder, ShowMeta(name="Renamed Before Upgrade"))
    store.set_collection_identity(
        "my_show__bge-m3__semantic", show_id="", show="My Show"
    )

    assert migrate_index_to_show_ids(store) == 1

    sid = show_id(folder)
    assert sid.startswith("renamed_before_upgrade_")
    assert store.resolve_collection(sid, "bge-m3", "semantic") == (
        "my_show__bge-m3__semantic"
    )


def test_adoption_does_not_steal_another_shows_collection(env):
    """Two shows must not both claim one collection."""
    store, folder = env
    other = folder.parent / "Other Show"
    other.mkdir()
    save_show_meta(other, ShowMeta(name="Other Show"))

    import podcodex.core.app_config as app_config

    cfg = app_config.load_config()
    cfg.show_folders = [str(folder), str(other)]

    migrate_index_to_show_ids(store)

    sid = show_id(folder)
    other_id = show_id(other)
    assert store.collections_for_show(sid) == ["my_show__bge-m3__semantic"]
    assert store.collections_for_show(other_id) == []


# ── Regressions found in review ─────────────────────────────────────────


def test_indexing_then_renaming_keeps_the_collection(env):
    """A show created and indexed in one session, then renamed.

    Before identity was minted at save time, the collection was written with
    no id, the rename matched nothing, and the index was orphaned.
    """
    store, folder = env
    sid = show_id(folder)
    assert sid, "saving show.toml must establish identity"

    col = store.ensure_collection_for_show(sid, "My Show", "bge-m3", "sentence", DIM)
    save_show_meta(folder, ShowMeta(id=sid, name="Renamed"))
    assert store.set_show_label(sid, "Renamed", previous_label="My Show") >= 1

    assert store.resolve_collection(sid, "bge-m3", "sentence") == col
    assert store.collection_label(col) == "Renamed"


def test_relabel_adopts_rows_that_never_had_an_id(env):
    """Rows carrying neither the new label nor an id must still be found."""
    store, folder = env
    sid = show_id(folder)
    store.set_collection_identity(
        "my_show__bge-m3__semantic", show_id="", show="My Show"
    )

    assert store.set_show_label(sid, "Renamed", previous_label="My Show") == 1

    assert store.resolve_collection(sid, "bge-m3", "semantic") is not None


def test_password_set_before_indexing_is_rekeyed(env):
    """A show can be protected before it is ever indexed; that row still has
    to be migrated, or the app reports it public while the bot locks it."""
    store, folder = env
    for col in store.list_collections():
        store.delete_collection(col)
    _write_legacy_password_row(store, "My Show", "sha256:deadbeef")

    migrate_index_to_show_ids(store)

    sid = show_id(folder)
    entries = store.get_show_password_entries()
    assert sid in entries
    assert entries[sid]["password_hash"] == "sha256:deadbeef"
    assert "My Show" not in entries


# ── Read-time reconciliation ─────────────────────────────────────────────


def test_label_heals_on_read_after_an_offline_rename(env):
    """A rename that never went through the API (hand-edited show.toml, an
    import, the index offline) still reaches the index on the next read."""
    store, folder = env
    migrate_index_to_show_ids(store)
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
    migrate_index_to_show_ids(store)
    sid = show_id(folder)
    col = store.resolve_collection(sid, "bge-m3", "semantic")

    save_show_meta(folder, ShowMeta(id=sid, name="Renamed"))
    store._collection_info_cache = None

    # No resolver registered: this is the bot, and nothing heals.
    assert store.collection_label(col) == "My Show"
