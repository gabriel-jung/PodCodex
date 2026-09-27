"""Tests for the one-time name-keyed to id-keyed index migration.

Every test here is legacy and goes with ``rag/show_id_migration.py``; the
permanent identity rules live in ``test_show_identity_index.py``.
"""

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


@pytest.mark.legacy("show-id")
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
    assert store.episode_chunk_count("my_show__bge-m3__semantic", "ep1") == 1


@pytest.mark.legacy("show-id")
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


@pytest.mark.legacy("show-id")
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


@pytest.mark.legacy("show-id")
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


@pytest.mark.legacy("show-id")
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


@pytest.mark.legacy("show-id")
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


@pytest.mark.legacy("show-id")
def test_password_of_a_show_renamed_before_upgrading_is_rekeyed(env):
    """The password is under the old name, which only the collection row keeps.
    Missing it would serve a protected show as public in the bot."""
    store, folder = env
    save_show_meta(folder, ShowMeta(name="Renamed Before Upgrade"))
    store.set_collection_identity(
        "my_show__bge-m3__semantic", show_id="", show="My Show"
    )
    _write_legacy_password_row(store, "my show", "sha256:abc")

    migrate_index_to_show_ids(store)

    sid = show_id(folder)
    entries = store.get_show_password_entries()
    assert list(entries) == [sid]
    assert entries[sid]["password_hash"] == "sha256:abc"
    assert entries[sid]["label"] == "Renamed Before Upgrade"


@pytest.mark.legacy("name-keyed-passwords")
def test_deleting_a_legacy_password_keeps_another_show_with_that_label(env):
    store, _folder = env
    store.set_show_password("show_a", "sha256:a", show_label="Twin")
    _write_legacy_password_row(store, "Twin", "sha256:legacy")

    store.delete_show_password("Twin")

    assert list(store.get_show_password_entries()) == ["show_a"]


@pytest.mark.legacy("name-keyed-passwords")
def test_legacy_password_with_odd_whitespace_is_deleted(env):
    store, _folder = env
    _write_legacy_password_row(store, " Foo\t", "sha256:legacy")

    store.delete_show_password("foo")

    assert store.get_show_password_entries() == {}


@pytest.mark.legacy("show-id")
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


@pytest.mark.legacy("show-id")
def test_relabel_adopts_rows_that_never_had_an_id(env):
    """Rows carrying neither the new label nor an id must still be found."""
    store, folder = env
    sid = show_id(folder)
    store.set_collection_identity(
        "my_show__bge-m3__semantic", show_id="", show="My Show"
    )

    assert store.set_show_label(sid, "Renamed", previous_label="My Show") == 1

    assert store.resolve_collection(sid, "bge-m3", "semantic") is not None


@pytest.mark.legacy("show-id")
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
