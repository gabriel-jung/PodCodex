"""Tests for index ownership: machine identity, origin marker, write guard."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fresh_data_dir():
    """``app_paths.data_dir`` is lru_cached, so a per-test PODCODEX_DATA_DIR
    would otherwise resolve to whichever directory the first test used."""
    from podcodex.core.app_paths import data_dir

    data_dir.cache_clear()
    yield
    data_dir.cache_clear()


# ── Machine identity ─────────────────────────────────────────────────────


def test_machine_id_prefers_env(monkeypatch, tmp_path):
    from podcodex.core import machine_id as mod

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "from-env")
    monkeypatch.setenv("PODCODEX_DATA_DIR", str(tmp_path))
    assert mod.machine_id() == "from-env"
    assert not (tmp_path / "machine_id").exists()


def test_machine_id_generated_once_and_persisted(monkeypatch, tmp_path):
    from podcodex.core import machine_id as mod

    monkeypatch.delenv("PODCODEX_MACHINE_ID", raising=False)
    monkeypatch.setenv("PODCODEX_DATA_DIR", str(tmp_path))
    first = mod.machine_id()
    assert first
    assert (tmp_path / "machine_id").read_text(encoding="utf-8").strip() == first
    assert mod.machine_id() == first


def test_machine_id_blank_env_falls_through(monkeypatch, tmp_path):
    from podcodex.core import machine_id as mod

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "   ")
    monkeypatch.setenv("PODCODEX_DATA_DIR", str(tmp_path))
    generated = mod.machine_id()
    assert generated.strip() == generated
    assert (tmp_path / "machine_id").exists()


# ── Origin marker ────────────────────────────────────────────────────────


def _unowned(monkeypatch, tmp_path, machine="machine-a"):
    monkeypatch.setenv("PODCODEX_MACHINE_ID", machine)
    idx = tmp_path / "index"
    idx.mkdir()
    return idx


def test_read_origin_empty_when_unstamped(monkeypatch, tmp_path):
    from podcodex.rag import index_origin as mod

    idx = _unowned(monkeypatch, tmp_path)
    assert mod.read_origin(idx) == ""


def test_unstamped_index_is_not_a_replica(monkeypatch, tmp_path):
    """D4: an unstamped index is unowned, so anyone may write."""
    from podcodex.rag import index_origin as mod

    idx = _unowned(monkeypatch, tmp_path)
    assert mod.is_replica(idx) is False
    mod.require_owner(idx, "set a password")  # must not raise


def test_stamp_only_when_directory_was_empty(monkeypatch, tmp_path):
    from podcodex.rag import index_origin as mod

    idx = _unowned(monkeypatch, tmp_path)
    mod.stamp_origin_if_new(idx, was_empty=False)
    assert mod.read_origin(idx) == ""

    mod.stamp_origin_if_new(idx, was_empty=True)
    assert mod.read_origin(idx) == "machine-a"


def test_stamp_is_not_overwritten(monkeypatch, tmp_path):
    from podcodex.rag import index_origin as mod

    idx = _unowned(monkeypatch, tmp_path)
    mod.stamp_origin_if_new(idx, was_empty=True)
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "machine-b")
    mod.stamp_origin_if_new(idx, was_empty=True)
    assert mod.read_origin(idx) == "machine-a"


def test_replica_detected_and_refused(monkeypatch, tmp_path):
    from podcodex.rag import index_origin as mod

    idx = _unowned(monkeypatch, tmp_path)
    mod.stamp_origin_if_new(idx, was_empty=True)

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "machine-b")
    assert mod.is_replica(idx) is True
    with pytest.raises(mod.IndexOwnershipError) as exc:
        mod.require_owner(idx, "set a password")
    assert "machine-a" in str(exc.value)


def test_claim_makes_this_machine_the_owner(monkeypatch, tmp_path):
    from podcodex.rag import index_origin as mod

    idx = _unowned(monkeypatch, tmp_path)
    mod.stamp_origin_if_new(idx, was_empty=True)

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "machine-b")
    previous = mod.claim_origin(idx)
    assert previous == "machine-a"
    assert mod.is_replica(idx) is False
    mod.require_owner(idx, "set a password")  # must not raise


def test_corrupt_marker_is_treated_as_unowned(monkeypatch, tmp_path):
    """A truncated marker (half-finished rsync) must not lock writes out."""
    from podcodex.rag import index_origin as mod

    idx = _unowned(monkeypatch, tmp_path)
    (idx / mod.ORIGIN_FILENAME).write_text("{not json", encoding="utf-8")
    assert mod.read_origin(idx) == ""
    assert mod.is_replica(idx) is False


# ── IndexStore integration ───────────────────────────────────────────────


def test_new_index_is_stamped_by_its_creator(monkeypatch, tmp_path):
    from podcodex.rag.index_origin import read_origin
    from podcodex.rag.index_store import IndexStore

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    store = IndexStore(tmp_path / "index")
    assert read_origin(store.path) == "desktop"


def test_existing_unstamped_index_stays_unowned(monkeypatch, tmp_path):
    """Every pre-existing install is in this state and must keep working."""
    from podcodex.rag.index_origin import read_origin
    from podcodex.rag.index_store import IndexStore

    idx = tmp_path / "index"
    idx.mkdir()
    (idx / "some_table.lance").mkdir()

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")
    store = IndexStore(idx)
    assert read_origin(store.path) == ""
    store.set_show_password("S", "sha256:abc")
    assert store.get_show_passwords() == {"S": "sha256:abc"}


def test_owner_may_write_passwords(monkeypatch, tmp_path):
    from podcodex.rag.index_store import IndexStore

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    store = IndexStore(tmp_path / "index")
    store.set_show_password("S", "sha256:abc")
    assert store.get_show_passwords() == {"S": "sha256:abc"}


def test_replica_refuses_password_write(monkeypatch, tmp_path):
    from podcodex.rag.index_origin import IndexOwnershipError, is_replica
    from podcodex.rag.index_store import IndexStore

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    IndexStore(tmp_path / "index")  # creates and stamps

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")
    replica = IndexStore(tmp_path / "index")
    assert is_replica(replica.path) is True
    with pytest.raises(IndexOwnershipError):
        replica.set_show_password("S", "sha256:abc")
    with pytest.raises(IndexOwnershipError):
        replica.delete_show_password("S")


def test_replica_still_reads_passwords(monkeypatch, tmp_path):
    """Refusal is write-only. The bot must keep serving access control."""
    from podcodex.rag.index_store import IndexStore

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    owner = IndexStore(tmp_path / "index")
    owner.set_show_password("S", "sha256:abc")

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")
    replica = IndexStore(tmp_path / "index")
    assert replica.get_show_passwords() == {"S": "sha256:abc"}


def test_claim_then_write_succeeds(monkeypatch, tmp_path):
    from podcodex.rag.index_origin import claim_origin
    from podcodex.rag.index_store import IndexStore

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    IndexStore(tmp_path / "index")

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")
    replica = IndexStore(tmp_path / "index")
    claim_origin(replica.path)
    replica.set_show_password("S", "sha256:abc")
    assert replica.get_show_passwords() == {"S": "sha256:abc"}


# ── Bot CLI ──────────────────────────────────────────────────────────────


def test_manage_passwords_cli_refuses_on_replica(monkeypatch, tmp_path, capsys):
    from podcodex.bot import bot as botmod
    from podcodex.rag.index_store import IndexStore

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    store = IndexStore(tmp_path / "index")
    store.ensure_collection(
        "s__bge-m3__semantic", show="S", model="bge-m3", chunker="semantic", dim=8
    )

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")
    botmod._manage_passwords_cli(str(tmp_path / "index"))
    out = capsys.readouterr().out
    assert "replica" in out
    assert "--claim-index" in out


def test_claim_index_cli_takes_ownership(monkeypatch, tmp_path, capsys):
    from podcodex.bot import bot as botmod
    from podcodex.rag.index_origin import read_origin
    from podcodex.rag.index_store import IndexStore

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    IndexStore(tmp_path / "index")

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")
    botmod._claim_index_cli(str(tmp_path / "index"))
    out = capsys.readouterr().out
    assert "now owned by this machine" in out
    assert "previous owner: desktop" in out
    assert read_origin(tmp_path / "index") == "bot-host"


def test_manage_passwords_cli_keys_on_show_id(monkeypatch, tmp_path, capsys):
    """The operator types a name; the table must still be keyed by id."""
    from podcodex.bot import bot as botmod
    from podcodex.rag.index_store import IndexStore

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    store = IndexStore(tmp_path / "index")
    store.ensure_collection(
        "s__bge-m3__semantic",
        show="My Show",
        model="bge-m3",
        chunker="semantic",
        dim=8,
        show_id="my_show_abcd1234",
    )
    monkeypatch.setattr("builtins.input", lambda *_a: "")
    botmod._manage_passwords_cli(str(tmp_path / "index"))
    assert "My Show" in capsys.readouterr().out

    store.set_show_password("my_show_abcd1234", "sha256:abc", show_label="My Show")
    assert list(store.get_show_password_entries()) == ["my_show_abcd1234"]


# ── Which writes the guard covers ────────────────────────────────────────


def _replica(monkeypatch, tmp_path):
    """An index created by "desktop", now being opened by "bot-host"."""
    from podcodex.rag.index_store import IndexStore

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "desktop")
    owner = IndexStore(tmp_path / "index")
    owner.ensure_collection(
        "s__bge-m3__semantic",
        show="S",
        model="bge-m3",
        chunker="semantic",
        dim=8,
        show_id="s_abcd1234",
    )
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")
    return IndexStore(tmp_path / "index")


def test_replica_refuses_identity_and_label_writes(monkeypatch, tmp_path):
    from podcodex.rag.index_origin import IndexOwnershipError

    replica = _replica(monkeypatch, tmp_path)
    with pytest.raises(IndexOwnershipError):
        replica.set_collection_identity("s__bge-m3__semantic", "other_id", "Other")
    with pytest.raises(IndexOwnershipError):
        replica.set_show_label("s_abcd1234", "Renamed")
    with pytest.raises(IndexOwnershipError):
        replica.delete_collection("s__bge-m3__semantic")

    assert replica.collection_exists("s__bge-m3__semantic")
    assert replica.get_collection_info("s__bge-m3__semantic")["show"] == "S"


def test_replica_still_serves_reads(monkeypatch, tmp_path):
    """The guard is write-only: refusing reads would take the bot offline."""
    replica = _replica(monkeypatch, tmp_path)

    assert replica.resolve_collection("s_abcd1234", "bge-m3", "semantic")
    assert replica.collections_for_show("s_abcd1234")
    assert replica.collection_label("s__bge-m3__semantic") == "S"
    assert replica.show_id_for_label("S") == "s_abcd1234"
