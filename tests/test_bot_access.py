"""Tests for the bot-access route (show password management)."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")


from podcodex.api.app import app  # noqa: E402
from podcodex.rag import index_store as rag_index_store  # noqa: E402
from tests.fixtures.api_client import client_for


DIM = 8


def _seed_store(tmp_path: Path):
    """Fresh IndexStore with two shows indexed under the default combo."""
    store = rag_index_store.IndexStore(tmp_path / "index")
    for show in ("Alpha", "Beta"):
        col = f"{show.lower()}__bge-m3__semantic"
        store.ensure_collection(
            col, show=show, model="bge-m3", chunker="semantic", dim=DIM
        )
        chunks = [
            {
                "text": "x",
                "episode": "ep1",
                "show": show,
                "source": "transcript",
                "dominant_speaker": "sp",
                "start": 0.0,
                "end": 1.0,
            }
        ]
        rng = np.random.default_rng(0)
        store.save_chunks(col, "ep1", chunks, rng.random((1, DIM), dtype=np.float32))
    return store


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    # Pinned before seeding: creating the index stamps its owner, and without
    # this the stamp would come from (and create) the developer's real
    # machine-id file.
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "test-owner")
    _seed_store(tmp_path)
    monkeypatch.setenv("PODCODEX_INDEX", str(tmp_path / "index"))
    # Isolate from the real config.json: show names come only from the
    # seeded index, not the developer's registered show folders.
    import podcodex.core.app_config as app_config

    monkeypatch.setattr(app_config, "load_config", lambda: app_config.AppConfig())
    rag_index_store.get_index_store.cache_clear()
    yield
    rag_index_store.get_index_store.cache_clear()


@pytest.fixture
def client():
    return client_for(app)


# ── List ────────────────────────────────────────────────────────────────


def test_list_shows_all_unprotected_initially(client):
    r = client.get("/api/bot-access/passwords")
    assert r.status_code == 200
    body = r.json()
    assert [b["show"] for b in body] == ["Alpha", "Beta"]
    assert all(b["is_protected"] is False for b in body)


def test_get_one_unknown_show_404(client):
    r = client.get("/api/bot-access/passwords/Nope")
    assert r.status_code == 404


# ── Generate ────────────────────────────────────────────────────────────


def test_generate_returns_plaintext_once(client):
    r = client.post("/api/bot-access/passwords/Alpha", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["show"] == "Alpha"
    assert body["generated"] is True
    assert isinstance(body["password"], str)
    assert len(body["password"]) >= 20  # 16 bytes -> 22 urlsafe chars

    # Status now reflects protected
    status = client.get("/api/bot-access/passwords/Alpha").json()
    assert status["is_protected"] is True


def test_generate_stores_sha256_hash(client):
    r = client.post("/api/bot-access/passwords/Alpha", json={})
    plaintext = r.json()["password"]
    expected = f"sha256:{hashlib.sha256(plaintext.encode()).hexdigest()}"

    store = rag_index_store.get_index_store()
    assert store.get_show_passwords()["Alpha"] == expected


# ── Manual ──────────────────────────────────────────────────────────────


def test_manual_password_accepts_16_chars(client):
    r = client.post("/api/bot-access/passwords/Alpha", json={"password": "a" * 16})
    assert r.status_code == 200
    body = r.json()
    assert body["generated"] is False
    assert body["password"] == "a" * 16


def test_manual_password_rejects_too_short(client):
    r = client.post("/api/bot-access/passwords/Alpha", json={"password": "short"})
    assert r.status_code == 422
    assert "at least 16" in r.json()["detail"]


def test_manual_password_whitespace_is_trimmed_then_rejected(client):
    r = client.post("/api/bot-access/passwords/Alpha", json={"password": "   "})
    # Trimmed to empty → treated as generate, not manual; should generate.
    # Confirm behaviour: empty-after-trim means generate.
    assert r.status_code == 200
    assert r.json()["generated"] is True


# ── Rotate ──────────────────────────────────────────────────────────────


def test_rotate_replaces_existing_hash(client):
    first = client.post("/api/bot-access/passwords/Alpha", json={}).json()
    second = client.post("/api/bot-access/passwords/Alpha", json={}).json()
    assert first["password"] != second["password"]

    store = rag_index_store.get_index_store()
    expected = f"sha256:{hashlib.sha256(second['password'].encode()).hexdigest()}"
    assert store.get_show_passwords()["Alpha"] == expected


# ── Delete ──────────────────────────────────────────────────────────────


def test_delete_removes_protection(client):
    client.post("/api/bot-access/passwords/Alpha", json={})
    r = client.delete("/api/bot-access/passwords/Alpha")
    assert r.status_code == 204
    assert client.get("/api/bot-access/passwords/Alpha").json()["is_protected"] is False


def test_delete_unknown_show_404(client):
    r = client.delete("/api/bot-access/passwords/Nope")
    assert r.status_code == 404


# ── Unknown show ────────────────────────────────────────────────────────


def test_set_unknown_show_404(client):
    r = client.post("/api/bot-access/passwords/Nope", json={})
    assert r.status_code == 404


# ── Index ownership ─────────────────────────────────────────────────────


def test_set_password_on_replica_returns_409(client, monkeypatch):
    """A replica must not accept a password the next rsync would erase."""
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")

    r = client.post("/api/bot-access/passwords/Alpha", json={})
    assert r.status_code == 409
    assert "replica" in r.json()["detail"]

    assert rag_index_store.get_index_store().get_show_passwords() == {}


def test_delete_password_on_replica_returns_409(client, monkeypatch):
    assert client.post("/api/bot-access/passwords/Alpha", json={}).status_code == 200
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")

    r = client.delete("/api/bot-access/passwords/Alpha")
    assert r.status_code == 409
    assert client.get("/api/bot-access/passwords/Alpha").json()["is_protected"] is True


def test_claiming_the_index_restores_writes(client, monkeypatch):
    from podcodex.rag.index_origin import claim_origin

    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")
    assert client.post("/api/bot-access/passwords/Alpha", json={}).status_code == 409

    claim_origin(rag_index_store.get_index_store().path)
    assert client.post("/api/bot-access/passwords/Alpha", json={}).status_code == 200


def test_unstamped_index_still_accepts_writes(client, monkeypatch):
    """Indexes predating the marker are unowned and must keep working."""
    from podcodex.rag.index_origin import ORIGIN_FILENAME

    (rag_index_store.get_index_store().path / ORIGIN_FILENAME).unlink()
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "some-other-host")

    assert client.post("/api/bot-access/passwords/Alpha", json={}).status_code == 200


# ── Guild unlock lists keyed by show id ─────────────────────────────────


def _bare_bot(store, server_cfg):
    """A bot object with just enough wired for the access mixin."""
    from podcodex.bot.bot import PodCodexBot

    bot = PodCodexBot.__new__(PodCodexBot)
    bot._shows = {}
    bot._local = store
    bot._server_cfg = server_cfg
    bot._save_server_config = lambda: None
    return bot


def test_a_protected_legacy_entry_reads_as_an_unlock(tmp_path):
    """`allowed_shows` predates the split and records nothing about which
    command wrote it, so it is read the way the pre-split code read it: an
    unlock exactly while the show is protected. No upgrade can silently
    revoke access."""
    from podcodex.bot.config import ServerSettings
    from podcodex.core.show_passwords import hash_show_password

    store = rag_index_store.get_index_store()
    store.set_collection_identity(
        "alpha__bge-m3__semantic", show_id="alpha_1234abcd", show="Alpha"
    )
    store.set_show_password(
        "alpha_1234abcd", hash_show_password("x" * 16), show_label="Alpha"
    )
    settings = ServerSettings(allowed_shows=["Alpha"])
    bot = _bare_bot(store, {1: settings})

    bot._reload_shows()

    assert settings.allowed_shows == ["alpha_1234abcd"]  # names → ids, nothing more
    assert bot._show_allowed_by_label("Alpha", settings) is True
    # Not offered as a pin: it is an unlock as far as anything can tell, and
    # counting it as both made `/setup show_remove` silently fail to unpin.
    assert bot._pinned_ids(settings) == []


def test_a_reload_never_reclassifies_or_drops_a_legacy_entry(tmp_path):
    """The drain this replaces was one-shot and destructive: a reload that
    landed mid-rsync, on an index whose password table had not arrived, read
    every unlock as a pin and saved that, with no recovery but re-running
    `/unlock` per guild."""
    from podcodex.bot.config import ServerSettings

    store = rag_index_store.get_index_store()
    store.set_collection_identity(
        "alpha__bge-m3__semantic", show_id="alpha_1234abcd", show="Alpha"
    )
    settings = ServerSettings(allowed_shows=["alpha_1234abcd"])
    saves: list[bool] = []
    bot = _bare_bot(store, {1: settings})
    bot._save_server_config = lambda: saves.append(True)

    bot._reload_shows()  # no password table at all: the dangerous shape
    bot._reload_shows()

    assert settings.allowed_shows == ["alpha_1234abcd"]
    assert settings.unlocked_shows == [] and settings.pinned_shows == []
    assert saves == []  # already ids, so nothing to write


def test_lock_settles_a_legacy_entry(tmp_path):
    """An admin naming the show is the one moment its meaning is known."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(allowed_shows=["alpha"])
    bot, saves = _guild_bot({1: settings}, locked_ids={"alpha"})
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_lock(interaction, "Alpha"))

    assert settings.allowed_shows == []
    assert saves == [True]
    assert "removed" in interaction.response.messages[0]


def test_unlocked_show_survives_a_rename(tmp_path):
    """The bug one layer out: renaming used to re-lock every guild."""
    from podcodex.bot.config import ServerSettings
    from podcodex.core.show_passwords import hash_show_password

    store = rag_index_store.get_index_store()
    store.set_collection_identity(
        "alpha__bge-m3__semantic", show_id="alpha_1234abcd", show="Alpha"
    )
    store.set_show_password(
        "alpha_1234abcd", hash_show_password("x" * 16), show_label="Alpha"
    )

    settings = ServerSettings(allowed_shows=["Alpha"])
    bot = _bare_bot(store, {1: settings})
    bot._reload_shows()
    assert bot._show_allowed_by_label("Alpha", settings) is True

    # Rename: label changes in the index, identity does not.
    store.set_show_label("alpha_1234abcd", "Renamed")
    store.set_show_password(
        "alpha_1234abcd", hash_show_password("x" * 16), show_label="Renamed"
    )
    bot._reload_shows()

    assert bot._show_allowed_by_label("Renamed", settings) is True
    assert settings.allowed_shows == ["alpha_1234abcd"]


def test_protected_show_stays_locked_for_other_guilds(tmp_path):
    from podcodex.bot.config import ServerSettings
    from podcodex.core.show_passwords import hash_show_password

    store = rag_index_store.get_index_store()
    store.set_collection_identity(
        "alpha__bge-m3__semantic", show_id="alpha_1234abcd", show="Alpha"
    )
    store.set_show_password(
        "alpha_1234abcd", hash_show_password("x" * 16), show_label="Alpha"
    )

    other = ServerSettings(unlocked_shows=[])
    bot = _bare_bot(store, {2: other})
    bot._reload_shows()

    assert bot._show_allowed_by_label("Alpha", other) is False
    store.set_show_label("alpha_1234abcd", "Renamed")
    bot._reload_shows()
    assert bot._show_allowed_by_label("Renamed", other) is False


# ── DM guard: guild_id is None outside a guild ──────────────────────────


class _DMResponse:
    def __init__(self):
        self.messages: list[str] = []

    async def send_message(self, content, *, ephemeral=False):
        self.messages.append(content)


class _DMInteraction:
    def __init__(self):
        self.guild_id = None
        self.response = _DMResponse()


def _dm_bot():
    from podcodex.bot.bot import BotConfig, PodCodexBot

    bot = PodCodexBot.__new__(PodCodexBot)
    bot._shows = {}
    bot._server_cfg = {}
    bot.config = BotConfig()
    saved: list[bool] = []
    bot._save_server_config = lambda: saved.append(True)
    return bot, saved


def test_unlock_in_a_dm_writes_no_settings(tmp_path):
    """guild_id None used to be stored as the JSON key "None", which the next
    bot start could not parse back into an int."""
    bot, saved = _dm_bot()

    async def _noop_refresh():
        return None

    bot._refresh_if_stale = _noop_refresh
    bot._reload_shows = lambda: None
    interaction = _DMInteraction()

    asyncio.run(bot._handle_unlock(interaction, "x" * 16))

    assert bot._server_cfg == {}
    assert saved == []
    assert "server" in interaction.response.messages[0]


def test_setup_in_a_dm_writes_no_settings(tmp_path):
    bot, saved = _dm_bot()
    interaction = _DMInteraction()

    asyncio.run(bot._handle_setup(interaction, "bge-m3", None, None))

    assert bot._server_cfg == {}
    assert saved == []


def test_lock_in_a_dm_writes_no_settings(tmp_path):
    bot, saved = _dm_bot()
    interaction = _DMInteraction()

    asyncio.run(bot._handle_lock(interaction, "Alpha"))

    assert bot._server_cfg == {}
    assert saved == []


def test_load_server_config_skips_a_non_guild_key(tmp_path):
    """Configs written before the DM guard carry a literal "None" key; the bot
    must start anyway instead of dying in int('None')."""
    import json

    from podcodex.bot.bot import PodCodexBot

    path = tmp_path / "server_config.json"
    path.write_text(
        json.dumps({"None": {"top_k": 3}, "42": {"top_k": 7}}), encoding="utf-8"
    )
    bot = PodCodexBot.__new__(PodCodexBot)
    bot.server_config_path = path

    cfg = bot._load_server_config()

    assert list(cfg) == [42]
    assert cfg[42].top_k == 7


# ── Pins and unlocks are separate acts ──────────────────────────────────
#
# ``allowed_shows`` used to mean both "unlocked here" and "pinned as this
# server's default", so ``/lock`` un-pinned public shows, ``/setup`` refused
# to pin anything once one show had a password, and a typo pinned a name
# that resolved to nothing. These pin the split that fixed all three.


class _Reply:
    """Records what a handler answered, for both response and followup."""

    def __init__(self):
        self.messages: list[str] = []
        self.deferred = False

    async def send_message(self, content=None, *, ephemeral=False, **_kw):
        self.messages.append(content or "")

    async def send(self, content=None, *, ephemeral=False, **_kw):
        self.messages.append(content or "")

    async def defer(self, **_kw):
        self.deferred = True


class _GuildInteraction:
    def __init__(self, guild_id=1):
        self.guild_id = guild_id
        self.response = _Reply()
        self.followup = self.response


def _guild_bot(server_cfg, *, locked_ids=(), known_shows=()):
    """A bot wired for the /setup, /lock and /episodes handlers."""
    from podcodex.bot.access import ShowEntry
    from podcodex.bot.autocomplete import _AutocompleteCache
    from podcodex.bot.bot import BotConfig, PodCodexBot

    bot = PodCodexBot.__new__(PodCodexBot)
    bot.config = BotConfig()
    bot._ac_cache = _AutocompleteCache()
    # `_locked_show_ids` is derived from `_shows`, so seed that rather than
    # patching the property onto the shared class.
    bot._shows = {
        sid: ShowEntry(show_id=sid, name=sid, password_hash="") for sid in locked_ids
    }
    bot._server_cfg = server_cfg
    saves: list[bool] = []
    bot._save_server_config = lambda: saves.append(True)
    # Labels round-trip through a lowercase id; the real pair is index-backed.
    bot._show_id_for_label = lambda label: label.strip().lower().replace(" ", "_")
    bot._label_for_show_id = lambda sid: sid.replace("_", " ").title()

    async def _known(settings, model="", chunker=""):
        return list(known_shows)

    bot._known_show_labels = _known

    async def _noop_refresh():
        return None

    bot._refresh_if_stale = _noop_refresh
    bot._cache_clear_if_stale = lambda: None
    return bot, saves


def test_lock_on_a_public_show_changes_nothing(tmp_path):
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(pinned_shows=["alpha"])
    bot, saves = _guild_bot({1: settings})
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_lock(interaction, "Alpha"))

    assert settings.pinned_shows == ["alpha"]
    assert saves == []
    assert "not password-protected" in interaction.response.messages[0]


def test_setup_pins_while_another_show_is_protected(tmp_path):
    """The guard used to be global: one password blocked every pin."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings()
    bot, saves = _guild_bot({1: settings}, locked_ids={"beta"}, known_shows=["Alpha"])
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_setup(interaction, None, None, None, show_add="Alpha"))

    assert bot._server_cfg[1].pinned_shows == ["alpha"]
    assert saves == [True]


def test_setup_rejects_a_show_name_that_resolves_to_nothing(tmp_path):
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings()
    bot, saves = _guild_bot({1: settings}, known_shows=["Alpha", "Beta"])
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_setup(interaction, None, None, None, show_add="Alfa"))

    assert settings.pinned_shows == []
    assert saves == []
    assert "Alpha" in interaction.response.messages[0]


def test_setup_show_remove_drops_a_pin_whose_show_left_the_index(tmp_path):
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(pinned_shows=["gone"])
    bot, saves = _guild_bot({1: settings}, known_shows=[])
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_setup(interaction, None, None, None, show_remove="Gone"))

    assert bot._server_cfg[1].pinned_shows == []
    assert saves == [True]


def test_episodes_lists_the_pins_instead_of_picking_the_first(tmp_path):
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(pinned_shows=["alpha", "beta"])
    bot, _saves = _guild_bot({1: settings})

    async def _noop_refresh():
        return None

    bot._refresh_if_stale = _noop_refresh
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_episodes(interaction, None, None))

    reply = interaction.response.messages[0]
    assert "Alpha" in reply and "Beta" in reply


def test_an_unprotected_legacy_entry_reads_as_a_pin(tmp_path):
    """The other half of the same rule: nothing is protected, so the entry
    can only have come from `/setup show_add`."""
    from podcodex.bot.config import ServerSettings

    store = rag_index_store.get_index_store()
    store.set_collection_identity(
        "alpha__bge-m3__semantic", show_id="alpha_1234abcd", show="Alpha"
    )
    settings = ServerSettings(allowed_shows=["Alpha"])
    bot = _bare_bot(store, {1: settings})

    bot._reload_shows()

    assert bot._pinned_ids(settings) == ["alpha_1234abcd"]
    assert bot._unlocked_ids(settings) == []


def test_setup_show_remove_settles_an_unprotected_legacy_pin(tmp_path):
    """It used to report success while the entry stayed, because the legacy
    list was kept and still counted as a pin."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(allowed_shows=["alpha"])
    bot, saves = _guild_bot({1: settings}, known_shows=["Alpha"])
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_setup(interaction, None, None, None, show_remove="Alpha"))

    updated = bot._server_cfg[1]
    assert bot._pinned_ids(updated) == []
    assert updated.allowed_shows == []
    assert saves == [True]


def test_setup_show_remove_leaves_a_protected_legacy_entry_alone(tmp_path):
    """It is an unlock, not a pin, so there is nothing to unpin — and
    dropping it would revoke access the guild may well have earned."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(allowed_shows=["alpha"])
    bot, saves = _guild_bot({1: settings}, locked_ids={"alpha"}, known_shows=["Alpha"])
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_setup(interaction, None, None, None, show_remove="Alpha"))

    assert settings.allowed_shows == ["alpha"]
    assert saves == []
    assert "not pinned" in interaction.response.messages[0]


def test_an_unrelated_setup_change_never_rewrites_the_legacy_list(tmp_path):
    """An index rsynced mid-transfer reports no protected shows without
    erroring. Recomputing the legacy list on every `/setup` filed every
    unlock as a pin in that window and cleared the list, losing the guild's
    access for good; the read path re-derives precisely so it survives."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(allowed_shows=["alpha"])
    # The dangerous state: an unlock on the books, nothing readable as locked.
    bot, saves = _guild_bot({1: settings}, locked_ids=set())
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_setup(interaction, "bge-m3", None, None))

    updated = bot._server_cfg[1]
    assert updated.allowed_shows == ["alpha"]
    assert updated.pinned_shows == []
    assert updated.model == "bge-m3"  # the change the admin actually asked for


def test_lock_on_a_public_show_leaves_a_legacy_pin_alone(tmp_path):
    """`/lock` revokes access. A public show's legacy entry is a pin, so
    deleting it here would drop a search default while reporting an access
    change — the confusion the split exists to end."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(allowed_shows=["beta"])
    bot, saves = _guild_bot({1: settings}, locked_ids={"alpha"})
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_lock(interaction, "Beta"))

    assert settings.allowed_shows == ["beta"]
    assert saves == []
    assert "not password-protected" in interaction.response.messages[0]


def test_changepassword_refuses_a_show_that_is_no_longer_protected(tmp_path):
    """`unlocked_shows` is never pruned when the app makes a show public, so
    a stale entry would otherwise re-protect it index-wide and lock every
    other guild out, with the password DM'd only to the caller."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(unlocked_shows=["alpha"])
    bot, _saves = _guild_bot({1: settings}, locked_ids=set())
    rotated: list[str] = []
    bot._local = _RecordingStore(rotated)
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_changepassword(interaction, "Alpha"))

    assert rotated == []
    assert "no password to rotate" in interaction.response.messages[0]


class _RecordingStore:
    """Records any attempt to write a password."""

    def __init__(self, sink):
        self._sink = sink

    def set_show_password(self, show_id, *_a, **_kw):
        self._sink.append(show_id)
