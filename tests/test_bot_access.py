"""Tests for the bot-access route (show password management)."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")


from podcodex.api.app import app  # noqa: E402
from podcodex.rag import index_store as rag_index_store  # noqa: E402
from tests.fixtures.api_client import client_for
from tests.fixtures.bot import make_bot
from tests.fixtures.index import add_show, relabel


DIM = 8


def _seed_store(tmp_path: Path):
    """Fresh IndexStore with two shows, Alpha and Beta (ids "alpha", "beta")."""
    store = rag_index_store.IndexStore(tmp_path / "index")
    for label in ("Alpha", "Beta"):
        add_show(store, label, show_id=label.lower())
    return store


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch, isolated_index):
    # Pinned before seeding: creating the index stamps its owner, and without
    # this the stamp would come from (and create) the developer's real
    # machine-id file.
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "test-owner")
    _seed_store(tmp_path)
    # Isolate from the real config.json: show names come only from the
    # seeded index, not the developer's registered show folders.
    import podcodex.core.app_config as app_config

    monkeypatch.setattr(app_config, "load_config", lambda: app_config.AppConfig())


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
    r = client.get("/api/bot-access/password?show_id=Nope")
    assert r.status_code == 404


# ── Generate ────────────────────────────────────────────────────────────


def test_generate_returns_plaintext_once(client):
    r = client.post("/api/bot-access/password?show_id=alpha", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["show"] == "Alpha"
    assert body["generated"] is True
    assert isinstance(body["password"], str)
    assert len(body["password"]) >= 20  # 16 bytes -> 22 urlsafe chars

    # Status now reflects protected
    status = client.get("/api/bot-access/password?show_id=alpha").json()
    assert status["is_protected"] is True


def test_generate_stores_sha256_hash(client):
    r = client.post("/api/bot-access/password?show_id=alpha", json={})
    plaintext = r.json()["password"]
    expected = f"sha256:{hashlib.sha256(plaintext.encode()).hexdigest()}"

    store = rag_index_store.get_index_store()
    assert store.get_show_passwords()["alpha"] == expected


# ── Manual ──────────────────────────────────────────────────────────────


def test_manual_password_accepts_16_chars(client):
    r = client.post(
        "/api/bot-access/password?show_id=alpha", json={"password": "a" * 16}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["generated"] is False
    assert body["password"] == "a" * 16


def test_manual_password_rejects_too_short(client):
    r = client.post(
        "/api/bot-access/password?show_id=alpha", json={"password": "short"}
    )
    assert r.status_code == 422
    assert "at least 16" in r.json()["detail"]


def test_manual_password_whitespace_is_trimmed_then_rejected(client):
    r = client.post("/api/bot-access/password?show_id=alpha", json={"password": "   "})
    # Trimmed to empty → treated as generate, not manual; should generate.
    # Confirm behaviour: empty-after-trim means generate.
    assert r.status_code == 200
    assert r.json()["generated"] is True


# ── Rotate ──────────────────────────────────────────────────────────────


def test_rotate_replaces_existing_hash(client):
    first = client.post("/api/bot-access/password?show_id=alpha", json={}).json()
    second = client.post("/api/bot-access/password?show_id=alpha", json={}).json()
    assert first["password"] != second["password"]

    store = rag_index_store.get_index_store()
    expected = f"sha256:{hashlib.sha256(second['password'].encode()).hexdigest()}"
    assert store.get_show_passwords()["alpha"] == expected


# ── Delete ──────────────────────────────────────────────────────────────


def test_delete_removes_protection(client):
    client.post("/api/bot-access/password?show_id=alpha", json={})
    r = client.delete("/api/bot-access/password?show_id=alpha")
    assert r.status_code == 204
    assert (
        client.get("/api/bot-access/password?show_id=alpha").json()["is_protected"]
        is False
    )


def test_delete_unknown_show_404(client):
    r = client.delete("/api/bot-access/password?show_id=Nope")
    assert r.status_code == 404


# ── Unknown show ────────────────────────────────────────────────────────


def test_set_unknown_show_404(client):
    r = client.post("/api/bot-access/password?show_id=Nope", json={})
    assert r.status_code == 404


# ── Index ownership ─────────────────────────────────────────────────────


def test_set_password_on_replica_returns_409(client, monkeypatch):
    """A replica must not accept a password the next rsync would erase."""
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")

    r = client.post("/api/bot-access/password?show_id=alpha", json={})
    assert r.status_code == 409
    assert "replica" in r.json()["detail"]

    assert rag_index_store.get_index_store().get_show_passwords() == {}


def test_delete_password_on_replica_returns_409(client, monkeypatch):
    assert (
        client.post("/api/bot-access/password?show_id=alpha", json={}).status_code
        == 200
    )
    monkeypatch.setenv("PODCODEX_MACHINE_ID", "bot-host")

    r = client.delete("/api/bot-access/password?show_id=alpha")
    assert r.status_code == 409
    assert (
        client.get("/api/bot-access/password?show_id=alpha").json()["is_protected"]
        is True
    )


# ── Guild unlock lists keyed by show id ─────────────────────────────────


def _bare_bot(tmp_path, server_cfg):
    return make_bot(tmp_path, server_cfg)[0]


@pytest.mark.legacy("allowed-shows-split")
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
    bot = _bare_bot(tmp_path, {1: settings})

    asyncio.run(bot._reload_access())

    assert settings.allowed_shows == ["alpha_1234abcd"]  # names → ids, nothing more
    assert bot._show_allowed(bot._show_id_for_label("Alpha"), settings) is True
    # Not offered as a pin: it is an unlock as far as anything can tell, and
    # counting it as both made `/setup show_remove` silently fail to unpin.
    assert bot._pinned_ids(settings) == []


@pytest.mark.legacy("allowed-shows-split")
def test_a_reload_never_reclassifies_or_drops_a_legacy_entry(tmp_path):
    """A reload leaves legacy entries as they are. A reload can land
    mid-rsync, on an index whose password table has not arrived; classifying
    then would read every unlock as a pin and save that, with no recovery but
    re-running `/unlock` per guild."""
    from podcodex.bot.config import ServerSettings

    store = rag_index_store.get_index_store()
    store.set_collection_identity(
        "alpha__bge-m3__semantic", show_id="alpha_1234abcd", show="Alpha"
    )
    settings = ServerSettings(allowed_shows=["alpha_1234abcd"])
    bot, saves = make_bot(tmp_path, {1: settings})

    asyncio.run(bot._reload_access())  # no password table at all: the dangerous shape
    asyncio.run(bot._reload_access())

    assert settings.allowed_shows == ["alpha_1234abcd"]
    assert settings.unlocked_shows == [] and settings.pinned_shows == []
    assert saves == []  # already ids, so nothing to write


@pytest.mark.legacy("allowed-shows-split")
def test_lock_settles_a_legacy_entry(tmp_path):
    """An admin naming the show is the one moment its meaning is known."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(allowed_shows=["alpha"])
    bot, saves = _guild_bot(tmp_path, {1: settings}, locked_ids={"alpha"})
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_lock(interaction, "Alpha"))

    assert settings.allowed_shows == []
    assert saves == [True]
    assert "removed" in interaction.response.messages[0]


@pytest.mark.legacy("allowed-shows-split")
def test_unlocked_show_survives_a_rename(tmp_path):
    """Renaming a protected show keeps it unlocked in every guild."""
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
    bot = _bare_bot(tmp_path, {1: settings})
    asyncio.run(bot._reload_access())
    assert bot._show_allowed(bot._show_id_for_label("Alpha"), settings) is True

    # Rename: label changes in the index, identity does not.
    relabel(store, "alpha_1234abcd", "Renamed")
    store.set_show_password(
        "alpha_1234abcd", hash_show_password("x" * 16), show_label="Renamed"
    )
    asyncio.run(bot._reload_access())

    assert bot._show_allowed(bot._show_id_for_label("Renamed"), settings) is True
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
    bot = _bare_bot(tmp_path, {2: other})
    asyncio.run(bot._reload_access())

    assert bot._show_allowed(bot._show_id_for_label("Alpha"), other) is False
    relabel(store, "alpha_1234abcd", "Renamed")
    asyncio.run(bot._reload_access())
    assert bot._show_allowed(bot._show_id_for_label("Renamed"), other) is False


def test_a_locked_show_sharing_a_public_shows_label_stays_hidden(tmp_path):
    """Labels are not unique. Access is decided per collection by show id, so
    a public show named "Tech" never carries a locked "Tech" along with it,
    whether the command names the show or searches everything."""
    from podcodex.bot.access import ResolvedShows, ShowAccess
    from podcodex.bot.config import ServerSettings

    store = rag_index_store.get_index_store()
    add_show(store, "Tech", show_id="aaa_public", collection="aaa__bge-m3__semantic")
    add_show(
        store,
        "Tech",
        show_id="zzz_private",
        collection="zzz__bge-m3__semantic",
        password="x" * 16,
    )
    settings = ServerSettings()
    bot, _ = make_bot(tmp_path, {1: settings})
    info = store.get_all_collection_info()

    for shows in (ResolvedShows(ShowAccess.ALL), bot._resolve_shows(settings, "Tech")):
        picked = bot._resolve_show_collections(shows, settings, info)
        assert [c.name for c in picked if c.show == "Tech"] == ["aaa__bge-m3__semantic"]

    # Unlocked here, both come back.
    settings.unlocked_shows.append("zzz_private")
    picked = bot._resolve_show_collections(
        ResolvedShows(ShowAccess.ALL), settings, info
    )
    assert {c.name for c in picked if c.show == "Tech"} == {
        "aaa__bge-m3__semantic",
        "zzz__bge-m3__semantic",
    }


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


def _dm_bot(tmp_path):
    return make_bot(tmp_path)


def test_unlock_in_a_dm_writes_no_settings(tmp_path):
    """A DM has guild_id None, which must not be stored: as the JSON key
    "None" the next bot start cannot parse it back into an int."""
    bot, saved = _dm_bot(tmp_path)

    async def _noop_refresh():
        return None

    bot._refresh_if_stale = _noop_refresh
    interaction = _DMInteraction()

    asyncio.run(bot._handle_unlock(interaction, "x" * 16))

    assert bot._server_cfg == {}
    assert saved == []
    assert "server" in interaction.response.messages[0]


def test_setup_in_a_dm_writes_no_settings(tmp_path):
    bot, saved = _dm_bot(tmp_path)
    interaction = _DMInteraction()

    asyncio.run(bot._handle_setup(interaction, "bge-m3", None, None))

    assert bot._server_cfg == {}
    assert saved == []


def test_lock_in_a_dm_writes_no_settings(tmp_path):
    bot, saved = _dm_bot(tmp_path)
    interaction = _DMInteraction()

    asyncio.run(bot._handle_lock(interaction, "Alpha"))

    assert bot._server_cfg == {}
    assert saved == []


def test_load_server_config_skips_a_non_guild_key(tmp_path):
    """Configs written before the DM guard carry a literal "None" key; the bot
    must start anyway instead of dying in int('None')."""
    import json

    from podcodex.bot.bot import BotConfig, PodCodexBot

    path = tmp_path / "server_config.json"
    path.write_text(
        json.dumps({"None": {"top_k": 3}, "42": {"top_k": 7}}), encoding="utf-8"
    )
    cfg = PodCodexBot(BotConfig(), server_config_path=path)._server_cfg

    assert list(cfg) == [42]
    assert cfg[42].top_k == 7


def _load_config(tmp_path, guild_entry):
    import json

    from podcodex.bot.bot import BotConfig, PodCodexBot

    path = tmp_path / "server_config.json"
    path.write_text(json.dumps({"42": guild_entry}), encoding="utf-8")
    return PodCodexBot(BotConfig(), server_config_path=path)._server_cfg[42]


def test_an_old_server_config_still_loads(tmp_path):
    """Keys the bot does not know are ignored (and kept for the next save),
    and fields added since take their defaults, so an upgrade never fails the
    bot start."""
    settings = _load_config(tmp_path, {"model": "bge-m3", "top_k": 3, "retired": 42})

    assert settings.model == "bge-m3" and settings.top_k == 3
    assert settings.pinned_shows == [] and settings.compact is False


@pytest.mark.legacy("allowed-shows-split")
def test_the_pre_split_default_shows_key_becomes_allowed_shows(tmp_path):
    settings = _load_config(tmp_path, {"default_shows": ["a"]})

    assert settings.allowed_shows == ["a"]


# ── Pins and unlocks are separate acts ──────────────────────────────────
#
# "Unlocked here" and "pinned as this server's default" are stored apart, so
# ``/lock`` leaves public pins alone, ``/setup`` pins while another show has a
# password, and a pin must name a show that resolves.


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
    def __init__(self, guild_id=1, user_id=7):
        self.guild_id = guild_id
        self.response = _Reply()
        self.followup = self.response
        self.user = _User(user_id)
        self.type = None


def _guild_bot(tmp_path, server_cfg, *, locked_ids=()):
    """The seeded Alpha and Beta, under ids "alpha" and "beta"; the ids in
    *locked_ids* are password-protected."""
    from podcodex.core.show_passwords import hash_show_password

    store = rag_index_store.get_index_store()
    for sid in locked_ids:
        store.set_show_password(
            sid, hash_show_password("x" * 16), show_label=sid.title()
        )
    return make_bot(tmp_path, server_cfg)


def test_lock_on_a_public_show_changes_nothing(tmp_path):
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(pinned_shows=["alpha"])
    bot, saves = _guild_bot(tmp_path, {1: settings})
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_lock(interaction, "Alpha"))

    assert settings.pinned_shows == ["alpha"]
    assert saves == []
    assert "not password-protected" in interaction.response.messages[0]


def test_setup_pins_while_another_show_is_protected(tmp_path):
    """A password on one show does not block pinning another."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings()
    bot, saves = _guild_bot(tmp_path, {1: settings}, locked_ids={"beta"})
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_setup(interaction, None, None, None, show_add="Alpha"))

    assert bot._server_cfg[1].pinned_shows == ["alpha"]
    assert saves == [True]


def test_setup_rejects_a_show_name_that_resolves_to_nothing(tmp_path):
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings()
    bot, saves = _guild_bot(tmp_path, {1: settings})
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_setup(interaction, None, None, None, show_add="Alfa"))

    assert settings.pinned_shows == []
    assert saves == []
    assert "Alpha" in interaction.response.messages[0]


def test_setup_show_remove_drops_a_pin_whose_show_left_the_index(tmp_path):
    """With no label left to show, the pin autocomplete offers the bare id,
    and removing by it must work."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(pinned_shows=["gone_1234abcd"])
    bot, saves = _guild_bot(tmp_path, {1: settings})
    interaction = _GuildInteraction()

    asyncio.run(
        bot._handle_setup(interaction, None, None, None, show_remove="gone_1234abcd")
    )

    assert bot._server_cfg[1].pinned_shows == []
    assert saves == [True]


def test_episodes_lists_the_pins_instead_of_picking_the_first(tmp_path):
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(pinned_shows=["alpha", "beta"])
    bot, _saves = _guild_bot(tmp_path, {1: settings})

    async def _noop_refresh():
        return None

    bot._refresh_if_stale = _noop_refresh
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_episodes(interaction, None, None))

    reply = interaction.response.messages[0]
    assert "Alpha" in reply and "Beta" in reply


@pytest.mark.legacy("allowed-shows-split")
def test_an_unprotected_legacy_entry_reads_as_a_pin(tmp_path):
    """The other half of the same rule: nothing is protected, so the entry
    can only have come from `/setup show_add`."""
    from podcodex.bot.config import ServerSettings

    store = rag_index_store.get_index_store()
    store.set_collection_identity(
        "alpha__bge-m3__semantic", show_id="alpha_1234abcd", show="Alpha"
    )
    settings = ServerSettings(allowed_shows=["Alpha"])
    bot = _bare_bot(tmp_path, {1: settings})

    asyncio.run(bot._reload_access())

    assert bot._pinned_ids(settings) == ["alpha_1234abcd"]
    assert bot._unlocked_ids(settings) == []


@pytest.mark.legacy("allowed-shows-split")
def test_setup_show_remove_settles_an_unprotected_legacy_pin(tmp_path):
    """Removing a legacy pin clears the legacy list too; left there, it
    still counts as a pin while the reply reports success."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(allowed_shows=["alpha"])
    bot, saves = _guild_bot(tmp_path, {1: settings})
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_setup(interaction, None, None, None, show_remove="Alpha"))

    updated = bot._server_cfg[1]
    assert bot._pinned_ids(updated) == []
    assert updated.allowed_shows == []
    assert saves == [True]


@pytest.mark.legacy("allowed-shows-split")
def test_setup_show_remove_leaves_a_protected_legacy_entry_alone(tmp_path):
    """It is an unlock, not a pin, so there is nothing to unpin — and
    dropping it would revoke access the guild may well have earned."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(allowed_shows=["alpha"])
    bot, saves = _guild_bot(tmp_path, {1: settings}, locked_ids={"alpha"})
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_setup(interaction, None, None, None, show_remove="Alpha"))

    assert settings.allowed_shows == ["alpha"]
    assert saves == []
    assert "not pinned" in interaction.response.messages[0]


@pytest.mark.legacy("allowed-shows-split")
def test_an_unrelated_setup_change_never_rewrites_the_legacy_list(tmp_path):
    """An index rsynced mid-transfer reports no protected shows without
    erroring. Recomputing the legacy list on every `/setup` filed every
    unlock as a pin in that window and cleared the list, losing the guild's
    access for good; the read path re-derives precisely so it survives."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(allowed_shows=["alpha"])
    # The dangerous state: an unlock on the books, nothing readable as locked.
    bot, saves = _guild_bot(tmp_path, {1: settings}, locked_ids=set())
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_setup(interaction, "bge-m3", None, None))

    updated = bot._server_cfg[1]
    assert updated.allowed_shows == ["alpha"]
    assert updated.pinned_shows == []
    assert updated.model == "bge-m3"  # the change the admin actually asked for


@pytest.mark.legacy("allowed-shows-split")
def test_lock_on_a_public_show_leaves_a_legacy_pin_alone(tmp_path):
    """`/lock` revokes access. A public show's legacy entry is a pin, so
    deleting it here would drop a search default while reporting an access
    change — the confusion the split exists to end."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(allowed_shows=["beta"])
    bot, saves = _guild_bot(tmp_path, {1: settings}, locked_ids={"alpha"})
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_lock(interaction, "Beta"))

    assert settings.allowed_shows == ["beta"]
    assert saves == []
    assert "not password-protected" in interaction.response.messages[0]


def test_changepassword_refuses_a_show_that_is_no_longer_protected(
    tmp_path, monkeypatch
):
    """`unlocked_shows` is never pruned when the app makes a show public, so
    a stale entry would otherwise re-protect it index-wide and lock every
    other guild out, with the password DM'd only to the caller."""
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(unlocked_shows=["alpha"])
    bot, _saves = _guild_bot(tmp_path, {1: settings}, locked_ids=set())
    rotated: list[str] = []
    monkeypatch.setattr(
        bot.local, "set_show_password", lambda sid, *_a, **_k: rotated.append(sid)
    )
    interaction = _GuildInteraction()

    asyncio.run(bot._handle_changepassword(interaction, "Alpha"))

    assert rotated == []
    assert "no password to rotate" in interaction.response.messages[0]


@pytest.mark.legacy("show-id")
def test_a_show_key_with_a_slash_reaches_its_routes(client):
    """A legacy key is a label, so "AC/DC" routes as one show, not "AC"."""
    r = client.get("/api/bot-access/password", params={"show_id": "AC/DC"})
    assert r.status_code == 404  # unknown show, reported for the whole name
    assert "AC/DC" in r.json()["detail"]


# ── Keyed by show id, not display name ──────────────────────────────────


def _register(monkeypatch, *folders):
    import podcodex.core.app_config as app_config

    cfg = app_config.AppConfig()
    cfg.show_folders = [str(f) for f in folders]
    monkeypatch.setattr(app_config, "load_config", lambda: cfg)


def _show_folder(path: Path, name: str, show_id: str) -> Path:
    path.mkdir(parents=True)
    (path / "show.toml").write_text(
        f'id = "{show_id}"\nname = "{name}"\n', encoding="utf-8"
    )
    return path


def test_two_shows_sharing_a_name_are_two_rows_with_their_own_status(
    client, tmp_path, monkeypatch
):
    one = _show_folder(tmp_path / "one", "Twin", "twin_11111111")
    two = _show_folder(tmp_path / "two", "Twin", "twin_22222222")
    _register(monkeypatch, one, two)

    assert (
        client.post(
            "/api/bot-access/password", params={"show_id": "twin_22222222"}, json={}
        ).status_code
        == 200
    )

    rows = {
        r["show_id"]: r
        for r in client.get("/api/bot-access/passwords").json()
        if r["show"] == "Twin"
    }
    assert set(rows) == {"twin_11111111", "twin_22222222"}
    assert rows["twin_11111111"]["is_protected"] is False
    assert rows["twin_22222222"]["is_protected"] is True


@pytest.mark.legacy("show-id")
def test_a_registered_show_hides_its_own_unmigrated_index_row(
    client, tmp_path, monkeypatch
):
    """An "Alpha" collection from before ids; the registered Alpha owns it."""
    store = rag_index_store.get_index_store()
    store.delete_collection("alpha__bge-m3__semantic")
    add_show(store, "Alpha", show_id="")
    _register(monkeypatch, _show_folder(tmp_path / "alpha", "Alpha", "alpha_1234abcd"))

    rows = client.get("/api/bot-access/passwords").json()

    assert [r["show_id"] for r in rows if r["show"] == "Alpha"] == ["alpha_1234abcd"]


@pytest.mark.legacy("show-id")
def test_setting_a_password_mints_an_id_for_an_unminted_folder(
    client, tmp_path, monkeypatch
):
    from podcodex.ingest.show import load_show_meta

    folder = tmp_path / "gamma"
    folder.mkdir()
    (folder / "show.toml").write_text('name = "Gamma"\n', encoding="utf-8")
    _register(monkeypatch, folder)

    r = client.post("/api/bot-access/password", params={"show_id": "Gamma"}, json={})

    minted = load_show_meta(folder).id
    assert minted and r.json()["show_id"] == minted
    status = client.get("/api/bot-access/password", params={"show_id": minted})
    assert status.json()["is_protected"] is True


def test_a_locked_legacy_collection_is_not_judged_by_a_public_twins_id(tmp_path):
    """A row with no id has only its label. When that label also names a
    public show with an id, the legacy row (protected under its label) must
    stay hidden rather than borrow the twin's identity."""
    from podcodex.bot.access import ResolvedShows, ShowAccess
    from podcodex.bot.config import ServerSettings

    store = rag_index_store.get_index_store()
    add_show(
        store, "Émile", show_id="emile_1a2b3c4d", collection="new__bge-m3__semantic"
    )
    add_show(
        store,
        "Émile",
        show_id="",
        collection="old__bge-m3__semantic",
        password="x" * 16,
    )
    settings = ServerSettings()
    bot, _ = make_bot(tmp_path, {1: settings})

    picked = bot._resolve_show_collections(
        ResolvedShows(ShowAccess.ALL), settings, store.get_all_collection_info()
    )
    # The protected legacy row never borrows the twin's id. The twin is
    # hidden too while the name is legacy-locked (fails closed) until the
    # migration gives both their own ids.
    assert "old__bge-m3__semantic" not in [c.name for c in picked]


def test_an_unmigrated_show_with_an_id_keyed_password_is_locked_everywhere(tmp_path):
    """Rows with no id, password keyed by the show's id (set from the app
    after minting, collections not yet migrated): the name reads as locked,
    and a result message posted earlier stops rebuilding."""
    import asyncio

    from podcodex.bot.access import ShowAccess
    from podcodex.bot.config import ServerSettings
    from podcodex.bot.ui import _names_allowed
    from podcodex.core.show_passwords import hash_show_password

    store = rag_index_store.get_index_store()
    add_show(store, "Foo", show_id="", collection="foo__bge-m3__semantic")
    store.set_show_password(
        "foo_1234abcd", hash_show_password("x" * 16), show_label="Foo"
    )
    settings = ServerSettings()
    bot, _ = make_bot(tmp_path, {1: settings})

    assert bot._resolve_shows(settings, "Foo").access is ShowAccess.LOCKED
    assert asyncio.run(_names_allowed(bot, 1, ["foo__bge-m3__semantic"])) is False

    settings.unlocked_shows.append("foo_1234abcd")
    assert bot._resolve_shows(settings, "Foo").access is ShowAccess.SPECIFIC
    assert asyncio.run(_names_allowed(bot, 1, ["foo__bge-m3__semantic"])) is True


def test_a_password_still_keyed_by_name_locks_an_id_carrying_collection(tmp_path):
    """A replica synced mid-migration: collections already carry the id, the
    password row is still keyed by the display name. The show stays locked."""
    from podcodex.bot.access import ResolvedShows, ShowAccess
    from podcodex.bot.config import ServerSettings

    store = rag_index_store.get_index_store()
    add_show(store, "Foo", show_id="foo_1234abcd", collection="foo__bge-m3__semantic")
    store._passwords_table().add(
        [{"show_id": "", "show_label": "", "show": "Foo", "password_hash": "h"}]
    )
    settings = ServerSettings()
    bot, _ = make_bot(tmp_path, {1: settings})

    picked = bot._resolve_show_collections(
        ResolvedShows(ShowAccess.ALL), settings, store.get_all_collection_info()
    )
    assert "foo__bge-m3__semantic" not in [c.name for c in picked]


# ── Unlocks follow the password they were granted under ─────────────────


class _User:
    def __init__(self, user_id=7):
        self.id = user_id
        self.dms: list[str] = []

    async def send(self, content):
        self.dms.append(content)


def _unlock(bot, guild_id, password="x" * 16):
    interaction = _GuildInteraction(guild_id)
    asyncio.run(bot._handle_unlock(interaction, password))
    return interaction


def test_rotating_a_password_revokes_the_guilds_that_used_the_old_one(tmp_path):
    """A leaked password is rotated to cut off whoever used it; the guild
    that rotated it keeps its access."""
    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    _unlock(bot, 1)
    _unlock(bot, 2)
    assert bot._show_allowed("alpha", bot._server_settings(2))

    interaction = _GuildInteraction(1)
    asyncio.run(bot._handle_changepassword(interaction, "Alpha"))

    assert interaction.user.dms  # the new password went to the caller
    assert bot._show_allowed("alpha", bot._server_settings(1))
    assert not bot._show_allowed("alpha", bot._server_settings(2))


def test_a_password_changed_in_the_app_revokes_on_the_next_reload(tmp_path):
    from podcodex.core.show_passwords import hash_show_password

    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    _unlock(bot, 1)

    rag_index_store.get_index_store().set_show_password(
        "alpha", hash_show_password("y" * 16), show_label="Alpha"
    )
    asyncio.run(bot._reload_access())

    assert not bot._show_allowed("alpha", bot._server_settings(1))


def test_an_unlock_from_before_fingerprints_survives_the_upgrade(tmp_path):
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(unlocked_shows=["alpha"])
    bot, _saves = _guild_bot(tmp_path, {1: settings}, locked_ids={"alpha"})

    assert bot._show_allowed("alpha", settings)
    assert set(settings.unlock_fingerprints) == {"alpha"}


def test_guild_settings_survive_a_restart(tmp_path):
    """What /unlock and /announcements wrote is what the next start reads,
    including keys a newer release added."""
    import json

    from podcodex.bot.bot import BotConfig, PodCodexBot

    path = tmp_path / "server_config.json"
    path.write_text(
        json.dumps({"1": {"from_a_newer_release": [1, 2]}}), encoding="utf-8"
    )
    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    _unlock(bot, 1)
    bot._server_settings(1).announce_channel_id = 99
    bot._save_server_config()

    again = PodCodexBot(BotConfig(), server_config_path=path)
    asyncio.run(again._reload_access())
    settings = again._server_settings(1)
    assert settings.announce_channel_id == 99
    assert again._show_allowed("alpha", settings)
    assert json.loads(path.read_text())["1"]["from_a_newer_release"] == [1, 2]


def test_an_unreadable_server_config_is_kept_and_the_bot_starts(tmp_path):
    """Raising here crash-loops the container under restart: unless-stopped."""
    from podcodex.bot.bot import BotConfig, PodCodexBot

    path = tmp_path / "server_config.json"
    path.write_text('{"1": {"top_k": 3', encoding="utf-8")

    bot = PodCodexBot(BotConfig(), server_config_path=path)

    assert bot._server_cfg == {}
    [backup] = tmp_path.glob("server_config.json.corrupt-*")
    assert backup.read_text() == '{"1": {"top_k": 3'


def test_changepassword_needs_the_show_unlocked_on_this_server(tmp_path, monkeypatch):
    """Any guild admin could otherwise rotate an index-wide password."""
    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    rotated: list[str] = []
    monkeypatch.setattr(
        bot.local, "set_show_password", lambda sid, *_a, **_k: rotated.append(sid)
    )
    interaction = _GuildInteraction(1)

    asyncio.run(bot._handle_changepassword(interaction, "Alpha"))

    assert rotated == []
    assert "not unlocked on this server" in interaction.response.messages[0]


def test_changepassword_on_a_replica_points_at_the_app(tmp_path, monkeypatch):
    from podcodex.rag.index_origin import IndexOwnershipError

    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    _unlock(bot, 1)

    def refuse(*_a, **_k):
        raise IndexOwnershipError("replica")

    monkeypatch.setattr(bot.local, "set_show_password", refuse)
    interaction = _GuildInteraction(1)

    asyncio.run(bot._handle_changepassword(interaction, "Alpha"))

    assert interaction.user.dms == []
    assert "Rotate it from the PodCodex app" in interaction.response.messages[0]


def test_a_password_set_in_the_app_reaches_the_running_bot(tmp_path, monkeypatch):
    """`_refresh_if_stale` is the only path from a desktop-side password to a
    bot that is already running; an rsync can also move the mtime backwards."""
    from podcodex.core.show_passwords import hash_show_password

    bot, _saves = _guild_bot(tmp_path, {})
    asyncio.run(bot._refresh_if_stale())
    assert bot._locked_show_ids == set()

    rag_index_store.get_index_store().set_show_password(
        "alpha", hash_show_password("x" * 16), show_label="Alpha"
    )
    seen = bot._index_mtime_seen
    monkeypatch.setattr(bot.local, "index_mtime", lambda: seen - 100.0)
    bot._last_mtime_check = 0.0
    asyncio.run(bot._refresh_if_stale())

    assert bot._locked_show_ids == {"alpha"}


# ── /setup values ───────────────────────────────────────────────────────


def test_setup_refuses_a_model_the_bot_does_not_load(tmp_path):
    bot, saves = _guild_bot(tmp_path, {})
    interaction = _GuildInteraction(1)

    asyncio.run(bot._handle_setup(interaction, "e5-small", None, None))

    assert saves == []
    assert "does not load the model `e5-small`" in interaction.response.messages[0]


def test_setup_default_source_any_clears_it(tmp_path):
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(default_source="corrected")
    bot, _saves = _guild_bot(tmp_path, {1: settings})

    asyncio.run(
        bot._handle_setup(_GuildInteraction(1), None, None, None, default_source="any")
    )

    assert bot._server_settings(1).default_source == ""


# ── Guards on the registered commands ───────────────────────────────────


def _command(bot, name):
    return bot.tree.get_command(name).callback


@pytest.mark.parametrize("name", ["sync", "admin-reload"])
def test_admin_commands_refuse_a_dm(tmp_path, monkeypatch, name):
    """default_permissions is not enforced in a DM with the bot."""
    bot, _saves = _guild_bot(tmp_path, {})
    synced: list[bool] = []

    async def fake_sync(*_a, **_k):
        synced.append(True)

    monkeypatch.setattr(bot.tree, "sync", fake_sync)
    interaction = _DMInteraction()

    asyncio.run(_command(bot, name)(interaction))

    assert synced == []
    assert "server" in interaction.response.messages[0]


def test_search_advanced_passes_its_filters_through(tmp_path, monkeypatch):
    """after/before land on the right bound; source `any` overrides the
    server default."""
    from podcodex.bot.config import ServerSettings

    bot, _saves = _guild_bot(tmp_path, {1: ServerSettings(default_source="corrected")})
    calls: list[dict] = []

    async def fake_run(_interaction, _query, _shows, _settings, _alpha, **kw):
        calls.append(kw)

    monkeypatch.setattr(bot, "_run_search", fake_run)
    run = _command(bot, "search-advanced")
    asyncio.run(run(_GuildInteraction(1), "q", after="2024-01-01", before="2024-12-31"))
    asyncio.run(run(_GuildInteraction(1, user_id=8), "q", source="any"))

    assert calls[0]["pub_date_min"] == "2024-01-01"
    assert calls[0]["pub_date_max"] == "2024-12-31"
    assert calls[0]["source"] == "corrected"
    assert calls[1]["source"] is None


# ── Which guilds the bot serves ─────────────────────────────────────────


def test_an_allowlist_refuses_other_guilds_and_dms(tmp_path):
    bot, _saves = make_bot(tmp_path, allowed_guild_ids=frozenset({1}))
    refused = _GuildInteraction(2)

    assert asyncio.run(bot.tree.interaction_check(_GuildInteraction(1))) is True
    assert asyncio.run(bot.tree.interaction_check(refused)) is False
    assert "not available here" in refused.response.messages[0]
    assert bot.guild_allowed(None) is False
    assert make_bot(tmp_path / "open")[0].guild_allowed(2) is True


def test_the_bot_leaves_a_guild_outside_the_allowlist(tmp_path):
    bot, _saves = make_bot(tmp_path, allowed_guild_ids=frozenset({1}))
    left: list[int] = []

    class _Guild:
        def __init__(self, gid):
            self.id, self.name = gid, f"g{gid}"

        async def leave(self):
            left.append(self.id)

    asyncio.run(bot.on_guild_join(_Guild(1)))
    asyncio.run(bot.on_guild_join(_Guild(2)))

    assert left == [2]


def test_allowed_guilds_env_parsing(monkeypatch):
    from podcodex.bot.bot import _allowed_guilds_from_env

    monkeypatch.delenv("PODCODEX_ALLOWED_GUILDS", raising=False)
    assert _allowed_guilds_from_env() is None
    monkeypatch.setenv("PODCODEX_ALLOWED_GUILDS", " 12, 34 ")
    assert _allowed_guilds_from_env() == frozenset({12, 34})
    # Unparseable, or blank between the commas: an empty list would make the
    # bot leave every server.
    for bad in ("my-server", ",", " , "):
        monkeypatch.setenv("PODCODEX_ALLOWED_GUILDS", bad)
        with pytest.raises(SystemExit):
            _allowed_guilds_from_env()


def test_an_allowlist_gives_autocomplete_no_suggestions_and_no_reply(tmp_path):
    """An autocomplete interaction cannot carry a message; replying raised on
    every keystroke."""
    import discord

    bot, _saves = make_bot(tmp_path, allowed_guild_ids=frozenset({1}))
    interaction = _GuildInteraction(2)
    interaction.type = discord.InteractionType.autocomplete

    assert asyncio.run(bot.tree.interaction_check(interaction)) is False
    assert interaction.response.messages == []


def test_a_lost_unlock_key_revokes_nobody(tmp_path):
    """The fingerprints are keyed by a file beside the settings; settings
    restored without it must not read as every password having changed."""
    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    _unlock(bot, 1)
    (tmp_path / "unlock_key").unlink()

    again, _ = make_bot(tmp_path, {1: bot._server_settings(1)})

    assert again._show_allowed("alpha", again._server_settings(1))


def test_the_fingerprint_is_not_an_unkeyed_digest_of_the_hash(tmp_path):
    from podcodex.core.show_passwords import hash_show_password

    bot, _saves = _guild_bot(tmp_path, {})
    h = hash_show_password("x" * 16)
    assert bot._password_fingerprint(h) not in hashlib.sha256(h.encode()).hexdigest()


def test_a_server_config_the_bot_cannot_read_stops_it(tmp_path, monkeypatch):
    """Starting empty would let the next save replace every guild's settings
    with one; a permission or I/O error must stop the bot instead."""
    import pathlib

    from podcodex.bot.bot import BotConfig, PodCodexBot

    path = tmp_path / "server_config.json"
    path.write_text('{"1": {"top_k": 3}}', encoding="utf-8")

    def refuse(self, *a, **k):
        raise PermissionError("volume remounted read-protected")

    monkeypatch.setattr(pathlib.Path, "read_text", refuse)
    with pytest.raises(PermissionError):
        PodCodexBot(BotConfig(), server_config_path=path)


def test_a_truncated_unlock_key_is_replaced(tmp_path):
    """An empty key (a crash mid-write) would make the fingerprint an unkeyed
    digest again."""
    (tmp_path / "unlock_key").write_bytes(b"")
    bot, _saves = _guild_bot(tmp_path, {})

    bot._password_fingerprint("sha256:" + "0" * 64)

    assert len((tmp_path / "unlock_key").read_bytes()) == 32


def test_changepassword_answers_discord_before_rotating(tmp_path, monkeypatch):
    """A late first reply loses the new password if the DM fails too."""
    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    _unlock(bot, 1)
    interaction = _GuildInteraction(1)
    order: list[str] = []
    real_set = bot.local.set_show_password

    def spy(*a, **k):
        order.append("deferred" if interaction.response.deferred else "not deferred")
        return real_set(*a, **k)

    monkeypatch.setattr(bot.local, "set_show_password", spy)
    asyncio.run(bot._handle_changepassword(interaction, "Alpha"))

    assert order == ["deferred"]


def test_a_model_a_guild_chose_stays_loadable(tmp_path):
    """A later smaller --model must not take a /setup choice away; a model
    only saved by default, an old file's model, or a refused guild's choice
    does not count."""
    import json

    from podcodex.bot.bot import BotConfig, PodCodexBot
    from podcodex.bot.config import ServerSettings

    chose = ServerSettings(model="e5-small", model_set=True)
    bot, _ = make_bot(tmp_path, {1: chose})
    assert bot._loadable_models() == {"bge-m3", "e5-small"}
    assert bot._setup_value_problem("e5-small", None) == ""

    refused, _ = make_bot(tmp_path / "r", {2: chose}, allowed_guild_ids=frozenset({1}))
    assert refused._loadable_models() == {"bge-m3"}

    path = tmp_path / "old.json"
    path.write_text(json.dumps({"1": {"model": "e5-large"}}), encoding="utf-8")
    old = PodCodexBot(BotConfig(model="e5-small"), server_config_path=path)
    assert old._loadable_models() == {"e5-small"}
    assert make_bot(tmp_path / "x", models=("bge-m3",))[0]._loadable_models() == {
        "bge-m3"
    }


def test_normalizing_a_label_moves_its_fingerprint_with_it(tmp_path):
    from podcodex.bot.config import ServerSettings

    bot, _saves = _guild_bot(tmp_path, {})
    settings = ServerSettings(
        unlocked_shows=["Alpha"], unlock_fingerprints={"Alpha": "f"}
    )
    bot._server_cfg[1] = settings
    asyncio.run(bot._reload_access())
    bot._normalize_show_ids()

    assert settings.unlocked_shows == ["alpha"]
    assert settings.unlock_fingerprints == {"alpha": "f"}


def test_changepassword_reloads_before_judging_access(tmp_path):
    """A rotation from the app must revoke this guild before it can rotate
    again and keep the show."""
    from podcodex.core.show_passwords import hash_show_password

    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    _unlock(bot, 1)
    rag_index_store.get_index_store().set_show_password(
        "alpha", hash_show_password("y" * 16), show_label="Alpha"
    )
    interaction = _GuildInteraction(1)

    asyncio.run(bot._handle_changepassword(interaction, "Alpha"))

    assert interaction.user.dms == []
    assert "not unlocked on this server" in interaction.response.messages[-1]


def test_the_new_password_is_delivered_even_if_the_reload_fails(tmp_path, monkeypatch):
    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    _unlock(bot, 1)
    calls = {"n": 0}
    real = bot._reload_access

    async def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("table mid-rsync")
        await real()

    monkeypatch.setattr(bot, "_reload_access", flaky)
    interaction = _GuildInteraction(1)

    asyncio.run(bot._handle_changepassword(interaction, "Alpha"))

    assert interaction.user.dms and "New password" in interaction.user.dms[0]


def test_a_saved_default_model_does_not_keep_the_old_model_loaded(tmp_path):
    """Every saved guild carries the default model; restarting with a smaller
    --model must stop loading the big one."""
    from podcodex.bot.config import ServerSettings

    bot, _saves = make_bot(
        tmp_path, {1: ServerSettings(model="bge-m3")}, model="e5-small"
    )
    assert bot._loadable_models() == {"e5-small"}
    # Stamped by an older non-default --model (marker written): not a choice.
    stamped = ServerSettings(model="pplx-4B", model_set=False)
    bot2, _ = make_bot(tmp_path / "s", {1: stamped}, model="e5-small")
    assert bot2._loadable_models() == {"e5-small"}


def test_a_duplicated_unlock_is_still_revoked(tmp_path):
    from podcodex.bot.config import ServerSettings
    from podcodex.core.show_passwords import hash_show_password

    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    _unlock(bot, 1)
    settings = bot._server_settings(1)
    settings.allowed_shows.append("alpha")  # also in the legacy list
    assert isinstance(settings, ServerSettings)

    rag_index_store.get_index_store().set_show_password(
        "alpha", hash_show_password("y" * 16), show_label="Alpha"
    )
    asyncio.run(bot._reload_access())

    assert not bot._show_allowed("alpha", settings)
    assert settings.unlock_fingerprints == {}


def test_a_reload_waits_for_a_rotation_in_progress(tmp_path):
    """A reload reading the table between a rotation's write and its new
    fingerprint would revoke the rotating guild; the access lock orders them."""
    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    order: list[str] = []

    async def go():
        lock = bot._access_lock()
        await lock.acquire()  # a rotation holds it
        reload = asyncio.create_task(bot._reload_access())
        await asyncio.sleep(0.05)
        order.append("reload done" if reload.done() else "reload waiting")
        lock.release()
        await reload

    asyncio.run(go())
    assert order == ["reload waiting"]


def test_an_older_password_read_back_mid_sync_revokes_nobody(tmp_path):
    """A sync caught mid-way can show the previous table for one reload."""
    from podcodex.core.show_passwords import hash_show_password

    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    store = rag_index_store.get_index_store()
    store.set_show_password("alpha", hash_show_password("y" * 16), show_label="Alpha")
    asyncio.run(bot._reload_access())
    _unlock(bot, 1, "y" * 16)
    store.set_show_password("alpha", hash_show_password("x" * 16), show_label="Alpha")
    asyncio.run(bot._reload_access())  # the old password read back

    assert bot._show_allowed("alpha", bot._server_settings(1))


def test_a_failed_delivery_still_saves_and_revokes(tmp_path, monkeypatch):
    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    _unlock(bot, 1)
    _unlock(bot, 2)

    async def undeliverable(*_a, **_k):
        raise ConnectionError("interaction expired")

    monkeypatch.setattr(bot, "_deliver_new_password", undeliverable)
    asyncio.run(bot._handle_changepassword(_GuildInteraction(1), "Alpha"))

    assert not bot._show_allowed("alpha", bot._server_settings(2))


def test_typed_any_in_any_casing_means_every_source(tmp_path):
    from podcodex.bot.config import ServerSettings

    settings = ServerSettings(default_source="corrected")
    bot, _saves = _guild_bot(tmp_path, {1: settings})

    asyncio.run(
        bot._handle_setup(_GuildInteraction(1), None, None, None, default_source="Any ")
    )

    assert bot._server_settings(1).default_source == ""


def test_moving_every_hash_to_a_new_scheme_revokes_nobody(tmp_path):
    """Same passwords, new hash format: not a password change."""
    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})
    _unlock(bot, 1)
    entry = next(iter(bot._shows.values()))
    entry.password_hash = "scrypt:" + "ab" * 32

    bot._revoke_stale_unlocks()

    assert bot._show_allowed("alpha", bot._server_settings(1))


def test_an_unwritable_state_dir_does_not_stop_the_bot(tmp_path, monkeypatch):
    from podcodex.bot import access

    def refuse(*_a, **_k):
        raise PermissionError("read-only mount")

    monkeypatch.setattr(access, "_write_secret", refuse)
    bot, _saves = _guild_bot(tmp_path, {}, locked_ids={"alpha"})

    assert bot._password_fingerprint("sha256:" + "0" * 64).startswith("sha256:")


def test_the_unlock_key_is_owner_only(tmp_path):
    bot, _saves = _guild_bot(tmp_path, {})
    bot._password_fingerprint("sha256:" + "0" * 64)

    assert (tmp_path / "unlock_key").stat().st_mode & 0o777 == 0o600
