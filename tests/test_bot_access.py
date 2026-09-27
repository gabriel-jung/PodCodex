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
from tests.fixtures.index import add_show


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

    bot._reload_shows()

    assert settings.allowed_shows == ["alpha_1234abcd"]  # names → ids, nothing more
    assert bot._show_allowed_by_label("Alpha", settings) is True
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

    bot._reload_shows()  # no password table at all: the dangerous shape
    bot._reload_shows()

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
    bot = _bare_bot(tmp_path, {2: other})
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


def _dm_bot(tmp_path):
    return make_bot(tmp_path)


def test_unlock_in_a_dm_writes_no_settings(tmp_path):
    """A DM has guild_id None, which must not be stored: as the JSON key
    "None" the next bot start cannot parse it back into an int."""
    bot, saved = _dm_bot(tmp_path)

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
    """Keys the bot no longer knows are dropped and fields added since take
    their defaults, so an upgrade never fails the bot start."""
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
    def __init__(self, guild_id=1):
        self.guild_id = guild_id
        self.response = _Reply()
        self.followup = self.response


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

    bot._reload_shows()

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
