"""Tests for podcodex.bot.announce — diff state + pure embed builders."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("discord")

from podcodex.bot.announce import (  # noqa: E402
    AnnounceStore,
    build_new_episodes_embed,
    bot_revision,
    build_update_embed,
    changelog_section,
    commit_subjects,
)


def _store(tmp_path):
    return AnnounceStore(tmp_path / "announce_state.db")


# ── AnnounceStore.observe ─────────────────────────────────────────────


def test_first_observation_is_silent_baseline(tmp_path):
    store = _store(tmp_path)
    # A brand-new collection records its whole back-catalogue and announces none.
    assert store.observe("col", {"a", "b", "c"}) == []
    store.close()


def test_only_genuinely_new_stems_returned(tmp_path):
    store = _store(tmp_path)
    store.observe("col", {"a", "b"})  # baseline
    assert store.observe("col", {"a", "b", "c", "d"}) == ["c", "d"]
    # Re-observing the same set yields nothing new.
    assert store.observe("col", {"a", "b", "c", "d"}) == []
    store.close()


def test_collections_are_independent(tmp_path):
    store = _store(tmp_path)
    store.observe("col1", {"a"})  # baseline col1
    # col2's first observation is its own baseline (silent), not a diff vs col1.
    assert store.observe("col2", {"x", "y"}) == []
    assert store.observe("col2", {"x", "y", "z"}) == ["z"]
    store.close()


def test_removed_then_readded_stem_not_reannounced(tmp_path):
    store = _store(tmp_path)
    store.observe("col", {"a", "b"})
    store.observe("col", {"a", "b", "c"})  # c announced, now seen
    # c disappears (rechunk) then returns — already seen, so not re-announced.
    assert store.observe("col", {"a", "b"}) == []
    assert store.observe("col", {"a", "b", "c"}) == []
    store.close()


def test_state_survives_reopen(tmp_path):
    store = _store(tmp_path)
    store.observe("col", {"a", "b"})
    store.close()
    reopened = AnnounceStore(tmp_path / "announce_state.db")
    assert reopened.observe("col", {"a", "b", "c"}) == ["c"]
    reopened.close()


# ── version meta ──────────────────────────────────────────────────────


def test_version_meta_roundtrip(tmp_path):
    store = _store(tmp_path)
    assert store.get_meta("announced_version") is None
    store.set_meta("announced_version", "0.2.4")
    assert store.get_meta("announced_version") == "0.2.4"
    store.set_meta("announced_version", "0.2.5")
    assert store.get_meta("announced_version") == "0.2.5"
    store.close()


# ── embed builders ────────────────────────────────────────────────────


def test_new_episodes_embed_counts_and_thumbnail():
    episodes = [
        {
            "episode_title": "Ep Two",
            "pub_date": "2026-04-21",
            "artwork_url": "https://cdn/x.jpg",
        },
        {"episode_title": "Ep One", "pub_date": "2026-03-10"},
    ]
    embed = build_new_episodes_embed("My Show", episodes)
    assert embed.title == "📣 2 new episodes — My Show"
    assert embed.thumbnail.url == "https://cdn/x.jpg"
    assert "Ep Two · 21 Apr 2026" in embed.description
    assert "Ep One · 10 Mar 2026" in embed.description


def test_new_episodes_embed_omits_absent_date_and_thumbnail():
    embed = build_new_episodes_embed("Show", [{"episode_title": "Solo"}])
    assert embed.title == "📣 1 new episode — Show"
    assert embed.thumbnail.url is None  # no artwork → no thumbnail
    assert "Solo" in embed.description
    assert "·" not in embed.description  # no invented date


def test_new_episodes_embed_truncates_large_batch():
    episodes = [{"episode_title": f"E{i}"} for i in range(20)]
    embed = build_new_episodes_embed("Show", episodes)
    assert "…and 5 more" in embed.description


def test_update_embed_states_what_is_running():
    e = build_update_embed("0.2.5")
    assert e.title == "🔖 PodCodex bot updated"
    assert "v0.2.5" in e.description


def test_update_embed_links_the_commit_when_both_are_known():
    e = build_update_embed(
        "0.2.7", sha="abc1234", repo="https://github.com/o/r", changes=["fix: x"]
    )
    assert "[`abc1234`](https://github.com/o/r/commit/abc1234)" in e.description
    assert "• fix: x" in e.description


def test_update_embed_shows_the_bare_commit_without_a_repo_url():
    e = build_update_embed("0.2.7", sha="abc1234")
    assert "`abc1234`" in e.description
    assert "http" not in e.description


def test_update_embed_omits_the_commit_it_does_not_know():
    # The Docker image ships no .git, so the commit is simply unknown.
    e = build_update_embed("0.2.7")
    assert e.description == "Now running **v0.2.7**"


def test_update_embed_prefers_release_notes_over_the_commit_list():
    e = build_update_embed(
        "0.2.7", sha="abc1234", notes="### Fixes\n- thing", changes=["fix: x"]
    )
    assert "### Fixes" in e.description
    assert "fix: x" not in e.description


def test_version_notes_come_from_the_changelog():
    import podcodex

    section = changelog_section(podcodex.__version__)
    assert section, "the running version should have a CHANGELOG entry"
    assert build_update_embed(podcodex.__version__, notes=section).description.endswith(
        section[-40:]
    )


def test_changelog_section_stops_at_the_next_version():
    section = changelog_section("0.2.6")
    assert section
    assert "## [" not in section  # did not bleed into 0.2.5


def test_changelog_section_empty_for_an_unknown_version():
    assert changelog_section("9.9.9") == ""


def test_bot_revision_reports_the_running_version():
    """An empty version silently turns off update announcements
    (_announce_version_if_changed returns early on it)."""
    import podcodex

    version, sha = bot_revision()
    assert version == podcodex.__version__
    # sha is "" wherever the checkout is unavailable; here it is a real repo.
    assert sha == "" or len(sha) >= 7


def test_commit_subjects_needs_both_revisions():
    assert commit_subjects("", "abc1234") == []
    assert commit_subjects("abc1234", "") == []


def test_changelog_falls_back_to_cwd(tmp_path, monkeypatch):
    """Non-editable installs have no repo checkout; the Docker image copies
    CHANGELOG.md to the WORKDIR, so cwd is the fallback."""
    from podcodex.bot import announce as announce_mod

    monkeypatch.setattr(announce_mod, "_REPO_ROOT", tmp_path / "not-a-checkout")
    workdir = tmp_path / "app"
    workdir.mkdir()
    (workdir / "CHANGELOG.md").write_text("## [1.2.3]\n- shipped note\n## [1.2.2]\n")
    monkeypatch.chdir(workdir)
    assert changelog_section("1.2.3") == "- shipped note"


def test_changelog_empty_when_no_file_anywhere(tmp_path, monkeypatch):
    from podcodex.bot import announce as announce_mod

    monkeypatch.setattr(announce_mod, "_REPO_ROOT", tmp_path / "not-a-checkout")
    monkeypatch.chdir(tmp_path)
    assert changelog_section("1.2.3") == ""


def test_git_refuses_foreign_root(tmp_path, monkeypatch):
    """A site-packages install nested in some unrelated git checkout must not
    report that repo's HEAD as the bot's revision."""
    from podcodex.bot import announce as announce_mod

    monkeypatch.setattr(announce_mod, "_REPO_ROOT", tmp_path)  # no .git here
    _version, sha = bot_revision()
    assert sha == ""


# ── announce tick orchestration (real bot + seeded index, fake channel) ──


def _seed(tmp_path, shows):
    from podcodex.rag.index_store import IndexStore
    from tests.fixtures.index import add_show, chunk

    store = IndexStore(tmp_path / "index")
    for show, stems in shows.items():
        add_show(
            store,
            show,
            {
                stem: [
                    chunk(
                        "long enough sentence to survive the chunker minimum "
                        "length filter here",
                        episode=stem,
                        show=show,
                        episode_title=stem.upper(),
                        pub_date="2026-04-21",
                    )
                ]
                for stem in stems
            },
        )
    return store


class _FakeChannel:
    def __init__(self):
        self.embeds = []

    async def send(self, *, embed):
        self.embeds.append(embed)


def _bot(tmp_path):
    from podcodex.bot.bot import BotConfig, PodCodexBot

    bot = PodCodexBot(
        BotConfig(index_path=str(tmp_path / "index")),
        server_config_path=tmp_path / "server_config.json",
    )
    _ = bot.local
    return bot


def test_tick_baselines_silently_then_announces_new(tmp_path):
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    fake = _FakeChannel()
    bot.get_channel = lambda cid: fake  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)

    # First tick: baseline the existing catalogue, announce nothing.
    asyncio.run(bot._run_announce_tick())
    assert fake.embeds == []

    # A new episode is indexed; force the mtime gate open and tick again.
    _seed(tmp_path, {"Alpha": ["ep2"]})  # adds ep2 to same collection
    bot._announce_mtime_seen = 0.0
    asyncio.run(bot._run_announce_tick())
    assert len(fake.embeds) == 1
    assert "Alpha" in fake.embeds[0].title
    assert "EP2" in fake.embeds[0].description


def test_tick_respects_locked_show_access(tmp_path):
    from podcodex.bot.bot import ServerSettings

    store = _seed(tmp_path, {"Pub": ["p1"], "Secret": ["s1"]})
    bot = _bot(tmp_path)
    from tests.fixtures.index import show_id_of

    store.set_show_password(show_id_of(store, "Secret"), "sha256:" + "0" * 64)
    asyncio.run(bot._reload_access())
    fake = _FakeChannel()
    bot.get_channel = lambda cid: fake  # type: ignore[assignment]
    # Guild has NOT unlocked "Secret".
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)

    asyncio.run(bot._run_announce_tick())  # baseline
    _seed(tmp_path, {"Pub": ["p2"], "Secret": ["s2"]})  # new in both
    bot._announce_mtime_seen = 0.0
    asyncio.run(bot._run_announce_tick())

    titles = " ".join(e.title for e in fake.embeds)
    assert "Pub" in titles
    assert "Secret" not in titles  # locked show never leaks


def test_tick_keeps_watermark_when_reload_fails(tmp_path):
    """A tick that dies mid-reload must be retried, not swallowed.

    ``_announce_loop`` catches every exception, so advancing the watermark
    before the reload would mark a half-written rsync as seen and drop those
    episodes permanently.
    """
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    fake = _FakeChannel()
    bot.get_channel = lambda cid: fake  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)

    asyncio.run(bot._run_announce_tick())  # baseline
    baseline = bot._announce_mtime_seen
    assert baseline > 0

    _seed(tmp_path, {"Alpha": ["ep2"]})

    def _boom():
        raise OSError("manifest landed before its data files")

    bot.local.reconnect = _boom  # type: ignore[method-assign]
    with pytest.raises(OSError):
        asyncio.run(bot._run_announce_tick())
    assert bot._announce_mtime_seen == baseline
    assert fake.embeds == []

    # Recovered: the retry still sees the change and announces it.
    del bot.local.reconnect
    asyncio.run(bot._run_announce_tick())
    assert bot._announce_mtime_seen != baseline
    assert len(fake.embeds) == 1
    assert "EP2" in fake.embeds[0].description


def test_tick_detects_mtime_moving_backwards(tmp_path):
    """``rsync -a`` stamps source mtimes, so the signal can drop, not just rise.

    Syncing an older dev-machine copy over a table last written on the bot
    host lowers the value. A ``<=`` gate would treat that as "nothing new"
    forever after.
    """
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    fake = _FakeChannel()
    bot.get_channel = lambda cid: fake  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)

    asyncio.run(bot._run_announce_tick())  # baseline
    _seed(tmp_path, {"Alpha": ["ep2"]})

    # Watermark stranded above the on-disk value, as a backwards sync leaves it.
    bot._announce_mtime_seen = bot.local.index_mtime() + 10_000
    asyncio.run(bot._run_announce_tick())

    assert len(fake.embeds) == 1
    assert "EP2" in fake.embeds[0].description


def test_tick_that_fails_after_diffing_announces_on_the_retry(tmp_path):
    """The new stems used to be marked seen before their details were read,
    so a failure there (a read mid-rsync) lost the announcement for good."""
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    fake = _FakeChannel()
    bot.get_channel = lambda cid: fake  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)
    asyncio.run(bot._run_announce_tick())  # baseline

    _seed(tmp_path, {"Alpha": ["ep2"]})
    real_stats = bot.local.get_episode_stats

    def _boom(_col):
        raise OSError("data file not there yet")

    bot.local.get_episode_stats = _boom  # type: ignore[method-assign]
    asyncio.run(bot._run_announce_tick())
    assert fake.embeds == []

    bot.local.get_episode_stats = real_stats  # type: ignore[method-assign]
    asyncio.run(bot._run_announce_tick())
    assert len(fake.embeds) == 1
    assert "EP2" in fake.embeds[0].description


def test_fan_out_survives_a_guild_configured_mid_tick(tmp_path):
    """Iterating the live settings dict across awaits raised "dictionary
    changed size" when a guild ran /setup during the fan-out."""
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    channels = {999: _FakeChannel(), 998: _FakeChannel()}
    bot.get_channel = lambda cid: None  # type: ignore[assignment]

    async def fetch(cid):
        bot._server_cfg[len(bot._server_cfg) + 100] = ServerSettings()
        return channels[cid]

    bot.fetch_channel = fetch  # type: ignore[assignment]
    bot._server_cfg[41] = ServerSettings(announce_channel_id=999)
    bot._server_cfg[42] = ServerSettings(announce_channel_id=998)
    asyncio.run(bot._run_announce_tick())  # baseline
    _seed(tmp_path, {"Alpha": ["ep2"]})
    bot._announce_mtime_seen = 0.0

    asyncio.run(bot._run_announce_tick())

    assert [len(c.embeds) for c in channels.values()] == [1, 1]


def test_one_failing_collection_does_not_hold_back_the_others(tmp_path):
    """Each collection is announced and recorded on its own: a table that
    keeps failing is retried, the rest are posted once."""
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["a1"], "Beta": ["b1"]})
    bot = _bot(tmp_path)
    fake = _FakeChannel()
    bot.get_channel = lambda cid: fake  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)
    asyncio.run(bot._run_announce_tick())  # baseline

    _seed(tmp_path, {"Alpha": ["a2"], "Beta": ["b2"]})
    real_stats = bot.local.get_episode_stats

    def stats(col):
        if col.startswith("beta"):
            raise OSError("damaged table")
        return real_stats(col)

    bot.local.get_episode_stats = stats  # type: ignore[method-assign]
    asyncio.run(bot._run_announce_tick())
    asyncio.run(bot._run_announce_tick())  # Beta retried, Alpha not re-posted

    assert [e.title for e in fake.embeds].count(fake.embeds[0].title) == 1
    assert "Alpha" in fake.embeds[0].title and len(fake.embeds) == 1

    bot.local.get_episode_stats = real_stats  # type: ignore[method-assign]
    asyncio.run(bot._run_announce_tick())
    assert len(fake.embeds) == 2 and "Beta" in fake.embeds[1].title


def test_a_guild_whose_send_fails_does_not_stop_the_others(tmp_path):
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    good = _FakeChannel()

    class _Broken:
        async def send(self, *, embed):
            raise ConnectionResetError("gateway dropped")

    bot.get_channel = lambda cid: _Broken() if cid == 998 else good  # type: ignore[assignment]
    bot._server_cfg[41] = ServerSettings(announce_channel_id=998)
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)
    asyncio.run(bot._run_announce_tick())  # baseline
    _seed(tmp_path, {"Alpha": ["ep2"]})
    bot._announce_mtime_seen = 0.0

    asyncio.run(bot._run_announce_tick())
    asyncio.run(bot._run_announce_tick())

    assert len(good.embeds) == 1


def test_an_episode_with_no_stats_row_yet_is_announced_later(tmp_path):
    """Mid-rsync the episode list can be ahead of the stats; marking those
    stems seen anyway lost them."""
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    fake = _FakeChannel()
    bot.get_channel = lambda cid: fake  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)
    asyncio.run(bot._run_announce_tick())  # baseline

    _seed(tmp_path, {"Alpha": ["ep2"]})
    real_stats = bot.local.get_episode_stats
    bot.local.get_episode_stats = lambda col: [  # type: ignore[method-assign]
        s for s in real_stats(col) if s.get("episode") != "ep2"
    ]
    asyncio.run(bot._run_announce_tick())
    assert fake.embeds == []

    asyncio.run(bot._run_announce_tick())  # quiet tick: only a sync can fix it
    assert fake.embeds == [] and bot._announce_retry == {}

    bot.local.get_episode_stats = real_stats  # type: ignore[method-assign]
    bot._announce_mtime_seen = 0.0  # the rest of the sync lands
    asyncio.run(bot._run_announce_tick())
    assert len(fake.embeds) == 1 and "EP2" in fake.embeds[0].description


def test_a_collection_that_keeps_failing_is_given_up_without_reloading(tmp_path):
    """Retries touch only that collection (no reconnect, no cache clear), and
    stop after a bound."""
    from podcodex.bot import announce_tasks
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    bot.get_channel = lambda cid: _FakeChannel()  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)
    asyncio.run(bot._run_announce_tick())  # baseline
    _seed(tmp_path, {"Alpha": ["ep2"]})

    def _boom(_col):
        raise OSError("damaged table")

    bot.local.get_episode_stats = _boom  # type: ignore[method-assign]
    asyncio.run(bot._run_announce_tick())
    reconnects: list[bool] = []
    real_reconnect = bot.local.reconnect
    bot.local.reconnect = lambda: reconnects.append(True) or real_reconnect()  # type: ignore[method-assign]
    for _ in range(announce_tasks._ANNOUNCE_RETRIES + 2):
        asyncio.run(bot._run_announce_tick())

    assert reconnects == []
    assert bot._announce_retry == {}


def test_announcements_skip_guilds_outside_the_allowlist(tmp_path):
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    bot.config.allowed_guild_ids = frozenset({42})
    chans = {998: _FakeChannel(), 999: _FakeChannel()}
    bot.get_channel = lambda cid: chans[cid]  # type: ignore[assignment]
    bot._server_cfg[41] = ServerSettings(announce_channel_id=998)
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)
    asyncio.run(bot._run_announce_tick())  # baseline
    _seed(tmp_path, {"Alpha": ["ep2"]})
    bot._announce_mtime_seen = 0.0

    asyncio.run(bot._run_announce_tick())

    assert (len(chans[998].embeds), len(chans[999].embeds)) == (0, 1)


def test_a_retry_for_a_collection_that_left_the_index_is_dropped(tmp_path):

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    asyncio.run(bot._run_announce_tick())
    bot._announce_retry["gone__bge-m3__semantic"] = 1
    read: list[str] = []
    real = bot.local.list_episodes
    bot.local.list_episodes = lambda c: read.append(c) or real(c)  # type: ignore[method-assign]

    asyncio.run(bot._run_announce_tick())

    assert read == [] and bot._announce_retry == {}


def test_episodes_wait_while_no_announce_channel_is_reachable(tmp_path):
    """A cold channel cache plus a failing fetch left nobody to tell, and the
    episodes were marked announced anyway."""
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    fake = _FakeChannel()
    reachable = {"yes": False}
    bot.get_channel = lambda cid: fake if reachable["yes"] else None  # type: ignore[assignment]

    async def fetch(cid):
        raise ConnectionError("discord 503")

    bot.fetch_channel = fetch  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)
    asyncio.run(bot._run_announce_tick())  # baseline
    _seed(tmp_path, {"Alpha": ["ep2"]})
    asyncio.run(bot._run_announce_tick())
    assert fake.embeds == []

    reachable["yes"] = True
    asyncio.run(bot._run_announce_tick())
    assert len(fake.embeds) == 1


def test_giving_up_marks_the_episodes_seen(tmp_path):
    """Otherwise they come back on every later index change, or are
    announced as new once their stats finally land."""
    from podcodex.bot import announce_tasks
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    fake = _FakeChannel()
    bot.get_channel = lambda cid: fake  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)
    asyncio.run(bot._run_announce_tick())  # baseline
    _seed(tmp_path, {"Alpha": ["ep2"]})
    real_stats = bot.local.get_episode_stats
    bot.local.get_episode_stats = lambda col: [  # type: ignore[method-assign]
        s for s in real_stats(col) if s.get("episode") != "ep2"
    ]
    for _ in range(announce_tasks._ANNOUNCE_RETRIES + 1):
        bot._announce_mtime_seen = 0.0  # an index change each time, still no stats
        asyncio.run(bot._run_announce_tick())

    bot.local.get_episode_stats = real_stats  # type: ignore[method-assign]
    bot._announce_mtime_seen = 0.0
    asyncio.run(bot._run_announce_tick())
    assert fake.embeds == []


def test_a_dead_channel_elsewhere_does_not_hold_back_a_show_nobody_sees(tmp_path):
    from podcodex.bot.bot import ServerSettings

    store = _seed(tmp_path, {"Secret": ["s1"]})
    bot = _bot(tmp_path)
    from tests.fixtures.index import show_id_of

    store.set_show_password(show_id_of(store, "Secret"), "sha256:" + "0" * 64)
    asyncio.run(bot._reload_access())
    bot.get_channel = lambda cid: None  # type: ignore[assignment]

    async def fetch(cid):
        raise ConnectionError("channel deleted")

    bot.fetch_channel = fetch  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)  # not unlocked
    asyncio.run(bot._run_announce_tick())  # baseline
    _seed(tmp_path, {"Secret": ["s2"]})
    asyncio.run(bot._run_announce_tick())

    assert bot._announce_retry == {}


def test_giving_up_on_an_unreadable_collection_rebaselines_it(tmp_path):
    """Its episodes could not be marked; once it reads again, what it holds
    is taken silently instead of posted as a burst of "new" episodes."""
    from podcodex.bot import announce_tasks
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    fake = _FakeChannel()
    bot.get_channel = lambda cid: fake  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)
    asyncio.run(bot._run_announce_tick())  # baseline
    _seed(tmp_path, {"Alpha": ["ep2"]})
    real = bot.local.list_episodes

    def _boom(_col):
        raise OSError("damaged table")

    bot.local.list_episodes = _boom  # type: ignore[method-assign]
    for _ in range(announce_tasks._ANNOUNCE_RETRIES + 1):
        asyncio.run(bot._run_announce_tick())

    bot.local.list_episodes = real  # type: ignore[method-assign]
    _seed(tmp_path, {"Alpha": ["ep3"]})
    asyncio.run(bot._run_announce_tick())
    assert fake.embeds == []


def test_a_new_waiting_episode_gets_its_own_count(tmp_path):
    from podcodex.bot import announce_tasks

    bot = _bot(tmp_path)
    for _ in range(announce_tasks._ANNOUNCE_RETRIES):
        bot._wait_for_stats("c", ["a"], changed=True)
    bot._wait_for_stats("c", ["b"], changed=True)  # a announced, b just arrived
    bot._wait_for_stats("c", ["b"], changed=False)  # a quiet tick counts nothing

    assert bot._announce_pending["c"] == (frozenset({"b"}), 1)


def test_a_failed_record_after_posting_does_not_post_again(tmp_path):
    from podcodex.bot.bot import ServerSettings

    _seed(tmp_path, {"Alpha": ["ep1"]})
    bot = _bot(tmp_path)
    fake = _FakeChannel()
    bot.get_channel = lambda cid: fake  # type: ignore[assignment]
    bot._server_cfg[42] = ServerSettings(announce_channel_id=999)
    asyncio.run(bot._run_announce_tick())  # baseline
    _seed(tmp_path, {"Alpha": ["ep2"]})

    def disk_full(*_a, **_k):
        raise OSError("disk full")

    bot.announce.mark_seen = disk_full  # type: ignore[method-assign]
    asyncio.run(bot._run_announce_tick())
    asyncio.run(bot._run_announce_tick())

    assert len(fake.embeds) == 1
