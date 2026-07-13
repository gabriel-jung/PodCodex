"""Tests for podcodex.bot.announce — diff state + pure embed builders."""

from __future__ import annotations

import pytest

pytest.importorskip("discord")

from podcodex.bot.announce import (  # noqa: E402
    AnnounceStore,
    build_new_episodes_embed,
    build_version_embed,
    changelog_section,
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
    assert "Ep Two · Apr 2026" in embed.description
    assert "Ep One · Mar 2026" in embed.description


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


def test_version_embed():
    assert build_version_embed("0.2.5").title == "🔖 PodCodex bot v0.2.5"


def test_version_embed_carries_the_changelog_section():
    import podcodex

    embed = build_version_embed(podcodex.__version__)
    section = changelog_section(podcodex.__version__)
    assert section, "the running version should have a CHANGELOG entry"
    # The notes are shown, not summarized: the card body is the section itself.
    assert embed.description
    assert embed.description.startswith(section[:40])


def test_changelog_section_stops_at_the_next_version():
    section = changelog_section("0.2.6")
    assert section
    assert "## [" not in section  # did not bleed into 0.2.5


def test_version_embed_omits_notes_it_cannot_read():
    # An unknown version (and, on the Docker image, a missing CHANGELOG) must
    # degrade to the bare card rather than inventing a summary.
    assert changelog_section("9.9.9") == ""
    assert build_version_embed("9.9.9").description is None


def test_changelog_section_empty_when_file_is_absent(monkeypatch, tmp_path):
    import podcodex.bot.announce as announce_mod

    monkeypatch.setattr(announce_mod, "__file__", str(tmp_path / "bot" / "x.py"))
    assert announce_mod.changelog_section("0.2.7") == ""


# ── announce tick orchestration (real bot + seeded index, fake channel) ──


def _seed(tmp_path, shows):
    import numpy as np
    from podcodex.rag.index_store import IndexStore

    store = IndexStore(tmp_path / "index")
    for show, stems in shows.items():
        col = f"{show.lower()}__bge-m3__semantic"
        store.ensure_collection(
            col, show=show, model="bge-m3", chunker="semantic", dim=8
        )
        for stem in stems:
            chunks = [
                {
                    "text": "long enough sentence to survive the chunker minimum length filter here",
                    "episode": stem,
                    "show": show,
                    "source": "transcript",
                    "dominant_speaker": "sp",
                    "start": 0.0,
                    "end": 1.0,
                    "episode_title": stem.upper(),
                    "pub_date": "2026-04-21",
                }
            ]
            store.save_chunks(
                col,
                stem,
                chunks,
                np.random.default_rng(0).random((1, 8), dtype=np.float32),
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
    import asyncio
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
    import asyncio
    from podcodex.bot.bot import ServerSettings

    store = _seed(tmp_path, {"Pub": ["p1"], "Secret": ["s1"]})
    bot = _bot(tmp_path)
    store.set_show_password("Secret", "sha256:" + "0" * 64)
    bot._reload_shows()
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
