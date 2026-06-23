"""Offline simulation of the Discord pagination flow.

Replicates discord.py's dynamic-item dispatch (match custom_id -> from_custom_id
-> callback) against a fake Interaction, so the persistent-button click paths are
exercised end to end without a gateway: paging, jump, expand, transcript nav, and
crucially the two cases that motivated the rewrite -- clicking after a restart
(RAM tier dropped, reload from SQLite) and after eviction (expired message).
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("discord")

import discord  # noqa: E402

from podcodex.bot import ui  # noqa: E402
from podcodex.bot.bot import _chunk_to_ref  # noqa: E402
from podcodex.bot.result_store import CachedSearch, SearchCacheStore  # noqa: E402

# ── Fake index (LanceDB stand-in) ──────────────

_EPISODES = {
    ("col_a", "ep_alpha"): [
        {
            "chunk_index": i,
            "episode": "ep_alpha",
            "show": "Alpha Show",
            "start": i * 12.0,
            "end": i * 12 + 11.0,
            "text": f"alpha segment number {i} about coffee and code",
            "timed": True,
            "score": 0.9 - i * 0.01,
        }
        for i in range(8)
    ],
    ("col_b", "ep_beta"): [
        {
            "chunk_index": i,
            "episode": "ep_beta",
            "show": "Beta Show",
            "start": i * 9.0,
            "end": i * 9 + 8.0,
            "text": f"beta segment {i} discussing cafe culture",
            "timed": True,
            "score": 0.8 - i * 0.02,
        }
        for i in range(5)
    ],
}


class _FakeLocal:
    def load_chunks_no_embeddings(self, collection, episode):
        return [dict(c) for c in _EPISODES[(collection, episode)]]


class _FakeBot:
    def __init__(self, store):
        self.results = store
        self.local = _FakeLocal()


# ── Fake Discord Interaction ───────────────────


class _FakeResponse:
    def __init__(self):
        self.action = self.embed = self.view = self.content = None
        self.ephemeral = False

    async def edit_message(self, *, embed=None, view=None):
        self.action, self.embed, self.view = "edit", embed, view

    async def send_message(
        self, content=None, *, embed=None, view=None, ephemeral=False
    ):
        self.action, self.content = "send", content
        self.embed, self.view, self.ephemeral = embed, view, ephemeral


class _FakeInteraction:
    def __init__(self, client):
        self.client = client
        self.response = _FakeResponse()


async def _click(client, custom_id, *, select_value=None):
    """Dispatch a click the way discord.py routes a component interaction."""
    interaction = _FakeInteraction(client)
    for cls in ui.DYNAMIC_ITEMS:
        m = cls.__discord_ui_compiled_template__.fullmatch(custom_id)
        if m:
            handler = await cls.from_custom_id(interaction, None, m)
            if select_value is not None:
                handler.item._values = [str(select_value)]
            await handler.callback(interaction)
            return interaction.response
    raise AssertionError(f"no dynamic item matched custom_id {custom_id!r}")


def _run(coro):
    return asyncio.run(coro)


def _store():
    return SearchCacheStore(Path(tempfile.mkdtemp()) / "sub" / "search_cache.db")


def _cid(view, suffix=None, prefix=None):
    for c in view.children:
        if suffix and c.custom_id.endswith(suffix):
            return c.custom_id
        if prefix and c.custom_id.startswith(prefix):
            return c.custom_id
    raise AssertionError(f"no custom_id matching prefix={prefix} suffix={suffix}")


def _footer(embed):
    return embed.footer.text if embed and embed.footer else ""


def _search_bot():
    """A bot whose cache holds a 13-result, two-episode search; returns (bot, sid)."""
    store = _store()
    alpha = _EPISODES[("col_a", "ep_alpha")]
    beta = _EPISODES[("col_b", "ep_beta")]
    results = [(c, "col_a") for c in alpha] + [(c, "col_b") for c in beta]
    refs = [_chunk_to_ref(c, col) for c, col in results]
    sid = store.save(CachedSearch("search", "α=0.50", "coffee", refs))
    return _FakeBot(store), sid


# ── Tests ──────────────────────────────────────


def test_initial_page_and_paging():
    bot, sid = _search_bot()

    async def go():
        embed, view = await ui.build_results_view(bot, sid, 0)
        assert "#1 of 13" in _footer(embed)
        assert view.children[0].item.disabled  # prev disabled on page 1
        assert any(c.custom_id.startswith("pcx:rx:") for c in view.children)
        assert any(c.custom_id.startswith("pcx:rj:") for c in view.children)

        next_id = _cid(view, suffix=":n")
        resp = None
        for _ in range(3):
            resp = await _click(bot, next_id)
            assert resp.action == "edit"
            next_id = _cid(resp.view, suffix=":n")
        assert "#4 of 13" in _footer(resp.embed)

        resp = await _click(bot, _cid(resp.view, suffix=":p"))
        assert "#3 of 13" in _footer(resp.embed)

    _run(go())


def test_jump_crosses_into_other_episode():
    bot, sid = _search_bot()

    async def go():
        _, view = await ui.build_results_view(bot, sid, 0)
        jump_id = _cid(view, prefix="pcx:rj:")
        resp = await _click(bot, jump_id, select_value=9)  # 0-based -> result #10
        assert "#10 of 13" in _footer(resp.embed)
        # Result #10 lives in the Beta episode: a different LanceDB fetch on click.
        assert "Beta Show" in (resp.embed.title or "")

    _run(go())


def test_expand_opens_ephemeral_transcript_and_navigates():
    bot, sid = _search_bot()

    async def go():
        _, view = await ui.build_results_view(bot, sid, 0)
        resp = await _click(bot, _cid(view, prefix="pcx:rx:"))
        assert resp.action == "send" and resp.ephemeral
        assert "matched" in _footer(resp.embed)

        before = _footer(resp.embed)
        resp = await _click(bot, _cid(resp.view, suffix=":n"))
        assert resp.action == "edit"
        assert _footer(resp.embed) != before  # segment advanced

    _run(go())


def test_buttons_survive_restart():
    bot, sid = _search_bot()

    async def go():
        _, view = await ui.build_results_view(bot, sid, 2)
        next_id = _cid(view, suffix=":n")
        bot.results._ram.clear()  # simulate process restart: RAM tier gone
        resp = await _click(bot, next_id)
        assert resp.action == "edit"  # rebuilt from SQLite, not "expired"
        assert "of 13" in _footer(resp.embed)

    _run(go())


def test_evicted_search_reports_expired():
    bot, sid = _search_bot()

    async def go():
        _, view = await ui.build_results_view(bot, sid, 0)
        next_id = _cid(view, suffix=":n")
        bot.results._ram.clear()
        bot.results._conn.execute(
            "DELETE FROM search_cache WHERE search_id = ?", (sid,)
        )
        resp = await _click(bot, next_id)
        assert resp.action == "send"
        assert resp.content == ui.EXPIRED_MSG

    _run(go())


def test_list_pagination_verbatim_embeds():
    store = _store()
    bot = _FakeBot(store)
    e1 = discord.Embed(title="🎙 Alpha", description="episodes 1").to_dict()
    e2 = discord.Embed(title="🎙 Alpha", description="episodes 2").to_dict()
    sid = store.save(CachedSearch("list", "", "", embeds=[e1, e2]))

    async def go():
        _, view = await ui.build_list_view(bot, sid, 0)
        resp = await _click(bot, _cid(view, suffix=":n"))
        assert resp.embed.description == "episodes 2"

    _run(go())


def test_random_expand_opens_at_its_chunk():
    store = _store()
    bot = _FakeBot(store)
    alpha = _EPISODES[("col_a", "ep_alpha")]
    sid = store.save(CachedSearch("random", "", "", [_chunk_to_ref(alpha[3], "col_a")]))

    async def go():
        view = discord.ui.View(timeout=None)
        view.add_item(ui.ExpandResult(sid, 0))
        resp = await _click(bot, _cid(view, prefix="pcx:rx:"))
        assert resp.action == "send" and resp.ephemeral
        assert "Segment 4 of 8" in _footer(resp.embed)

    _run(go())
