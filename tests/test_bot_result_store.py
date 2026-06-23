"""Tests for the persistent search-result cache and its stateless views."""

import asyncio
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("discord")

import discord  # noqa: E402

from podcodex.bot import ui  # noqa: E402
from podcodex.bot.result_store import (  # noqa: E402
    CachedSearch,
    ResultRef,
    SearchCacheStore,
    _decode,
    _encode,
)


def _store() -> SearchCacheStore:
    # Parent intentionally missing to exercise the mkdir guard.
    return SearchCacheStore(Path(tempfile.mkdtemp()) / "sub" / "search_cache.db")


_REFS = [
    ResultRef("col_a", "ep_042", 2, 0.83, episode_title="The Big One"),
    ResultRef(
        "col_a",
        "ep_042",
        4,
        0.5,
        fuzzy_match=True,
        match_text="seg",
        episode_title="The Big One",
    ),
    ResultRef("col_b", "ep_009", 1, 0.4, accent_match=True, episode_title="Pilot"),
]


# ── encode / decode ───────────────────────────


def test_refs_roundtrip_preserves_all_fields():
    refs, embeds = _decode(_encode(CachedSearch("search", "l", "q", _REFS)))
    assert embeds == []
    assert refs == _REFS


def test_flags_bitpack():
    ref = ResultRef("c", "e", 0, fuzzy_match=True, accent_match=True)
    assert ref.flags == 3
    refs, _ = _decode(_encode(CachedSearch("exact", "l", "q", [ref])))
    assert refs[0].fuzzy_match and refs[0].accent_match


def test_episode_titles_are_deduped():
    # Two refs share an episode; the title list interns one entry per episode.
    import json

    payload = _encode(CachedSearch("search", "l", "q", _REFS))
    data = json.loads(payload)
    assert data["eps"] == ["ep_042", "ep_009"]
    assert data["ept"] == ["The Big One", "Pilot"]


def test_embeds_payload_roundtrip():
    e1 = discord.Embed(title="Show A").to_dict()
    e2 = discord.Embed(title="Show A pg2").to_dict()
    refs, embeds = _decode(_encode(CachedSearch("list", "l", "", embeds=[e1, e2])))
    assert refs == []
    assert embeds == [e1, e2]


# ── store tiers ───────────────────────────────


def test_save_load_ram_then_sqlite():
    store = _store()
    sid = store.save(CachedSearch("search", "lbl", "q", _REFS))
    assert store.load(sid).refs == _REFS  # RAM
    store._ram.clear()
    assert store.load(sid).refs == _REFS  # SQLite
    store.close()


def test_missing_id_returns_none():
    store = _store()
    assert store.load("deadbeef") is None
    store.close()


def test_corrupt_row_is_dropped_as_miss():
    store = _store()
    sid = store.save(CachedSearch("search", "l", "q", _REFS))
    store._ram.clear()
    store._conn.execute(
        "UPDATE search_cache SET payload = ? WHERE search_id = ?", ("{bad", sid)
    )
    assert store.load(sid) is None
    assert store.load(sid) is None  # row removed; still a clean miss
    store.close()


# ── view builders ─────────────────────────────


class _FakeLocal:
    def load_chunks_no_embeddings(self, collection, episode):
        return [
            {
                "chunk_index": i,
                "episode": episode,
                "show": "My Show",
                "start": i * 10.0,
                "end": i * 10 + 9.0,
                "text": f"segment {i}",
                "timed": True,
            }
            for i in range(5)
        ]


class _FakeBot:
    def __init__(self, store):
        self.results = store
        self.local = _FakeLocal()


def _run(coro):
    return asyncio.run(coro)


def test_build_results_view_components_and_clamping():
    store = _store()
    sid = store.save(CachedSearch("search", "lbl", "q", _REFS))
    bot = _FakeBot(store)
    embed, view = _run(ui.build_results_view(bot, sid, 0))
    cids = [c.custom_id for c in view.children]
    assert cids[0] == f"pcx:r:{sid}:0:p"
    assert "pcx:noop" in cids
    assert f"pcx:rx:{sid}:0" in cids  # expand
    assert f"pcx:rj:{sid}" in cids  # jump (n > 1)
    assert "#1 of 3" in embed.footer.text
    # Out-of-range index clamps to the last page.
    embed_last, _ = _run(ui.build_results_view(bot, sid, 99))
    assert "#3 of 3" in embed_last.footer.text
    store.close()


def test_jump_label_uses_cached_title():
    store = _store()
    sid = store.save(CachedSearch("search", "lbl", "q", _REFS))
    bot = _FakeBot(store)
    _, view = _run(ui.build_results_view(bot, sid, 0))
    jump = next(c for c in view.children if isinstance(c, ui.ResultJump))
    assert jump.item.options[0].label == "#1 • The Big One"
    store.close()


def test_exact_highlights_search_does_not():
    store = _store()
    bot = _FakeBot(store)
    # /exact hits carry a match_text; /search hits do not.
    ref_x = ResultRef("c", "ep_042", 0, 0.5, match_text="segment 0", episode_title="t")
    ref_s = ResultRef("c", "ep_042", 0, 0.5, episode_title="t")
    sid_x = store.save(CachedSearch("exact", "l", "segment", [ref_x]))
    sid_s = store.save(CachedSearch("search", "l", "segment", [ref_s]))
    emb_x, _ = _run(ui.build_results_view(bot, sid_x, 0))
    emb_s, _ = _run(ui.build_results_view(bot, sid_s, 0))
    assert "__segment 0__" in emb_x.description  # highlighted
    assert "__" not in emb_s.description  # plain, no highlight markers
    store.close()


def test_build_transcript_view_opens_at_match():
    store = _store()
    sid = store.save(CachedSearch("search", "lbl", "q", _REFS))
    bot = _FakeBot(store)
    embed, view = _run(ui.build_transcript_view(bot, sid, 0, None))
    assert "Segment 3 of 5 ◀ matched" in embed.footer.text  # ref.chunk_index == 2
    assert any(c.custom_id == f"pcx:t:{sid}:0:2:p" for c in view.children)
    store.close()


def test_build_list_view():
    store = _store()
    e1 = discord.Embed(title="A").to_dict()
    e2 = discord.Embed(title="A pg2").to_dict()
    sid = store.save(CachedSearch("list", "l", "", embeds=[e1, e2]))
    bot = _FakeBot(store)
    embed, view = _run(ui.build_list_view(bot, sid, 1))
    assert embed.title == "A pg2"
    assert [c.custom_id for c in view.children] == [
        f"pcx:l:{sid}:1:p",
        "pcx:noop",
        f"pcx:l:{sid}:1:n",
    ]
    store.close()


def test_builders_return_none_on_miss():
    bot = _FakeBot(_store())
    assert _run(ui.build_results_view(bot, "deadbeef", 0)) is None
    assert _run(ui.build_transcript_view(bot, "deadbeef", 0, None)) is None
    assert _run(ui.build_list_view(bot, "deadbeef", 0)) is None


# ── dynamic-item templates ────────────────────


def test_dynamic_templates_match_and_dont_collide():
    cases = {
        ui.ResultNav: "pcx:r:abc123:3:n",
        ui.ResultJump: "pcx:rj:abc123",
        ui.ExpandResult: "pcx:rx:abc123:3",
        ui.TranscriptNav: "pcx:t:abc123:3:17:p",
        ui.ListNav: "pcx:l:abc123:2:p",
    }
    for cls, cid in cases.items():
        assert cls.__discord_ui_compiled_template__.fullmatch(cid), (cls.__name__, cid)
    # ResultNav must not swallow the rj/rx/l prefixes.
    nav = ui.ResultNav.__discord_ui_compiled_template__
    assert not nav.fullmatch("pcx:rj:abc123")
    assert not nav.fullmatch("pcx:rx:abc123:1")
    assert not nav.fullmatch("pcx:l:abc123:1:p")
