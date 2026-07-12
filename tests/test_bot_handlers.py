"""Wiring smoke tests for the six rewired slash-command handler bodies.

Each test drives one ``_run_*``/``_handle_*`` method directly against a
``PodCodexBot`` built with ``__new__`` (no Discord login) and fakes for the
Discord interaction plus the shared search-service seam
(``podcodex.bot.bot.hybrid_search`` / ``exact_search`` / ``random_quote`` /
``load_show_rag_prefs``). The point is not to re-test the service functions
(covered in ``tests/test_search_service.py``) but to pin that each handler
calls the right seam with the right arguments and always lands a followup.
No real LanceDB is touched; ``PodCodexBot.local`` is a small fake.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("discord")

from podcodex.bot import bot as bot_module  # noqa: E402
from podcodex.bot.bot import (  # noqa: E402
    BotConfig,
    PodCodexBot,
    ResolvedShows,
    SearchCollection,
    ServerSettings,
    ShowAccess,
    _AutocompleteCache,
)
from podcodex.bot.result_store import SearchCacheStore  # noqa: E402

COL_INFO = {
    "alpha__bge-m3__semantic": {
        "show": "Alpha Show",
        "model": "bge-m3",
        "chunker": "semantic",
        "artwork_url": "",
    },
    "beta__bge-m3__semantic": {
        "show": "Beta Show",
        "model": "bge-m3",
        "chunker": "semantic",
        "artwork_url": "",
    },
}


def _chunk(episode="ep1", idx=0, text="hello world", show="Alpha Show", start=1.0):
    return {
        "episode": episode,
        "chunk_index": idx,
        "text": text,
        "score": 0.8,
        "show": show,
        "start": start,
        "end": start + 5.0,
        "dominant_speaker": "Alice",
        "pub_date": "2024-01-01",
        "source": "transcript",
        "timed": True,
    }


class _FakeLocal:
    """Stand-in for IndexStore: pre-canned answers, no LanceDB."""

    def __init__(self, episode_chunks=None, episode_stats=None, speaker_rows=None):
        self._episode_chunks = episode_chunks or {}
        self._episode_stats = episode_stats or {}
        self._speaker_rows = speaker_rows or []
        self.get_episode_stats_calls: list[str] = []
        self.speaker_stats_multi_calls: list[list[str]] = []

    def get_all_collection_info(self):
        return dict(COL_INFO)

    def load_chunks_no_embeddings(self, collection, episode):
        return [dict(c) for c in self._episode_chunks.get((collection, episode), [])]

    def get_episode_stats(self, collection):
        self.get_episode_stats_calls.append(collection)
        return self._episode_stats.get(collection, [])

    def speaker_stats_multi(self, collections):
        self.speaker_stats_multi_calls.append(list(collections))
        return self._speaker_rows

    def index_mtime(self):
        return 0.0

    def reconnect(self):
        pass


class _Recorder:
    """Callable that records every call and returns a fixed value."""

    def __init__(self, retval):
        self.retval = retval
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.retval


class _FakeFollowup:
    def __init__(self):
        self.messages: list[dict] = []

    async def send(
        self, content=None, *, embed=None, embeds=None, view=None, ephemeral=False
    ):
        self.messages.append(
            {
                "content": content,
                "embed": embed,
                "embeds": embeds,
                "view": view,
                "ephemeral": ephemeral,
            }
        )


class _FakeResponse:
    async def defer(self):
        pass


class _FakeInteraction:
    def __init__(self, guild_id=1):
        self.guild_id = guild_id
        self.response = _FakeResponse()
        self.followup = _FakeFollowup()


def _make_bot(local):
    """A PodCodexBot with no Discord login, wired to a fake IndexStore."""
    bot = PodCodexBot.__new__(PodCodexBot)
    bot._shows = {}
    bot._server_cfg = {}
    bot.config = BotConfig()
    bot._local = local
    bot._ac_cache = _AutocompleteCache(
        episodes={}, episode_titles={}, sources={}, speakers={}, col_info=dict(COL_INFO)
    )
    bot.results = SearchCacheStore(Path(tempfile.mkdtemp()) / "search_cache.db")

    async def _noop_refresh():
        return None

    bot._refresh_if_stale = _noop_refresh
    return bot


# ──────────────────────────────────────────────
# /search
# ──────────────────────────────────────────────


def test_run_search_wires_hybrid_search_and_sends_followup(monkeypatch):
    chunk = _chunk()
    fake_hybrid = _Recorder([(chunk, "alpha__bge-m3__semantic")])
    monkeypatch.setattr(bot_module, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(bot_module, "load_show_rag_prefs", lambda: {})

    local = _FakeLocal(episode_chunks={("alpha__bge-m3__semantic", "ep1"): [chunk]})
    bot = _make_bot(local)
    interaction = _FakeInteraction()
    settings = ServerSettings()
    shows = ResolvedShows(ShowAccess.ALL)

    asyncio.run(bot._run_search(interaction, "hello", shows, settings, 0.5, "α=0.50"))

    assert len(fake_hybrid.calls) == 1
    args, kwargs = fake_hybrid.calls[0]
    assert args[0] == "hello"
    cols = args[1]
    assert all(isinstance(c, SearchCollection) for c in cols)
    assert {c.name for c in cols} == set(COL_INFO)
    assert kwargs["strategy"] == bot.config.merge_strategy
    assert kwargs["score_floor"] == 0.05
    assert kwargs["top_k"] == settings.top_k

    assert interaction.followup.messages
    assert interaction.followup.messages[0]["embed"] is not None


# ──────────────────────────────────────────────
# /exact
# ──────────────────────────────────────────────


def test_run_exact_wires_exact_search_chronological_and_sends_followup(monkeypatch):
    chunk = _chunk(text="hello world")
    fake_exact = _Recorder([(chunk, "alpha__bge-m3__semantic")])
    monkeypatch.setattr(bot_module, "exact_search", fake_exact)
    monkeypatch.setattr(bot_module, "load_show_rag_prefs", lambda: {})

    local = _FakeLocal(episode_chunks={("alpha__bge-m3__semantic", "ep1"): [chunk]})
    bot = _make_bot(local)
    interaction = _FakeInteraction()
    shows = ResolvedShows(ShowAccess.ALL)

    asyncio.run(bot._run_exact(interaction, "hello world", shows))

    assert len(fake_exact.calls) == 1
    args, kwargs = fake_exact.calls[0]
    assert args[0] == "hello world"
    cols = args[1]
    assert all(isinstance(c, SearchCollection) for c in cols)
    assert {c.name for c in cols} == set(COL_INFO)
    assert kwargs["order"] == "chronological"

    assert interaction.followup.messages
    assert interaction.followup.messages[0]["embed"] is not None


# ──────────────────────────────────────────────
# /random
# ──────────────────────────────────────────────


def test_run_random_wires_random_quote_and_sends_followup(monkeypatch):
    chunk = _chunk()
    fake_random = _Recorder((chunk, "alpha__bge-m3__semantic"))
    monkeypatch.setattr(bot_module, "random_quote", fake_random)
    monkeypatch.setattr(bot_module, "load_show_rag_prefs", lambda: {})

    local = _FakeLocal()
    bot = _make_bot(local)
    interaction = _FakeInteraction()
    shows = ResolvedShows(ShowAccess.ALL)

    asyncio.run(bot._run_random(interaction, shows))

    assert len(fake_random.calls) == 1
    args, _kwargs = fake_random.calls[0]
    cols = args[0]
    assert all(isinstance(c, SearchCollection) for c in cols)
    assert {c.name for c in cols} == set(COL_INFO)

    assert interaction.followup.messages
    assert interaction.followup.messages[0]["embed"] is not None


# ──────────────────────────────────────────────
# /stats
# ──────────────────────────────────────────────


def test_handle_stats_single_show_calls_speaker_stats_multi(monkeypatch):
    monkeypatch.setattr(bot_module, "load_show_rag_prefs", lambda: {})
    local = _FakeLocal(
        episode_stats={
            "alpha__bge-m3__semantic": [
                {"episode": "ep1", "duration": 10.0, "pub_date": "2024-01-01"}
            ]
        },
        speaker_rows=[
            {
                "speaker": "Alice",
                "total_duration": 10.0,
                "chunk_count": 3,
                "episodes": 1,
            }
        ],
    )
    bot = _make_bot(local)
    interaction = _FakeInteraction()

    asyncio.run(bot._handle_stats(interaction, "Alpha Show", None))

    assert local.speaker_stats_multi_calls == [["alpha__bge-m3__semantic"]]
    assert interaction.followup.messages
    assert interaction.followup.messages[0]["embed"] is not None


# ──────────────────────────────────────────────
# /speakers
# ──────────────────────────────────────────────


def test_handle_speakers_calls_speaker_stats_multi_and_sends_embed(monkeypatch):
    monkeypatch.setattr(bot_module, "load_show_rag_prefs", lambda: {})
    local = _FakeLocal(
        speaker_rows=[
            {
                "speaker": "Alice",
                "total_duration": 10.0,
                "chunk_count": 3,
                "episodes": 1,
            },
            {"speaker": "Bob", "total_duration": 5.0, "chunk_count": 1, "episodes": 1},
        ]
    )
    bot = _make_bot(local)
    interaction = _FakeInteraction()

    asyncio.run(bot._handle_speakers(interaction, "Alpha Show", None))

    assert local.speaker_stats_multi_calls == [["alpha__bge-m3__semantic"]]
    assert interaction.followup.messages
    assert interaction.followup.messages[0]["embed"] is not None


# ──────────────────────────────────────────────
# /episodes
# ──────────────────────────────────────────────


def test_handle_episodes_calls_get_episode_stats_and_sends_embed(monkeypatch):
    monkeypatch.setattr(bot_module, "load_show_rag_prefs", lambda: {})
    local = _FakeLocal(
        episode_stats={
            "alpha__bge-m3__semantic": [
                {"episode": "ep1", "duration": 10.0, "pub_date": "2024-01-01"}
            ]
        }
    )
    bot = _make_bot(local)
    interaction = _FakeInteraction()

    asyncio.run(bot._handle_episodes(interaction, "Alpha Show", None))

    assert local.get_episode_stats_calls == ["alpha__bge-m3__semantic"]
    assert interaction.followup.messages
    assert interaction.followup.messages[0]["embed"] is not None
