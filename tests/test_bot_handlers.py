"""The slash-command handler bodies, driven over a real tmp index.

Each test runs one ``_run_*``/``_handle_*`` method on a real ``PodCodexBot``
(no Discord login) against a seeded LanceDB, and asserts what the user reads
in the reply. Only ``/search`` keeps a stub, for ``hybrid_search``: it would
otherwise load an embedding model. The service functions themselves are
covered in ``tests/test_search_service.py``.
"""

import asyncio

import pytest

pytest.importorskip("discord")

from podcodex.bot import resolution as resolution_module  # noqa: E402
from podcodex.bot import search_commands as search_module  # noqa: E402
from podcodex.bot.bot import ResolvedShows, ServerSettings, ShowAccess  # noqa: E402
from podcodex.rag import index_store as rag_index_store  # noqa: E402
from tests.fixtures.bot import make_bot  # noqa: E402
from tests.fixtures.index import add_show  # noqa: E402

ALPHA = "alpha_show__bge-m3__semantic"
BETA = "beta_show__bge-m3__semantic"


class _FakeFollowup:
    def __init__(self):
        self.messages: list[dict] = []

    async def send(
        self, content=None, *, embed=None, embeds=None, view=None, ephemeral=False
    ):
        self.messages.append(
            {"content": content, "embed": embed, "embeds": embeds, "view": view}
        )


class _FakeResponse:
    async def defer(self):
        pass


class _FakeInteraction:
    def __init__(self, guild_id=1):
        self.guild_id = guild_id
        self.response = _FakeResponse()
        self.followup = _FakeFollowup()


def _reply_text(interaction) -> str:
    """Everything the first reply shows: content and every embed's text."""
    msg = interaction.followup.messages[0]
    parts = [msg["content"] or ""]
    for embed in [msg["embed"], *(msg["embeds"] or [])]:
        if embed is None:
            continue
        d = embed.to_dict()
        parts += [d.get("title", ""), d.get("description", "")]
        parts += [f"{f['name']} {f['value']}" for f in d.get("fields", [])]
        parts.append(d.get("footer", {}).get("text", ""))
    return "\n".join(parts)


@pytest.fixture
def bot(tmp_path, isolated_index, monkeypatch):
    monkeypatch.setattr(resolution_module, "load_show_rag_prefs", lambda: {})
    store = rag_index_store.get_index_store()
    add_show(
        store,
        "Alpha Show",
        {"ep1": ["I met William yesterday", "John Williams composed, Williams again"]},
    )
    add_show(store, "Beta Show")
    return make_bot(tmp_path)[0]


def test_search_replies_with_the_hit(bot, monkeypatch):
    store = rag_index_store.get_index_store()
    (hit, *_rest) = store.load_chunks_no_embeddings(ALPHA, "ep1")
    seen: list[set[str]] = []

    def fake_hybrid(query, cols, **_kw):
        seen.append({c.name for c in cols})
        return [(hit, ALPHA)]

    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid)
    interaction = _FakeInteraction()

    asyncio.run(
        bot._run_search(
            interaction,
            "william",
            ResolvedShows(ShowAccess.ALL),
            ServerSettings(),
            0.5,
            "α=0.50",
        )
    )

    assert seen == [{ALPHA, BETA}]  # every show, once each
    assert "I met William yesterday" in _reply_text(interaction)


def test_search_explains_why_nothing_matched(bot, monkeypatch):
    """/search must give the same precise reason as /exact and /random when a
    scope resolves to no collections (locked show, wrong model), not the
    generic "no results, try simpler wording"."""
    ran: list[bool] = []
    monkeypatch.setattr(
        search_module, "hybrid_search", lambda *a, **k: ran.append(True) or []
    )
    interaction = _FakeInteraction()
    shows = ResolvedShows(ShowAccess.SPECIFIC, ("Nonexistent Show",))

    asyncio.run(
        bot._run_search(interaction, "hello", shows, ServerSettings(), 0.5, "α=0.50")
    )

    assert ran == []
    msg = interaction.followup.messages[0]["content"]
    assert "simpler wording" not in msg
    assert "Nonexistent Show" in msg


def test_exact_splits_word_and_partial_matches(bot):
    interaction = _FakeInteraction()

    asyncio.run(bot._run_exact(interaction, "william", ResolvedShows(ShowAccess.ALL)))

    text = _reply_text(interaction)
    # 1 standalone "William", 2 occurrences inside "Williams".
    assert "1 exact · 2 partial" in text
    assert "I met **__William__** yesterday" in text  # the match is highlighted


def test_stats_speakers_and_episodes_describe_the_show(bot):
    for handler in (bot._handle_stats, bot._handle_speakers, bot._handle_episodes):
        interaction = _FakeInteraction()
        asyncio.run(handler(interaction, "Alpha Show", None))
        text = _reply_text(interaction)
        assert "Alpha Show" in text, handler.__name__
        assert "Beta Show" not in text, handler.__name__
