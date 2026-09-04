"""Bot autocomplete: the locked-show filter, and what each picker offers.

`_visible_collections` is the gate that keeps password-protected shows out of
Discord's autocomplete. A regression there leaks locked show, episode and
speaker names to every guild while the suite stays green, so it is pinned
here alongside the pickers built on top of it.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

pytest.importorskip("fastapi")

from podcodex.core.show_passwords import hash_show_password  # noqa: E402
from podcodex.rag import index_store as rag_index_store  # noqa: E402

DIM = 8


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Two shows indexed under the default combo; "Locked" has a password."""
    monkeypatch.setenv("PODCODEX_INDEX", str(tmp_path / "index"))
    rag_index_store.get_index_store.cache_clear()
    st = rag_index_store.get_index_store()
    for show, show_id in (("Public", "public_1111aaaa"), ("Locked", "locked_2222bbbb")):
        col = f"{show.lower()}__bge-m3__semantic"
        st.ensure_collection(
            col, show=show, model="bge-m3", chunker="semantic", dim=DIM
        )
        st.set_collection_identity(col, show_id=show_id, show=show)
        st.save_chunks(
            col,
            f"{show.lower()}-ep1",
            [
                {
                    "text": "hello",
                    "episode": f"{show.lower()}-ep1",
                    "show": show,
                    "source": "transcript",
                    "dominant_speaker": f"{show} Speaker",
                    "start": 0.0,
                    "end": 1.0,
                }
            ],
            np.zeros((1, DIM), dtype=np.float32),
        )
    st.set_show_password(
        "locked_2222bbbb", hash_show_password("x" * 16), show_label="Locked"
    )
    yield st
    rag_index_store.get_index_store.cache_clear()


def _bot(store):
    from podcodex.bot.bot import BotConfig, PodCodexBot
    from podcodex.bot.autocomplete import _AutocompleteCache

    bot = PodCodexBot.__new__(PodCodexBot)
    bot.config = BotConfig()
    bot._local = store
    bot._server_cfg = {}
    bot._shows = {}
    bot._ac_cache = _AutocompleteCache()
    bot._save_server_config = lambda: None
    bot._reload_shows()  # populates _shows from the password table

    async def _noop():
        return None

    bot._refresh_if_stale = _noop
    return bot


class _Namespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, _name):
        return ""


class _Interaction:
    def __init__(self, guild_id=1, **ns):
        self.guild_id = guild_id
        self.namespace = _Namespace(**ns)


def _settings(**kw):
    from podcodex.bot.config import ServerSettings

    return ServerSettings(**kw)


def test_a_locked_show_is_hidden_from_a_guild_that_has_not_unlocked_it(store):
    bot = _bot(store)
    cols, _info = asyncio.run(
        bot._visible_collections(_settings(), "bge-m3", "semantic")
    )
    assert cols == ["public__bge-m3__semantic"]


def test_unlocking_reveals_the_show_for_that_guild_only(store):
    bot = _bot(store)
    unlocked = _settings(unlocked_shows=["locked_2222bbbb"])

    cols, _info = asyncio.run(bot._visible_collections(unlocked, "bge-m3", "semantic"))
    assert sorted(cols) == ["locked__bge-m3__semantic", "public__bge-m3__semantic"]

    # A different guild still sees only the public one.
    bot._ac_cache.reset()
    other, _info = asyncio.run(
        bot._visible_collections(_settings(), "bge-m3", "semantic")
    )
    assert other == ["public__bge-m3__semantic"]


def test_a_public_show_stays_visible_when_nothing_is_protected(store):
    bot = _bot(store)
    store.delete_show_password("locked_2222bbbb")
    bot._reload_shows()
    bot._ac_cache.reset()

    cols, _info = asyncio.run(
        bot._visible_collections(_settings(), "bge-m3", "semantic")
    )
    assert sorted(cols) == ["locked__bge-m3__semantic", "public__bge-m3__semantic"]


def test_visible_collections_filters_by_model_and_chunker(store):
    bot = _bot(store)
    cols, _info = asyncio.run(bot._visible_collections(_settings(), "e5-large", ""))
    assert cols == []


def test_known_show_labels_ignores_the_model_when_unfiltered(store):
    """`/setup show_add` pins a show whatever it is indexed under, so its
    validation set must not be narrowed by the guild's search model."""
    bot = _bot(store)
    assert asyncio.run(bot._known_show_labels(_settings())) == ["Public"]


def test_show_autocomplete_never_offers_a_locked_show(store):
    bot = _bot(store)
    choices = asyncio.run(bot._show_autocomplete(_Interaction(), ""))
    assert [c.value for c in choices] == ["Public"]


def test_show_autocomplete_filters_on_what_the_user_typed(store):
    bot = _bot(store)
    assert asyncio.run(bot._show_autocomplete(_Interaction(), "pub"))
    assert asyncio.run(bot._show_autocomplete(_Interaction(), "zz")) == []


def test_pinned_and_unlocked_pickers_read_their_own_list(store):
    """The two lists drive different commands: `/setup show_remove` unpins,
    `/lock` revokes. Offering one list to both is the conflation that made
    `/lock` look like it could act on a public show."""
    bot = _bot(store)
    bot._server_cfg[1] = _settings(
        pinned_shows=["public_1111aaaa"], unlocked_shows=["locked_2222bbbb"]
    )

    pinned = asyncio.run(bot._pinned_show_autocomplete(_Interaction(), ""))
    unlocked = asyncio.run(bot._unlocked_show_autocomplete(_Interaction(), ""))

    assert [c.value for c in pinned] == ["Public"]
    assert [c.value for c in unlocked] == ["Locked"]


def test_episode_autocomplete_hides_a_locked_show_s_episodes(store):
    bot = _bot(store)
    stems = [
        c.value for c in asyncio.run(bot._episode_autocomplete(_Interaction(), ""))
    ]
    assert "public-ep1" in stems
    assert "locked-ep1" not in stems


def test_episode_autocomplete_returns_nothing_for_an_explicit_locked_show(store):
    bot = _bot(store)
    interaction = _Interaction(show="Locked")
    assert asyncio.run(bot._episode_autocomplete(interaction, "")) == []
