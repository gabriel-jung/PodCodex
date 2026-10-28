"""Bot autocomplete: the locked-show filter, and what each picker offers.

`_visible_collections` is the gate that keeps password-protected shows out of
Discord's autocomplete. A regression there leaks locked show, episode and
speaker names to every guild while the suite stays green, so it is pinned
here alongside the pickers built on top of it.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")

from podcodex.core.show_passwords import hash_show_password  # noqa: E402
from podcodex.rag import index_store as rag_index_store  # noqa: E402
from tests.fixtures.bot import make_bot  # noqa: E402
from tests.fixtures.index import add_show, chunk  # noqa: E402

DIM = 8


@pytest.fixture
def store(isolated_index):
    """Two shows indexed under the default combo; "Locked" has a password."""
    st = rag_index_store.get_index_store()
    for show, show_id in (("Public", "public_1111aaaa"), ("Locked", "locked_2222bbbb")):
        stem = f"{show.lower()}-ep1"
        add_show(
            st,
            show,
            {stem: [chunk(episode=stem, show=show, speaker=f"{show} Speaker")]},
            show_id=show_id,
        )
    st.set_show_password(
        "locked_2222bbbb", hash_show_password("x" * 16), show_label="Locked"
    )
    return st


def _bot(store):
    """The real bot over the seeded store (they share the isolated index),
    loading both models the tests index under."""
    return make_bot(store.path.parent, models=("bge-m3", "e5-small"))[0]


class _Namespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, _name):
        return ""


class _Interaction:
    def __init__(self, guild_id=1, **ns):
        self.guild_id = guild_id
        self.namespace = _Namespace(**ns)
        self.command = None


def _settings(**kw):
    from podcodex.bot.config import ServerSettings

    return ServerSettings(**kw)


def test_a_locked_show_is_hidden_from_a_guild_that_has_not_unlocked_it(store):
    bot = _bot(store)
    cols, _info = asyncio.run(bot._visible_collections(_settings()))
    assert cols == ["public__bge-m3__semantic"]


def test_unlocking_reveals_the_show_for_that_guild_only(store):
    bot = _bot(store)
    unlocked = _settings(unlocked_shows=["locked_2222bbbb"])

    cols, _info = asyncio.run(bot._visible_collections(unlocked))
    assert sorted(cols) == ["locked__bge-m3__semantic", "public__bge-m3__semantic"]

    # A different guild still sees only the public one.
    bot._ac_cache.reset()
    other, _info = asyncio.run(bot._visible_collections(_settings()))
    assert other == ["public__bge-m3__semantic"]


def test_a_public_show_stays_visible_when_nothing_is_protected(store):
    bot = _bot(store)
    store.delete_show_password("locked_2222bbbb")
    asyncio.run(bot._reload_access())
    bot._ac_cache.reset()

    cols, _info = asyncio.run(bot._visible_collections(_settings()))
    assert sorted(cols) == ["locked__bge-m3__semantic", "public__bge-m3__semantic"]


def test_search_advanced_pickers_stay_within_the_models_the_bot_loads(store):
    """/search-advanced encodes the query, so its show picker must not offer
    a show the command will refuse; other commands read without a model."""
    add_show(store, "Other", show_id="other_3333cccc", model="e5-small")
    bot = make_bot(store.path.parent)[0]  # loads only bge-m3

    class _Cmd:
        def __init__(self, name):
            self.name = name

    search = _Interaction()
    search.command = _Cmd("search-advanced")
    stats = _Interaction()
    stats.command = _Cmd("stats")

    assert "Other" not in [
        c.value for c in asyncio.run(bot._show_autocomplete(search, ""))
    ]
    assert "Other" in [c.value for c in asyncio.run(bot._show_autocomplete(stats, ""))]
    assert [c.value for c in asyncio.run(bot._model_autocomplete(search, ""))] == [
        "bge-m3"
    ]
    assert "e5-small" in [
        c.value for c in asyncio.run(bot._model_autocomplete(stats, ""))
    ]


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


def test_pickers_offer_a_show_indexed_only_under_another_model(store):
    """Suggestions come from the collections the command will search, which
    fall back past the server's default combo; an exact-combo match offered
    nothing for such a show while /search found its hits."""
    add_show(
        store,
        "Other",
        {"other-ep1": [chunk(episode="other-ep1", show="Other", speaker="Olive")]},
        show_id="other_3333cccc",
        model="e5-small",
    )
    bot = _bot(store)
    interaction = _Interaction(show="Other")

    episodes = asyncio.run(bot._episode_autocomplete(interaction, ""))
    speakers = asyncio.run(bot._speaker_autocomplete(interaction, ""))

    assert [c.value for c in episodes] == ["other-ep1"]
    assert [c.value for c in speakers] == ["Olive"]
