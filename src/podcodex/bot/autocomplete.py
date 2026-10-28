"""Autocomplete handlers and their TTL cache."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import discord
from discord import app_commands

from podcodex.bot.config import ANY_SOURCE, ServerSettings
from podcodex.bot.resolution import explicit_combo
from podcodex.core._utils import resolve_episode_title
from podcodex.rag.defaults import (
    CHUNKING_STRATEGIES,
    MODELS,
)

# What a Discord user reads in the chunker picker. CHUNKING_STRATEGIES' own
# text is written for the desktop app (library names, install notes).
_CHUNKER_NAMES = {
    "semantic": "Semantic: passages grouped by topic",
    "speaker": "Speaker turns: one passage per turn",
}

# ── Autocomplete cache ───────────────────────


@dataclass
class _AutocompleteCache:
    episodes: dict[str, list[str]] = field(default_factory=dict)  # col -> stems
    episode_titles: dict[str, dict[str, str]] = field(
        default_factory=dict
    )  # col -> {stem: rss_title}
    sources: dict[str, list[str]] = field(default_factory=dict)  # col -> source values
    speakers: dict[str, list[str]] = field(default_factory=dict)  # col -> speaker names
    col_info: dict[str, dict] | None = None  # {collection: {show, model, chunker, dim}}
    timestamp: float = 0.0
    ttl: float = 300.0  # 5 minutes

    def is_stale(self) -> bool:
        # timestamp=0.0 is the "never populated" sentinel. On Linux fresh
        # containers monotonic() can be small enough that delta-from-0 < ttl,
        # so the sentinel must be checked explicitly.
        if self.timestamp == 0.0:
            return True
        return (time.monotonic() - self.timestamp) > self.ttl

    def reset(self) -> None:
        self.episodes.clear()
        self.episode_titles.clear()
        self.sources.clear()
        self.speakers.clear()
        self.col_info = None
        self.timestamp = time.monotonic()


def _filter_choices(
    choices: list[app_commands.Choice[str]], current: str
) -> list[app_commands.Choice[str]]:
    """Choices whose label or value contains what was typed, up to Discord's 25."""
    typed = current.lower()
    return [c for c in choices if typed in c.name.lower() or typed in c.value.lower()][
        :25
    ]


def _embeds_query(interaction: discord.Interaction) -> bool:
    """Whether the command being typed encodes a query (so the bot's model
    list applies). Only /search-advanced among the commands with pickers."""
    return getattr(interaction.command, "name", "") == "search-advanced"


class AutocompleteMixin:
    """Autocomplete methods mixed into PodCodexBot (bot.py).

    Expects on self: ``_ac_cache``, ``local``, ``_server_settings``,
    ``_filter_collections``, ``_resolve_shows``, ``_refresh_if_stale``.
    """

    async def _show_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        """Every show the command being typed can reach.

        Resolution falls back across models, so this is not filtered on the
        typed or default model; only /search-advanced, which encodes the
        query, is limited to the models the bot loads. ``/setup show_add``
        gets every show, since a pin searches nothing.
        """
        await self._refresh_if_stale()
        self._cache_clear_if_stale()
        settings = self._server_settings(interaction.guild_id)
        shows = await self._known_show_labels(
            settings, loadable_only=_embeds_query(interaction)
        )
        return _filter_choices(
            [app_commands.Choice(name=s, value=s) for s in shows], current
        )

    def _cache_clear_if_stale(self) -> None:
        if self._ac_cache.is_stale():
            self._ac_cache.reset()

    async def _cached_col_info(self) -> dict[str, dict]:
        """Return ``{collection: info}`` map, cached with the autocomplete TTL."""
        cache = self._ac_cache
        if cache.col_info is None:
            loop = asyncio.get_running_loop()
            cache.col_info = await loop.run_in_executor(
                None, self.local.get_all_collection_info
            )
        return cache.col_info or {}

    async def _visible_collections(
        self, settings: ServerSettings, *, loadable_only: bool = False
    ) -> tuple[list[str], dict[str, dict]]:
        """Collections this server may see (under a model the bot loads, with
        ``loadable_only``)."""
        col_info = await self._cached_col_info()
        pool = self._loadable(col_info) if loadable_only else col_info
        return self._filter_collections(list(pool), settings, col_info), col_info

    async def _command_collections(
        self, interaction: discord.Interaction, settings: ServerSettings
    ) -> list[str] | None:
        """Collections the command being typed will search, or None if locked.

        Picked exactly as the command will pick them (``_resolve_show_collections``:
        explicit model/chunker, show.toml prefs, server default, first by name),
        so a show indexed under another model still gets suggestions.
        """
        ns = interaction.namespace
        explicit = explicit_combo(
            getattr(ns, "model", "") or "", getattr(ns, "chunker", "") or "", settings
        )
        show = getattr(ns, "show", "") or ""
        col_info = await self._cached_col_info()

        def _pick() -> list[str] | None:
            resolved = self._resolve_shows(settings, show)
            if resolved.is_locked:
                return None
            return [
                c.name
                for c in self._resolve_show_collections(
                    resolved,
                    settings,
                    col_info,
                    explicit=explicit,
                    embeds=_embeds_query(interaction),
                )
            ]

        # Index reads (label lookups, access checks): off the event loop, as
        # every other autocomplete read is, since this runs per keystroke.
        return await asyncio.get_running_loop().run_in_executor(None, _pick)

    async def _known_show_labels(
        self, settings: ServerSettings, *, loadable_only: bool = False
    ) -> list[str]:
        """Sorted display names of the shows this server may see.

        The one definition of "which shows exist for this guild": the show
        autocompletes offer these, and ``/setup show_add`` validates against
        them, so a name the picker shows is never rejected as unknown.
        """
        collections, col_info = await self._visible_collections(
            settings, loadable_only=loadable_only
        )
        return sorted(
            {
                col_info[col]["show"]
                for col in collections
                if col_info.get(col, {}).get("show")
            }
        )

    async def _cached_episodes(self, collection: str) -> list[str]:
        """Return episode stems, using the TTL cache."""
        cache = self._ac_cache
        if collection not in cache.episodes:
            loop = asyncio.get_running_loop()
            eps = await loop.run_in_executor(
                None, lambda: self.local.list_episodes(collection)
            )
            cache.episodes[collection] = eps
        return cache.episodes.get(collection, [])

    async def _cached_episode_titles(self, collection: str) -> dict[str, str]:
        """Return {stem: rss_title} for episodes that have one, using the TTL cache."""
        cache = self._ac_cache
        if collection not in cache.episode_titles:
            loop = asyncio.get_running_loop()
            titles = await loop.run_in_executor(
                None, lambda: self.local.list_episode_titles(collection)
            )
            cache.episode_titles[collection] = titles
        return cache.episode_titles.get(collection, {})

    async def _cached_sources(self, collection: str) -> list[str]:
        """Return source values, using the TTL cache."""
        cache = self._ac_cache
        if collection not in cache.sources:
            loop = asyncio.get_running_loop()
            srcs = await loop.run_in_executor(
                None, lambda: self.local.list_sources(collection)
            )
            cache.sources[collection] = srcs
        return cache.sources.get(collection, [])

    async def _episode_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        await self._refresh_if_stale()
        self._cache_clear_if_stale()
        settings = self._server_settings(interaction.guild_id)
        cols = await self._command_collections(interaction, settings)
        if cols is None:
            return []

        all_episodes: dict[str, str] = {}  # stem -> display name
        for col in cols:
            stems = await self._cached_episodes(col)
            titles = await self._cached_episode_titles(col)
            for stem in stems:
                all_episodes[stem] = resolve_episode_title(titles.get(stem) or "", stem)

        ordered = sorted(all_episodes.items(), key=lambda x: x[1].lower())
        return _filter_choices(
            [app_commands.Choice(name=d, value=stem) for stem, d in ordered], current
        )

    async def _source_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        await self._refresh_if_stale()
        self._cache_clear_if_stale()
        settings = self._server_settings(interaction.guild_id)
        cols = await self._command_collections(interaction, settings)
        if cols is None:
            return []

        all_sources: set[str] = set()
        for col in cols:
            all_sources.update(await self._cached_sources(col))

        # First, so a server default_source can be overridden (on -advanced)
        # or cleared (on /setup).
        choices = [app_commands.Choice(name="Any source", value=ANY_SOURCE)]
        choices += [app_commands.Choice(name=s, value=s) for s in sorted(all_sources)]
        return _filter_choices(choices, current)

    async def _model_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        """Models present in the index, by name: only those the bot loads
        where they encode a query (/search-advanced) or become one (/setup)."""
        self._cache_clear_if_stale()
        col_info = await self._cached_col_info()
        name = getattr(interaction.command, "name", "")
        if name in ("search-advanced", "setup"):
            col_info = self._loadable(col_info)
        present = sorted(
            {info["model"] for info in col_info.values() if info["model"] in MODELS}
        )
        default = self._server_settings(interaction.guild_id).model
        choices = [
            app_commands.Choice(
                name=MODELS[m].label
                + (" (this server's default)" if m == default else ""),
                value=m,
            )
            for m in present
        ]
        return _filter_choices(choices, current)

    async def _chunker_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        """Only offer chunkers actually present in the index."""
        self._cache_clear_if_stale()
        col_info = await self._cached_col_info()
        present = sorted(
            {
                info["chunker"]
                for info in col_info.values()
                if info["chunker"] in CHUNKING_STRATEGIES
            }
        )
        choices = [
            app_commands.Choice(name=_CHUNKER_NAMES.get(c, c), value=c) for c in present
        ]
        return _filter_choices(choices, current)

    async def _cached_speakers(self, collection: str) -> list[str]:
        """Return speaker names, using the TTL cache."""
        cache = self._ac_cache
        if collection not in cache.speakers:
            loop = asyncio.get_running_loop()
            spks = await loop.run_in_executor(
                None, lambda: self.local.list_speakers(collection)
            )
            cache.speakers[collection] = spks
        return cache.speakers.get(collection, [])

    async def _speaker_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        await self._refresh_if_stale()
        self._cache_clear_if_stale()
        settings = self._server_settings(interaction.guild_id)
        cols = await self._command_collections(interaction, settings)
        if cols is None:
            return []

        all_speakers: set[str] = set()
        for col in cols:
            all_speakers.update(await self._cached_speakers(col))

        return _filter_choices(
            [app_commands.Choice(name=s, value=s) for s in sorted(all_speakers)],
            current,
        )

    async def _pinned_show_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        """Autocomplete from the shows this server has pinned as defaults."""
        settings = self._server_settings(interaction.guild_id)
        # Stored as ids; both the label and the value users see are names.
        labels = [self._label_for_show_id(s) for s in self._pinned_ids(settings)]
        return _filter_choices(
            [app_commands.Choice(name=s, value=s) for s in labels], current
        )

    async def _unlocked_show_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        """Autocomplete from the protected shows unlocked on this server.

        What ``/lock`` and ``/changepassword`` act on. Offering the pinned
        list here was the same conflation one layer out: it suggested public
        shows to a command that can only revoke an unlock.
        """
        settings = self._server_settings(interaction.guild_id)
        labels = [self._label_for_show_id(s) for s in self._unlocked_ids(settings)]
        return _filter_choices(
            [app_commands.Choice(name=s, value=s) for s in labels], current
        )
