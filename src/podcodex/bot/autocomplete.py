"""Autocomplete handlers and their TTL cache."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import discord
from discord import app_commands

from podcodex.bot.config import ServerSettings
from podcodex.core._utils import resolve_episode_title
from podcodex.rag.defaults import (
    CHUNKING_STRATEGIES,
    MODELS,
)

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
        await self._refresh_if_stale()
        self._cache_clear_if_stale()
        settings = self._server_settings(interaction.guild_id)
        model = getattr(interaction.namespace, "model", "") or settings.model
        chunker = settings.chunker

        collections, col_info = await self._visible_collections(
            settings, model, chunker
        )
        shows = sorted(
            {
                col_info[col]["show"]
                for col in collections
                if col_info.get(col, {}).get("show")
            }
        )
        return [
            app_commands.Choice(name=s, value=s)
            for s in shows
            if current.lower() in s.lower()
        ][:25]

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
        self, settings: ServerSettings, model: str, chunker: str
    ) -> tuple[list[str], dict[str, dict]]:
        """Return collections matching (model, chunker) that the server is allowed to see."""
        col_info = await self._cached_col_info()
        cols = [
            name
            for name, info in col_info.items()
            if (not model or info["model"] == model)
            and (not chunker or info["chunker"] == chunker)
        ]
        return self._filter_collections(cols, settings, col_info), col_info

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
        model = getattr(interaction.namespace, "model", "") or settings.model
        chunker = settings.chunker

        # Determine which show(s) to look at
        show = getattr(interaction.namespace, "show", "")
        resolved = self._resolve_shows(settings, show)
        if resolved.is_locked:
            return []

        all_episodes: dict[str, str] = {}  # stem -> display name
        if resolved.is_specific:
            # Resolved through the store, never rebuilt: the bot has no
            # show.toml, so a display name is all it has, and the collection
            # it maps to is the store's business.
            cols = [
                col
                for col in (
                    self.local.resolve_collection("", model, chunker, show_label=s)
                    for s in resolved.shows
                )
                if col
            ]
        else:
            cols, _ = await self._visible_collections(settings, model, chunker)
        for col in cols:
            stems = await self._cached_episodes(col)
            titles = await self._cached_episode_titles(col)
            for stem in stems:
                all_episodes[stem] = resolve_episode_title(titles.get(stem) or "", stem)

        return [
            app_commands.Choice(name=display, value=stem)
            for stem, display in sorted(
                all_episodes.items(), key=lambda x: x[1].lower()
            )
            if current.lower() in display.lower() or current.lower() in stem.lower()
        ][:25]

    async def _source_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        await self._refresh_if_stale()
        self._cache_clear_if_stale()
        settings = self._server_settings(interaction.guild_id)
        model = getattr(interaction.namespace, "model", "") or settings.model
        chunker = settings.chunker

        collections, _ = await self._visible_collections(settings, model, chunker)

        all_sources: set[str] = set()
        for col in collections:
            all_sources.update(await self._cached_sources(col))

        return [
            app_commands.Choice(name=s, value=s)
            for s in sorted(all_sources)
            if current.lower() in s.lower()
        ][:25]

    async def _model_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        """Only offer embedding models that are actually present in the index."""
        self._cache_clear_if_stale()
        col_info = await self._cached_col_info()
        present = sorted(
            {info["model"] for info in col_info.values() if info["model"] in MODELS}
        )
        return [
            app_commands.Choice(name=MODELS[m].description, value=m)
            for m in present
            if current.lower() in m.lower()
        ][:25]

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
        return [
            app_commands.Choice(name=CHUNKING_STRATEGIES[c], value=c)
            for c in present
            if current.lower() in c.lower()
        ][:25]

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
        model = getattr(interaction.namespace, "model", "") or settings.model
        chunker = settings.chunker

        show = getattr(interaction.namespace, "show", "")
        resolved = self._resolve_shows(settings, show)
        if resolved.is_locked:
            return []

        all_speakers: set[str] = set()
        if resolved.is_specific:
            for s in resolved.shows:
                col = self.local.resolve_collection("", model, chunker, show_label=s)
                if col:
                    all_speakers.update(await self._cached_speakers(col))
        else:
            collections, _ = await self._visible_collections(settings, model, chunker)
            for col in collections:
                all_speakers.update(await self._cached_speakers(col))

        return [
            app_commands.Choice(name=s, value=s)
            for s in sorted(all_speakers)
            if current.lower() in s.lower()
        ][:25]

    async def _pinned_show_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        """Autocomplete from allowed shows for the server."""
        settings = self._server_settings(interaction.guild_id)
        # Stored as ids; both the label and the value users see are names.
        labels = [self._label_for_show_id(s) for s in settings.allowed_shows]
        return [
            app_commands.Choice(name=s, value=s)
            for s in labels
            if current.lower() in s.lower()
        ][:25]
