"""
podcodex.bot.bot — Discord bot for podcast transcript search.

Entrypoint:
    podcodex-bot [--model bge-m3] [--chunking semantic] [--top-k 5]
                 [--qdrant-url URL] [--server-config FILE] [--dev-guild ID]

Slash commands (user-facing):
    /search   question [show] [episode] [speaker] [alpha] [model] [top_k] [source] [compact]
              Hybrid search: alpha blends keyword (0) ↔ semantic (1).
    /exact    query [show] [episode] [speaker] [top_k] [source] [compact]
              Literal substring match (case-insensitive, like Ctrl+F).
    /stats    [show] [model]
              Index overview: shows, episodes, segments, duration.
    /episodes show [model]
              List episodes for a show with segment counts.

Slash commands (info):
    /help     Show available commands and how to use them.

Slash commands (admin):
    /setup    [model] [chunker] [top_k] [show_add] [show_remove] [show_clear]
              [default_source] [compact]
              Configure server defaults.
    /sync     Manually sync the command tree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import discord
from discord import app_commands
from loguru import logger

from podcodex.bot.formatting import (
    CooldownManager,
    build_compact_embed,
    fmt_time,
    merge_results,
    score_bar,
    speaker_lines,
)
from podcodex.bot.ui import ExpandView, NoExpandView, PaginatedResultView
from podcodex.rag.defaults import (
    ALPHA,
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING,
    DEFAULT_MODEL,
    MODELS,
    TOP_K,
)
from podcodex.rag.retriever import Retriever
from podcodex.rag.store import QdrantStore, collection_name

# ── Slash-command choices ─────────────────────

_MODEL_CHOICES = [
    app_commands.Choice(name=spec.description, value=key)
    for key, spec in MODELS.items()
]

_CHUNKER_CHOICES = [
    app_commands.Choice(name=desc, value=key)
    for key, desc in CHUNKING_STRATEGIES.items()
]

_BOOL_CHOICES = [
    app_commands.Choice(name="True", value="true"),
    app_commands.Choice(name="False", value="false"),
]


# ── Autocomplete cache ───────────────────────


@dataclass
class _AutocompleteCache:
    episodes: dict[str, list[str]]  # collection -> episode names
    sources: dict[str, list[str]]  # collection -> source values
    speakers: dict[str, list[str]]  # collection -> speaker names
    timestamp: float = 0.0
    ttl: float = 300.0  # 5 minutes

    def is_stale(self) -> bool:
        return (time.monotonic() - self.timestamp) > self.ttl


# ── Config dataclasses ────────────────────────


@dataclass
class BotConfig:
    """Global bot configuration (set via CLI flags, immutable at runtime)."""

    model: str = DEFAULT_MODEL
    chunker: str = DEFAULT_CHUNKING
    top_k: int = TOP_K
    qdrant_url: str | None = None
    merge_strategy: str = "roundrobin"
    cooldown_seconds: float = 5.0
    dev_guild_id: int | None = None


@dataclass
class ServerSettings:
    """Per-server overrides persisted to server_config.json."""

    model: str = DEFAULT_MODEL
    chunker: str = DEFAULT_CHUNKING
    top_k: int = TOP_K
    default_shows: list[str] = field(default_factory=list)
    default_source: str = ""
    compact: bool = False


# ── Embed builder ─────────────────────────────


def _result_embed(
    chunk: dict,
    rank: int,
    total: int,
    collection: str,
    label: str,
    query: str = "",
    question: str = "",
) -> tuple[discord.Embed, ExpandView]:
    """Build a Discord embed + context-expand view for a single search result."""
    show = chunk.get("show", "")
    episode = chunk.get("episode", "")
    start = chunk.get("start", 0.0)
    end = chunk.get("end", 0.0)
    score = chunk.get("score", 0.0)

    q = question or query
    description = speaker_lines(chunk, query=query)
    embed = discord.Embed(description=description, color=discord.Color.blurple())
    if q:
        embed.set_author(name=f'🔎 "{q}"')
    title = episode or "(untitled)"
    if show:
        title += f" ({show})"
    embed.title = title
    embed.add_field(
        name="Timestamp", value=f"{fmt_time(start)} → {fmt_time(end)}", inline=True
    )
    embed.add_field(
        name="Relevance", value=f"{score_bar(score)} {score:.0%}", inline=True
    )
    embed.set_footer(text=f"#{rank} of {total} • {label}")

    return embed, ExpandView(collection, episode, show, start)


# ── Bot ───────────────────────────────────────


class PodCodexBot(discord.Client):
    """Discord client with slash commands for searching podcast transcripts."""

    def __init__(self, config: BotConfig, server_config_path: Path) -> None:
        super().__init__(intents=discord.Intents.default())
        self.config = config
        self.server_config_path = server_config_path
        self.tree = app_commands.CommandTree(self)
        self._cooldown = CooldownManager(seconds=config.cooldown_seconds)

        self._store: QdrantStore | None = None
        self._retrievers: dict[str, Retriever] = {}
        self._server_cfg: dict[int, ServerSettings] = self._load_server_config()
        self._ac_cache = _AutocompleteCache(episodes={}, sources={}, speakers={})

        self._register_commands()

    # ── Config persistence ────────────────────

    def _load_server_config(self) -> dict[int, ServerSettings]:
        if not self.server_config_path.exists():
            return {}
        raw = json.loads(self.server_config_path.read_text())
        valid_keys = {f.name for f in fields(ServerSettings)}
        return {
            int(sid): ServerSettings(**{k: v for k, v in d.items() if k in valid_keys})
            for sid, d in raw.items()
        }

    def _save_server_config(self) -> None:
        payload = json.dumps(
            {str(k): asdict(v) for k, v in self._server_cfg.items()}, indent=2
        )
        tmp = self.server_config_path.with_suffix(".tmp")
        tmp.write_text(payload)
        tmp.replace(self.server_config_path)

    def _server_settings(self, guild_id: int | None) -> ServerSettings:
        if guild_id and guild_id in self._server_cfg:
            return self._server_cfg[guild_id]
        return ServerSettings(
            model=self.config.model,
            chunker=self.config.chunker,
            top_k=self.config.top_k,
        )

    def _effective_settings(
        self,
        guild_id: int | None,
        model: str = "",
        top_k: int = 0,
    ) -> ServerSettings:
        """Merge per-query overrides with server defaults."""
        base = self._server_settings(guild_id)
        return ServerSettings(
            model=model or base.model,
            chunker=base.chunker,
            top_k=top_k or base.top_k,
            default_shows=base.default_shows,
            default_source=base.default_source,
            compact=base.compact,
        )

    # ── Lazy singletons ──────────────────────

    @property
    def store(self) -> QdrantStore:
        if self._store is None:
            self._store = QdrantStore(url=self.config.qdrant_url)
        return self._store

    def retriever(self, model: str) -> Retriever:
        if model not in self._retrievers:
            self._retrievers[model] = Retriever(model=model, store=self.store)
        return self._retrievers[model]

    # ── Lifecycle ─────────────────────────────

    async def setup_hook(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.store)
        if self.config.dev_guild_id:
            guild = discord.Object(id=self.config.dev_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info(f"Dev: commands synced to guild {self.config.dev_guild_id}")
        else:
            await self.tree.sync()
            logger.info("Commands synced globally")

    async def on_ready(self) -> None:
        cmds = [c.name for c in self.tree.get_commands()]
        logger.success(
            f"Logged in as {self.user} (id={self.user.id}) — commands: {cmds}"
        )

    # ── Cooldown ──────────────────────────────

    async def _check_cooldown(self, interaction: discord.Interaction) -> bool:
        """Return True if the user may proceed; sends an ephemeral notice if not."""
        remaining = self._cooldown.check(interaction.user.id)
        if remaining > 0:
            await interaction.response.send_message(
                f"⏳ Please wait **{remaining:.1f}s** before searching again.",
                ephemeral=True,
            )
            return False
        self._cooldown.consume(interaction.user.id)
        return True

    # ── Autocomplete ──────────────────────────

    async def _show_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        settings = self._server_settings(interaction.guild_id)
        model = getattr(interaction.namespace, "model", "") or settings.model
        chunker = settings.chunker
        loop = asyncio.get_running_loop()
        collections = await loop.run_in_executor(
            None,
            lambda: self.store.list_collections(model=model, chunker=chunker),
        )
        suffix = f"__{model}__{chunker}"
        shows = sorted({c.removesuffix(suffix) for c in collections})
        return [
            app_commands.Choice(name=s, value=s)
            for s in shows
            if current.lower() in s.lower()
        ][:25]

    async def _cached_episodes(self, collection: str) -> list[str]:
        """Return episode names, using the TTL cache."""
        cache = self._ac_cache
        if cache.is_stale():
            cache.episodes.clear()
            cache.sources.clear()
            cache.speakers.clear()
            cache.timestamp = time.monotonic()
        if collection not in cache.episodes:
            loop = asyncio.get_running_loop()
            eps = await loop.run_in_executor(
                None, lambda: self.store.list_episode_names(collection)
            )
            cache.episodes[collection] = eps
        return cache.episodes.get(collection, [])

    async def _cached_sources(self, collection: str) -> list[str]:
        """Return source values, using the TTL cache."""
        cache = self._ac_cache
        if cache.is_stale():
            cache.episodes.clear()
            cache.sources.clear()
            cache.speakers.clear()
            cache.timestamp = time.monotonic()
        if collection not in cache.sources:
            loop = asyncio.get_running_loop()
            srcs = await loop.run_in_executor(
                None, lambda: self.store.list_sources(collection)
            )
            cache.sources[collection] = srcs
        return cache.sources.get(collection, [])

    async def _episode_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        settings = self._server_settings(interaction.guild_id)
        model = getattr(interaction.namespace, "model", "") or settings.model
        chunker = settings.chunker

        # Determine which show(s) to look at
        show = getattr(interaction.namespace, "show", "")
        if show:
            shows = [show]
        elif settings.default_shows:
            shows = settings.default_shows
        else:
            return []

        all_episodes: set[str] = set()
        for s in shows:
            col = collection_name(s, model, chunker)
            all_episodes.update(await self._cached_episodes(col))

        return [
            app_commands.Choice(name=ep, value=ep)
            for ep in sorted(all_episodes)
            if current.lower() in ep.lower()
        ][:25]

    async def _source_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        settings = self._server_settings(interaction.guild_id)
        model = getattr(interaction.namespace, "model", "") or settings.model
        chunker = settings.chunker

        loop = asyncio.get_running_loop()
        collections = await loop.run_in_executor(
            None, lambda: self.store.list_collections(model=model, chunker=chunker)
        )

        all_sources: set[str] = set()
        for col in collections:
            all_sources.update(await self._cached_sources(col))

        return [
            app_commands.Choice(name=s, value=s)
            for s in sorted(all_sources)
            if current.lower() in s.lower()
        ][:25]

    async def _cached_speakers(self, collection: str) -> list[str]:
        """Return speaker names, using the TTL cache."""
        cache = self._ac_cache
        if cache.is_stale():
            cache.episodes.clear()
            cache.sources.clear()
            cache.speakers.clear()
            cache.timestamp = time.monotonic()
        if collection not in cache.speakers:
            loop = asyncio.get_running_loop()
            spks = await loop.run_in_executor(
                None, lambda: self.store.list_speakers(collection)
            )
            cache.speakers[collection] = spks
        return cache.speakers.get(collection, [])

    async def _speaker_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        settings = self._server_settings(interaction.guild_id)
        model = getattr(interaction.namespace, "model", "") or settings.model
        chunker = settings.chunker

        # Determine which show(s) to look at
        show = getattr(interaction.namespace, "show", "")
        if show:
            shows = [show]
        elif settings.default_shows:
            shows = settings.default_shows
        else:
            # No show context — list speakers across all collections
            loop = asyncio.get_running_loop()
            collections = await loop.run_in_executor(
                None, lambda: self.store.list_collections(model=model, chunker=chunker)
            )
            all_speakers: set[str] = set()
            for col in collections:
                all_speakers.update(await self._cached_speakers(col))
            return [
                app_commands.Choice(name=s, value=s)
                for s in sorted(all_speakers)
                if current.lower() in s.lower()
            ][:25]

        all_speakers: set[str] = set()
        for s in shows:
            col = collection_name(s, model, chunker)
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
        """Autocomplete from pinned default_shows for the server."""
        settings = self._server_settings(interaction.guild_id)
        return [
            app_commands.Choice(name=s, value=s)
            for s in settings.default_shows
            if current.lower() in s.lower()
        ][:25]

    # ── Command registration ──────────────────

    def _register_commands(self) -> None:

        # /search ─────────────────────────────
        @self.tree.command(
            name="search",
            description="Search podcast transcripts (keyword ↔ semantic blend)",
        )
        @app_commands.describe(
            question="What are you looking for?",
            show="Pick a show (searches all if empty)",
            episode="Pick an episode",
            speaker="Filter by speaker name",
            alpha="0 = keywords only → 1 = meaning only (default 0.5)",
            model="Search model (leave empty for server default)",
            top_k="How many results to show (leave empty for server default)",
            source="Where to search: polished, transcript, etc.",
            compact="Show results in a single compact embed",
        )
        @app_commands.choices(model=_MODEL_CHOICES, compact=_BOOL_CHOICES)
        async def search(
            interaction: discord.Interaction,
            question: str,
            show: str = "",
            episode: str = "",
            speaker: str = "",
            alpha: app_commands.Range[float, 0.0, 1.0] = ALPHA,
            model: str = "",
            top_k: app_commands.Range[int, 1, 25] = 0,
            source: str = "",
            compact: str = "",
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            settings = self._effective_settings(interaction.guild_id, model, top_k)
            effective_source = source or settings.default_source or None
            use_compact = compact == "true" if compact else settings.compact

            # Resolve shows: explicit > default_shows > all
            shows: list[str] | None = None
            if show:
                shows = [show]
            elif settings.default_shows:
                shows = settings.default_shows

            label = f"α={alpha:.2f} • {MODELS[settings.model].label}"
            await self._run_search(
                interaction,
                question,
                shows,
                settings,
                alpha,
                label,
                source=effective_source,
                episode=episode or None,
                speaker=speaker or None,
                compact=use_compact,
            )

        search.autocomplete("show")(self._show_autocomplete)
        search.autocomplete("episode")(self._episode_autocomplete)
        search.autocomplete("source")(self._source_autocomplete)
        search.autocomplete("speaker")(self._speaker_autocomplete)

        # /exact ──────────────────────────────
        @self.tree.command(
            name="exact",
            description="Literal substring search — case-insensitive, like Ctrl+F",
        )
        @app_commands.describe(
            query="Text to find (not case-sensitive)",
            show="Pick a show (searches all if empty)",
            episode="Pick an episode",
            speaker="Filter by speaker name",
            top_k="How many results to show (default 25)",
            source="Where to search: polished, transcript, etc.",
            compact="Show results in a single compact embed",
        )
        @app_commands.choices(compact=_BOOL_CHOICES)
        async def exact(
            interaction: discord.Interaction,
            query: str,
            show: str = "",
            episode: str = "",
            speaker: str = "",
            top_k: app_commands.Range[int, 1, 50] = 25,
            source: str = "",
            compact: str = "",
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            settings = self._server_settings(interaction.guild_id)
            effective_source = source or settings.default_source or None
            use_compact = compact == "true" if compact else settings.compact

            shows: list[str] | None = None
            if show:
                shows = [show]
            elif settings.default_shows:
                shows = settings.default_shows

            await self._run_exact(
                interaction,
                query,
                shows,
                top_k,
                source=effective_source,
                episode=episode or None,
                speaker=speaker or None,
                compact=use_compact,
            )

        exact.autocomplete("show")(self._show_autocomplete)
        exact.autocomplete("episode")(self._episode_autocomplete)
        exact.autocomplete("source")(self._source_autocomplete)
        exact.autocomplete("speaker")(self._speaker_autocomplete)

        # /stats ──────────────────────────────
        @self.tree.command(
            name="stats",
            description="Index overview: shows, episodes, segments, duration",
        )
        @app_commands.describe(
            show="Pick a show (shows all if empty)",
            model="Search model (leave empty for server default)",
        )
        @app_commands.choices(model=_MODEL_CHOICES)
        async def stats(
            interaction: discord.Interaction,
            show: str = "",
            model: str = "",
        ) -> None:
            await self._handle_stats(interaction, show or None, model or None)

        stats.autocomplete("show")(self._show_autocomplete)

        # /episodes ───────────────────────────
        @self.tree.command(
            name="episodes",
            description="List episodes for a show with segment count and duration",
        )
        @app_commands.describe(
            show="Pick a show (auto-selected if only one)",
            model="Search model (leave empty for server default)",
        )
        @app_commands.choices(model=_MODEL_CHOICES)
        async def episodes(
            interaction: discord.Interaction,
            show: str = "",
            model: str = "",
        ) -> None:
            await self._handle_episodes(interaction, show or None, model or None)

        episodes.autocomplete("show")(self._show_autocomplete)

        # /setup ──────────────────────────────
        @self.tree.command(
            name="setup",
            description="Configure bot defaults for this server (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(
            model="Default search model for this server",
            chunker="How transcripts are split up for search",
            top_k="How many results to show by default",
            show_add="Always search this show by default",
            show_remove="Stop searching this show by default",
            show_clear="Remove all default shows (search everything)",
            default_source="Default source to search: polished, transcript, etc.",
            compact="Use compact results by default",
        )
        @app_commands.choices(
            model=_MODEL_CHOICES,
            chunker=_CHUNKER_CHOICES,
            show_clear=_BOOL_CHOICES,
            compact=_BOOL_CHOICES,
        )
        async def setup(
            interaction: discord.Interaction,
            model: str = "",
            chunker: str = "",
            top_k: app_commands.Range[int, 1, 25] = 0,
            show_add: str = "",
            show_remove: str = "",
            show_clear: str = "",
            default_source: str = "",
            compact: str = "",
        ) -> None:
            await self._handle_setup(
                interaction,
                model or None,
                chunker or None,
                top_k or None,
                show_add=show_add or None,
                show_remove=show_remove or None,
                show_clear=show_clear == "true",
                default_source=default_source,
                compact=compact,
            )

        setup.autocomplete("show_add")(self._show_autocomplete)
        setup.autocomplete("show_remove")(self._pinned_show_autocomplete)
        setup.autocomplete("default_source")(self._source_autocomplete)

        # /sync ───────────────────────────────
        @self.tree.command(
            name="sync",
            description="Manually sync slash commands (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        async def sync(interaction: discord.Interaction) -> None:
            await interaction.response.defer(ephemeral=True)
            await self.tree.sync()
            await interaction.followup.send("✅ Command tree synced.", ephemeral=True)

        # /help ───────────────────────────────
        @self.tree.command(
            name="help",
            description="Show available commands and how to use them",
        )
        async def help_cmd(interaction: discord.Interaction) -> None:
            embed = discord.Embed(
                title="📖 PodCodex Help",
                description="Search and explore your podcast transcripts.",
                color=discord.Color.blurple(),
            )
            embed.add_field(
                name="/search `question`",
                value=(
                    "Find relevant passages using a mix of keyword and semantic search.\n"
                    "`alpha` controls the blend: 0 = keywords only, 1 = meaning only (default 0.5)."
                ),
                inline=False,
            )
            embed.add_field(
                name="/exact `query`",
                value="Find exact text matches, like Ctrl+F across all episodes.",
                inline=False,
            )
            embed.add_field(
                name="/episodes `show`",
                value="List all indexed episodes for a show.",
                inline=False,
            )
            embed.add_field(
                name="/stats",
                value="Overview of what's indexed: shows, episodes, segments, duration.",
                inline=False,
            )
            embed.add_field(
                name="/setup *(admin)*",
                value=(
                    "Configure server defaults: model, top-k, pinned shows, "
                    "default source, compact mode."
                ),
                inline=False,
            )
            embed.set_footer(
                text="Use the Show context ↕ button on results to see surrounding dialogue."
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

    # ── /search handler ───────────────────────

    async def _run_search(
        self,
        interaction: discord.Interaction,
        question: str,
        shows: list[str] | None,
        settings: ServerSettings,
        alpha: float,
        label: str,
        *,
        source: str | None = None,
        episode: str | None = None,
        speaker: str | None = None,
        compact: bool = False,
    ) -> None:
        await interaction.response.defer()
        loop = asyncio.get_running_loop()

        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._hybrid_search(
                    question,
                    shows,
                    settings,
                    alpha,
                    source=source,
                    episode=episode,
                    speaker=speaker,
                ),
            )
        except Exception:
            logger.exception(f"Search error: {question!r}")
            await interaction.followup.send(
                "❌ Something went wrong. Please try again later.",
                ephemeral=True,
            )
            return

        if not results:
            await interaction.followup.send("No results found.", ephemeral=True)
            return

        if compact:
            embed = build_compact_embed(results, label, question=question)
            await interaction.followup.send(embed=embed)
        else:
            pages = [
                _result_embed(chunk, rank, len(results), col, label, question=question)
                for rank, (chunk, col) in enumerate(results, 1)
            ]
            view = PaginatedResultView(pages)
            await interaction.followup.send(embed=view.current_embed, view=view)

    def _hybrid_search(
        self,
        question: str,
        shows: list[str] | None,
        settings: ServerSettings,
        alpha: float,
        *,
        source: str | None = None,
        episode: str | None = None,
        speaker: str | None = None,
    ) -> list[tuple[dict, str]]:
        """Run hybrid retrieval across collections and merge results."""
        model, chunker = settings.model, settings.chunker
        if shows:
            collections = [collection_name(s, model, chunker) for s in shows]
        else:
            collections = self.store.list_collections(model=model, chunker=chunker)
        if not collections:
            logger.warning(f"No collections for model={model!r} chunker={chunker!r}")
            return []

        ret = self.retriever(model)
        hits_by_col: dict[str, list[dict]] = {}
        for col in collections:
            hits = ret.retrieve(
                question,
                col,
                top_k=settings.top_k,
                alpha=alpha,
                source=source,
                episode=episode,
                speaker=speaker,
            )
            if hits:
                hits_by_col[col] = hits

        merged = merge_results(
            hits_by_col,
            top_k=settings.top_k,
            strategy=self.config.merge_strategy,
        )
        return [r for r in merged if r[0].get("score", 0) > 0.05]

    # ── /exact handler ────────────────────────

    async def _run_exact(
        self,
        interaction: discord.Interaction,
        query: str,
        shows: list[str] | None,
        top_k: int,
        *,
        source: str | None = None,
        episode: str | None = None,
        speaker: str | None = None,
        compact: bool = False,
    ) -> None:
        await interaction.response.defer()
        settings = self._server_settings(interaction.guild_id)
        loop = asyncio.get_running_loop()

        try:
            if shows:
                collections = [
                    collection_name(s, settings.model, settings.chunker) for s in shows
                ]
            else:
                collections = await loop.run_in_executor(
                    None,
                    lambda: self.store.list_collections(
                        model=settings.model,
                        chunker=settings.chunker,
                    ),
                )
            if not collections:
                await interaction.followup.send(
                    "No indexed shows found.", ephemeral=True
                )
                return

            all_results: list[tuple[dict, str]] = []
            for col in collections:
                hits = await loop.run_in_executor(
                    None,
                    lambda c=col: self.retriever(settings.model).find(
                        query,
                        c,
                        top_k=top_k,
                        source=source,
                        episode=episode,
                        speaker=speaker,
                    ),
                )
                all_results.extend((hit, col) for hit in hits)

        except Exception:
            logger.exception(f"Exact search error: {query!r}")
            await interaction.followup.send("❌ Search failed.", ephemeral=True)
            return

        if not all_results:
            await interaction.followup.send(
                f'No matches for **"{query}"**.',
                ephemeral=True,
            )
            return

        all_results.sort(
            key=lambda x: (x[0].get("episode", ""), x[0].get("start", 0.0))
        )
        all_results = all_results[:top_k]

        label = "exact match 🔍"
        if compact:
            embed = build_compact_embed(all_results, label, query=query)
            await interaction.followup.send(embed=embed)
        else:
            pages = [
                _result_embed(chunk, rank, len(all_results), col, label, query=query)
                for rank, (chunk, col) in enumerate(all_results, 1)
            ]
            view = PaginatedResultView(pages)
            await interaction.followup.send(embed=view.current_embed, view=view)

    # ── /setup handler ────────────────────────

    async def _handle_setup(
        self,
        interaction: discord.Interaction,
        model: str | None,
        chunker: str | None,
        top_k: int | None,
        *,
        show_add: str | None = None,
        show_remove: str | None = None,
        show_clear: bool = False,
        default_source: str = "",
        compact: str = "",
    ) -> None:
        guild_id = interaction.guild_id
        current = self._server_settings(guild_id)

        has_change = any(
            [
                model,
                chunker,
                top_k,
                show_add,
                show_remove,
                show_clear,
                default_source,
                compact,
            ]
        )
        if not has_change:
            shows_str = ", ".join(f"`{s}`" for s in current.default_shows) or "*(all)*"
            await interaction.response.send_message(
                f"**Current settings**\n"
                f"Model: `{current.model}`\n"
                f"Chunker: `{current.chunker}`\n"
                f"Top-k: `{current.top_k}`\n"
                f"Default shows: {shows_str}\n"
                f"Default source: `{current.default_source or '(any)'}`\n"
                f"Compact: `{current.compact}`\n"
                f"Merge: `{self.config.merge_strategy}`",
                ephemeral=True,
            )
            return

        # Build updated shows list
        new_shows = list(current.default_shows)
        if show_clear:
            new_shows = []
        if show_add and show_add not in new_shows:
            new_shows.append(show_add)
        if show_remove and show_remove in new_shows:
            new_shows.remove(show_remove)

        updated = ServerSettings(
            model=model or current.model,
            chunker=chunker or current.chunker,
            top_k=top_k or current.top_k,
            default_shows=new_shows,
            default_source=default_source if default_source else current.default_source,
            compact=compact == "true" if compact else current.compact,
        )
        self._server_cfg[guild_id] = updated
        self._save_server_config()
        logger.info(f"Guild {guild_id} updated: {updated}")

        shows_str = ", ".join(f"`{s}`" for s in updated.default_shows) or "*(all)*"
        await interaction.response.send_message(
            f"✅ Settings updated\n"
            f"Model: `{updated.model}`\n"
            f"Chunker: `{updated.chunker}`\n"
            f"Top-k: `{updated.top_k}`\n"
            f"Default shows: {shows_str}\n"
            f"Default source: `{updated.default_source or '(any)'}`\n"
            f"Compact: `{updated.compact}`",
            ephemeral=True,
        )

    # ── /stats handler ────────────────────────

    async def _handle_stats(
        self,
        interaction: discord.Interaction,
        show: str | None,
        model: str | None,
    ) -> None:
        await interaction.response.defer()
        settings = self._effective_settings(interaction.guild_id, model or "", 0)
        loop = asyncio.get_running_loop()

        try:
            collections = await loop.run_in_executor(
                None,
                lambda: self.store.list_collections(
                    show=show or "",
                    model=settings.model,
                    chunker=settings.chunker,
                ),
            )
            if not collections:
                await interaction.followup.send(
                    "No indexed shows found.", ephemeral=True
                )
                return

            all_stats: list[dict] = []
            for col in collections:
                stats = await loop.run_in_executor(
                    None,
                    lambda c=col: self.store.get_episode_stats(c),
                )
                all_stats.extend(stats)

        except Exception:
            logger.exception("Stats error")
            await interaction.followup.send(
                "❌ Could not retrieve stats.", ephemeral=True
            )
            return

        total_eps = len(all_stats)
        total_chunks = sum(s["chunk_count"] for s in all_stats)
        total_dur = sum(s["duration"] for s in all_stats)

        suffix = f"__{settings.model}__{settings.chunker}"
        show_names = sorted({c.removesuffix(suffix) for c in collections})

        embed = discord.Embed(
            title="📊 PodCodex Index Stats", color=discord.Color.blurple()
        )
        embed.add_field(name="Shows", value=str(len(collections)), inline=True)
        embed.add_field(name="Episodes", value=str(total_eps), inline=True)
        embed.add_field(name="Segments", value=str(total_chunks), inline=True)
        embed.add_field(name="Duration", value=fmt_time(total_dur), inline=True)
        embed.add_field(
            name="Indexed shows",
            value="\n".join(f"🎙 {s}" for s in show_names) or "—",
            inline=False,
        )
        embed.set_footer(text=f"Model: {MODELS[settings.model].label}")

        await interaction.followup.send(embed=embed)

    # ── /episodes handler ─────────────────────

    async def _handle_episodes(
        self,
        interaction: discord.Interaction,
        show: str | None,
        model: str | None,
    ) -> None:
        await interaction.response.defer()
        settings = self._effective_settings(interaction.guild_id, model or "", 0)
        loop = asyncio.get_running_loop()

        # Auto-resolve show: explicit > default_shows > single show > ask
        if not show:
            if settings.default_shows:
                show = settings.default_shows[0]
            else:
                collections = await loop.run_in_executor(
                    None,
                    lambda: self.store.list_collections(
                        model=settings.model, chunker=settings.chunker
                    ),
                )
                suffix = f"__{settings.model}__{settings.chunker}"
                shows = sorted({c.removesuffix(suffix) for c in collections})
                if len(shows) == 1:
                    show = shows[0]
                elif not shows:
                    await interaction.followup.send(
                        "No indexed shows found.", ephemeral=True
                    )
                    return
                else:
                    names = "\n".join(f"🎙 {s}" for s in shows)
                    await interaction.followup.send(
                        f"Multiple shows available — please specify one:\n{names}",
                        ephemeral=True,
                    )
                    return

        col = collection_name(show, settings.model, settings.chunker)

        try:
            ep_stats = await loop.run_in_executor(
                None,
                lambda: self.store.get_episode_stats(col),
            )
        except Exception:
            logger.exception(f"Episodes error for {col!r}")
            await interaction.followup.send(
                "❌ Could not retrieve episode list.",
                ephemeral=True,
            )
            return

        if not ep_stats:
            await interaction.followup.send(
                f"No episodes found for **{show}**.",
                ephemeral=True,
            )
            return

        # Paginate: 10 episodes per embed
        pages_data = [ep_stats[i : i + 10] for i in range(0, len(ep_stats), 10)]
        footer = f"{len(ep_stats)} episodes"

        embeds: list[discord.Embed] = []
        for page in pages_data:
            embed = discord.Embed(title=f"🎙 {show}", color=discord.Color.blurple())
            for ep in page:
                speakers = ", ".join(ep.get("speakers", [])) or "—"
                embed.add_field(
                    name=ep["episode"],
                    value=f"{speakers} · `{fmt_time(ep['duration'])}`",
                    inline=False,
                )
            embed.set_footer(text=footer)
            embeds.append(embed)

        if len(embeds) == 1:
            await interaction.followup.send(embed=embeds[0])
        else:
            pages = [(e, NoExpandView()) for e in embeds]
            view = PaginatedResultView(pages)
            await interaction.followup.send(embed=view.current_embed, view=view)


# ── Entrypoint ────────────────────────────────


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(prog="podcodex-bot")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()))
    parser.add_argument(
        "--chunking", default=DEFAULT_CHUNKING, choices=list(CHUNKING_STRATEGIES.keys())
    )
    parser.add_argument("--top-k", default=TOP_K, type=int)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument(
        "--merge-strategy", default="roundrobin", choices=["roundrobin", "score"]
    )
    parser.add_argument(
        "--cooldown", default=5.0, type=float, help="Per-user cooldown (seconds)"
    )
    parser.add_argument("--server-config", default="server_config.json")
    parser.add_argument(
        "--dev-guild", default=None, type=int, help="Guild ID for instant dev sync"
    )
    args = parser.parse_args()

    token = os.environ.get("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_TOKEN not set — add it to .env or environment.")

    config = BotConfig(
        model=args.model,
        chunker=args.chunking,
        top_k=args.top_k,
        qdrant_url=args.qdrant_url,
        merge_strategy=args.merge_strategy,
        cooldown_seconds=args.cooldown,
        dev_guild_id=args.dev_guild,
    )

    bot = PodCodexBot(config, server_config_path=Path(args.server_config))
    logger.info(f"Starting PodCodex bot (model={config.model}, top_k={config.top_k})")
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
