"""
podcodex.bot.bot — Discord bot for podcast transcript search.

Entrypoint:
    podcodex-bot [--model bge-m3] [--chunking semantic] [--top-k 5]
                 [--qdrant-url URL] [--server-config FILE]
                 [--shows-config shows.toml] [--dev-guild ID]
    podcodex-bot --hash-password
    podcodex-bot --add-show [--shows-config FILE]

Slash commands (user-facing):
    /search   question [show] [episode] [speaker] [alpha] [model] [top_k] [source] [compact]
              Hybrid search: alpha blends keyword (0) ↔ semantic (1).
    /exact    query [show] [episode] [speaker] [top_k] [source] [compact]
              Literal substring match (case-insensitive, like Ctrl+F).
    /random   [show] [episode] [speaker] [source]
              Pull a random quote from the transcripts.
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
    /unlock   show password
              Unlock a show for this server (requires shows.toml).
    /lock     show
              Remove a show from this server.
    /sync     Manually sync the command tree.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
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
    count_occurrences,
    fmt_time,
    fmt_timestamp,
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
class ShowEntry:
    """A single show registered by the bot owner in shows.toml."""

    name: str
    password_hash: str  # "sha256:<hex>"


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
    shows: dict[str, ShowEntry] = field(default_factory=dict)


@dataclass
class ServerSettings:
    """Per-server overrides persisted to server_config.json."""

    model: str = DEFAULT_MODEL
    chunker: str = DEFAULT_CHUNKING
    top_k: int = TOP_K
    allowed_shows: list[str] = field(default_factory=list)
    default_source: str = ""
    compact: bool = False


# ── Shows config ─────────────────────────────


def _load_shows_config(path: Path) -> dict[str, ShowEntry]:
    """Load shows.toml → {normalized_name: ShowEntry}."""
    if not path.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    shows: dict[str, ShowEntry] = {}
    for _key, entry in raw.get("shows", {}).items():
        name = entry.get("name", _key)
        pw_hash = entry.get("password_hash", "")
        if not pw_hash:
            logger.warning(f"Show {name!r} has no password_hash — skipping")
            continue
        shows[name.lower()] = ShowEntry(name=name, password_hash=pw_hash)
    logger.info(f"Loaded {len(shows)} show(s) from {path}")
    return shows


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored 'sha256:<hex>' hash."""
    if not stored_hash.startswith("sha256:"):
        return False
    expected = stored_hash.removeprefix("sha256:")
    actual = hashlib.sha256(password.encode()).hexdigest()
    return actual == expected


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
    episode_display = chunk.get("episode_title") or episode
    start = chunk.get("start", 0.0)
    end = chunk.get("end", 0.0)
    score = chunk.get("score", 0.0)

    q = question or query
    description = speaker_lines(chunk, query=query)
    embed = discord.Embed(description=description, color=discord.Color.blurple())
    if q:
        embed.set_author(name=f'🔎 "{q}"')
    title = episode_display or "(untitled)"
    if show:
        title += f" ({show})"
    embed.title = title
    timed = chunk.get("timed", True)
    ts_label = fmt_timestamp(start, end, timed=timed)
    if ts_label:
        embed.add_field(name="Timestamp", value=ts_label, inline=True)
    clamped = max(0.0, min(1.0, score))
    embed.add_field(
        name="Relevance", value=f"{score_bar(clamped)} {clamped:.0%}", inline=True
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
        self._access_control = bool(config.shows)

        self._store: QdrantStore | None = None
        self._retrievers: dict[str, Retriever] = {}
        self._server_cfg: dict[int, ServerSettings] = self._load_server_config()
        self._ac_cache = _AutocompleteCache(episodes={}, sources={}, speakers={})

        self._register_commands()

    # ── Config persistence ────────────────────

    def _load_server_config(self) -> dict[int, ServerSettings]:
        if not self.server_config_path.exists():
            return {}
        raw = json.loads(self.server_config_path.read_text(encoding="utf-8"))
        valid_keys = {f.name for f in fields(ServerSettings)}
        result: dict[int, ServerSettings] = {}
        for sid, d in raw.items():
            # Backward compat: rename old "default_shows" → "allowed_shows"
            if "default_shows" in d and "allowed_shows" not in d:
                d["allowed_shows"] = d.pop("default_shows")
            filtered = {k: v for k, v in d.items() if k in valid_keys}
            result[int(sid)] = ServerSettings(**filtered)
        return result

    def _save_server_config(self) -> None:
        payload = json.dumps(
            {str(k): asdict(v) for k, v in self._server_cfg.items()}, indent=2
        )
        tmp = self.server_config_path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
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
            allowed_shows=base.allowed_shows,
            default_source=base.default_source,
            compact=base.compact,
        )

    # ── Access control helpers ────────────────

    def _resolve_shows(
        self, settings: ServerSettings, explicit_show: str = ""
    ) -> list[str] | None:
        """Resolve which shows a command may query.

        Returns a list of show names, or None (= all shows, when access
        control is off and no filter is set).

        With access control ON (shows.toml loaded):
          - explicit show must be in allowed_shows, otherwise rejected
          - no explicit show → allowed_shows (empty = no access)
        With access control OFF (no shows.toml):
          - explicit show → [show]
          - allowed_shows set → allowed_shows
          - neither → None (all shows)
        """
        if self._access_control:
            allowed = settings.allowed_shows
            if explicit_show:
                if explicit_show in allowed:
                    return [explicit_show]
                return []  # not allowed → empty = no results
            return allowed if allowed else []
        # No access control — original behavior
        if explicit_show:
            return [explicit_show]
        if settings.allowed_shows:
            return settings.allowed_shows
        return None

    def _filter_collections(
        self, collections: list[str], settings: ServerSettings
    ) -> list[str]:
        """Filter a list of collections to only those the server may access."""
        if not self._access_control:
            return collections
        allowed = settings.allowed_shows
        if not allowed:
            return []
        allowed_cols = {
            collection_name(s, settings.model, settings.chunker) for s in allowed
        }
        return [c for c in collections if c in allowed_cols]

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

        if self._access_control:
            # Only offer shows this server has unlocked
            shows = sorted(settings.allowed_shows)
        else:
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
        shows = self._resolve_shows(settings, show)
        if not shows:
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
        collections = self._filter_collections(collections, settings)

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

        show = getattr(interaction.namespace, "show", "")
        shows = self._resolve_shows(settings, show)

        if shows:
            all_speakers: set[str] = set()
            for s in shows:
                col = collection_name(s, model, chunker)
                all_speakers.update(await self._cached_speakers(col))
        elif shows is None:
            # No access control, no filter — list speakers across all collections
            loop = asyncio.get_running_loop()
            collections = await loop.run_in_executor(
                None, lambda: self.store.list_collections(model=model, chunker=chunker)
            )
            all_speakers: set[str] = set()
            for col in collections:
                all_speakers.update(await self._cached_speakers(col))
        else:
            return []

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
        return [
            app_commands.Choice(name=s, value=s)
            for s in settings.allowed_shows
            if current.lower() in s.lower()
        ][:25]

    async def _available_show_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        """Autocomplete from shows defined in shows.toml (for /unlock)."""
        available = [entry.name for entry in self.config.shows.values()]
        return [
            app_commands.Choice(name=s, value=s)
            for s in sorted(available)
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

            shows = self._resolve_shows(settings, show)

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

            shows = self._resolve_shows(settings, show)

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

        # /random ─────────────────────────────
        @self.tree.command(
            name="random",
            description="Pull a random quote from the transcripts",
        )
        @app_commands.describe(
            show="Pick a show (random from all if empty)",
            episode="Pick an episode",
            speaker="Filter by speaker name",
            source="Where to search: polished, transcript, etc.",
        )
        async def random_cmd(
            interaction: discord.Interaction,
            show: str = "",
            episode: str = "",
            speaker: str = "",
            source: str = "",
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            settings = self._server_settings(interaction.guild_id)
            effective_source = source or settings.default_source or None

            shows = self._resolve_shows(settings, show)

            await self._run_random(
                interaction,
                shows,
                source=effective_source,
                episode=episode or None,
                speaker=speaker or None,
            )

        random_cmd.autocomplete("show")(self._show_autocomplete)
        random_cmd.autocomplete("episode")(self._episode_autocomplete)
        random_cmd.autocomplete("source")(self._source_autocomplete)
        random_cmd.autocomplete("speaker")(self._speaker_autocomplete)

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

        # /unlock ─────────────────────────────
        @self.tree.command(
            name="unlock",
            description="Unlock a show for this server (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(
            show="Name of the show to unlock",
            password="Access password provided by the bot owner",
        )
        async def unlock(
            interaction: discord.Interaction,
            show: str,
            password: str,
        ) -> None:
            await self._handle_unlock(interaction, show, password)

        unlock.autocomplete("show")(self._available_show_autocomplete)

        # /lock ───────────────────────────────
        @self.tree.command(
            name="lock",
            description="Remove a show from this server (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(show="Name of the show to lock")
        async def lock(
            interaction: discord.Interaction,
            show: str,
        ) -> None:
            await self._handle_lock(interaction, show)

        lock.autocomplete("show")(self._pinned_show_autocomplete)

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
                name="/random",
                value="Pull a random quote — optionally filter by show, episode, or speaker.",
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
                    "Configure server defaults: model, top-k, "
                    "default source, compact mode."
                ),
                inline=False,
            )
            if self._access_control:
                embed.add_field(
                    name="/unlock *(admin)*",
                    value="Unlock a show for this server with a password.",
                    inline=False,
                )
                embed.add_field(
                    name="/lock *(admin)*",
                    value="Remove a show from this server.",
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
            collections = self._filter_collections(
                self.store.list_collections(model=model, chunker=chunker), settings
            )
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
                collections = self._filter_collections(
                    await loop.run_in_executor(
                        None,
                        lambda: self.store.list_collections(
                            model=settings.model,
                            chunker=settings.chunker,
                        ),
                    ),
                    settings,
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

        total_mentions = sum(
            count_occurrences(c.get("text", ""), query) for c, _ in all_results
        )
        label = (
            f"exact match 🔍 · {total_mentions} mention"
            f"{'s' if total_mentions != 1 else ''} in {len(all_results)} chunk"
            f"{'s' if len(all_results) != 1 else ''}"
        )
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

    # ── /random handler ───────────────────────

    async def _run_random(
        self,
        interaction: discord.Interaction,
        shows: list[str] | None,
        *,
        source: str | None = None,
        episode: str | None = None,
        speaker: str | None = None,
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
                collections = self._filter_collections(
                    await loop.run_in_executor(
                        None,
                        lambda: self.store.list_collections(
                            model=settings.model,
                            chunker=settings.chunker,
                        ),
                    ),
                    settings,
                )
            if not collections:
                await interaction.followup.send(
                    "No indexed shows found.", ephemeral=True
                )
                return

            col = random.choice(collections)
            retriever = self.retriever(settings.model)
            chunk = await loop.run_in_executor(
                None,
                lambda: retriever.random(
                    col, episode=episode, source=source, speaker=speaker
                ),
            )
        except Exception:
            logger.exception("Random quote error")
            await interaction.followup.send(
                "❌ Could not fetch a random quote.", ephemeral=True
            )
            return

        if chunk is None:
            await interaction.followup.send("No segments found.", ephemeral=True)
            return

        show = chunk.get("show", "")
        ep = chunk.get("episode", "")
        ep_display = chunk.get("episode_title") or ep
        spk = chunk.get("speaker") or chunk.get("dominant_speaker") or "Unknown"
        start = chunk.get("start", 0.0)
        end = chunk.get("end", 0.0)
        text = chunk.get("text", "")

        embed = discord.Embed(
            description=f'"{text}"',
            color=discord.Color.blurple(),
        )
        title = ep_display or "(untitled)"
        if show:
            title += f" ({show})"
        embed.title = title
        embed.add_field(name="Speaker", value=spk, inline=True)
        timed = chunk.get("timed", True)
        ts_label = fmt_timestamp(start, end, timed=timed)
        if ts_label:
            embed.add_field(name="Timestamp", value=ts_label, inline=True)
        pool = chunk.get("_pool_size")
        footer_text = "🎲 random quote"
        if pool:
            footer_text += f" (from {pool:,} segments)"
        embed.set_footer(text=footer_text)

        view = ExpandView(col, ep, show, start)
        await interaction.followup.send(embed=embed, view=view)

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

        # When access control is on, show management goes through /unlock + /lock
        if self._access_control and (show_add or show_remove or show_clear):
            await interaction.response.send_message(
                "Show access is managed via `/unlock` and `/lock`.",
                ephemeral=True,
            )
            return

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
            if self._access_control:
                shows_str = (
                    ", ".join(f"`{s}`" for s in current.allowed_shows)
                    or "*(none — use /unlock)*"
                )
            else:
                shows_str = (
                    ", ".join(f"`{s}`" for s in current.allowed_shows) or "*(all)*"
                )
            await interaction.response.send_message(
                f"**Current settings**\n"
                f"Model: `{current.model}`\n"
                f"Chunker: `{current.chunker}`\n"
                f"Top-k: `{current.top_k}`\n"
                f"Shows: {shows_str}\n"
                f"Default source: `{current.default_source or '(any)'}`\n"
                f"Compact: `{current.compact}`\n"
                f"Merge: `{self.config.merge_strategy}`",
                ephemeral=True,
            )
            return

        # Build updated shows list (only when access control is off)
        new_shows = list(current.allowed_shows)
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
            allowed_shows=new_shows,
            default_source=default_source if default_source else current.default_source,
            compact=compact == "true" if compact else current.compact,
        )
        self._server_cfg[guild_id] = updated
        self._save_server_config()
        logger.info(f"Guild {guild_id} updated: {updated}")

        shows_str = ", ".join(f"`{s}`" for s in updated.allowed_shows) or "*(all)*"
        await interaction.response.send_message(
            f"✅ Settings updated\n"
            f"Model: `{updated.model}`\n"
            f"Chunker: `{updated.chunker}`\n"
            f"Top-k: `{updated.top_k}`\n"
            f"Shows: {shows_str}\n"
            f"Default source: `{updated.default_source or '(any)'}`\n"
            f"Compact: `{updated.compact}`",
            ephemeral=True,
        )

    # ── /unlock + /lock handlers ────────────────

    async def _handle_unlock(
        self,
        interaction: discord.Interaction,
        show: str,
        password: str,
    ) -> None:
        if not self._access_control:
            await interaction.response.send_message(
                "Access control is not enabled (no shows.toml configured).",
                ephemeral=True,
            )
            return

        # Look up show by name (case-insensitive)
        entry = self.config.shows.get(show.lower())
        if not entry:
            await interaction.response.send_message(
                f"Unknown show: **{show}**", ephemeral=True
            )
            return

        if not _verify_password(password, entry.password_hash):
            logger.warning(
                f"Failed unlock attempt for {show!r} in guild {interaction.guild_id}"
            )
            await interaction.response.send_message(
                "Incorrect password.", ephemeral=True
            )
            return

        guild_id = interaction.guild_id
        settings = self._server_settings(guild_id)
        if entry.name not in settings.allowed_shows:
            settings.allowed_shows.append(entry.name)
            self._server_cfg[guild_id] = settings
            self._save_server_config()
            logger.info(f"Guild {guild_id} unlocked show {entry.name!r}")

        await interaction.response.send_message(
            f"Show **{entry.name}** is now available on this server.",
            ephemeral=True,
        )

    async def _handle_lock(
        self,
        interaction: discord.Interaction,
        show: str,
    ) -> None:
        if not self._access_control:
            await interaction.response.send_message(
                "Access control is not enabled (no shows.toml configured).",
                ephemeral=True,
            )
            return

        guild_id = interaction.guild_id
        settings = self._server_settings(guild_id)
        if show in settings.allowed_shows:
            settings.allowed_shows.remove(show)
            self._server_cfg[guild_id] = settings
            self._save_server_config()
            logger.info(f"Guild {guild_id} locked show {show!r}")
            await interaction.response.send_message(
                f"Show **{show}** has been removed from this server.",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                f"Show **{show}** is not currently unlocked.", ephemeral=True
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
            collections = self._filter_collections(collections, settings)
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

        # Auto-resolve show: explicit > allowed_shows > single show > ask
        show_auto_resolved = not show
        # Check access control for explicit show
        if show and self._access_control and show not in settings.allowed_shows:
            await interaction.followup.send("No indexed shows found.", ephemeral=True)
            return

        if not show:
            if settings.allowed_shows:
                show = settings.allowed_shows[0]
            else:
                collections = self._filter_collections(
                    await loop.run_in_executor(
                        None,
                        lambda: self.store.list_collections(
                            model=settings.model, chunker=settings.chunker
                        ),
                    ),
                    settings,
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
        if show_auto_resolved:
            footer += f" (auto-selected: {show})"

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


def _hash_password_cli() -> None:
    """Interactive helper to generate a sha256 password hash."""
    import getpass

    password = getpass.getpass("Enter password: ")
    if not password:
        print("Empty password — aborting.")
        sys.exit(1)
    h = hashlib.sha256(password.encode()).hexdigest()
    print(f"sha256:{h}")


def _add_show_cli(shows_config: str) -> None:
    """Interactive helper to add a show to shows.toml."""
    import getpass
    import re

    path = Path(shows_config)

    # Ask for show name
    name = input("Show name: ").strip()
    if not name:
        print("Empty name — aborting.")
        sys.exit(1)

    # Generate TOML key from name (same logic as collection naming)
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    # Check for duplicates
    if path.exists():
        content = path.read_text(encoding="utf-8")
        if f"[shows.{key}]" in content:
            print(f"Show '{name}' (key: {key}) already exists in {path}")
            sys.exit(1)
    else:
        content = ""

    # Ask for password
    password = getpass.getpass("Password: ")
    if not password:
        print("Empty password — aborting.")
        sys.exit(1)
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords don't match — aborting.")
        sys.exit(1)

    h = hashlib.sha256(password.encode()).hexdigest()

    # Append to file
    entry = f'\n[shows.{key}]\nname = "{name}"\npassword_hash = "sha256:{h}"\n'
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)

    print(f"Added '{name}' to {path}")


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
        "--shows-config",
        default=None,
        help="Path to shows.toml for password-gated access control",
    )
    parser.add_argument(
        "--dev-guild", default=None, type=int, help="Guild ID for instant dev sync"
    )
    parser.add_argument(
        "--hash-password",
        action="store_true",
        help="Generate a sha256 password hash and exit",
    )
    parser.add_argument(
        "--add-show",
        action="store_true",
        help="Interactively add a show to shows.toml and exit",
    )
    args = parser.parse_args()

    if args.hash_password:
        _hash_password_cli()
        return

    if args.add_show:
        config_path = args.shows_config or "shows.toml"
        _add_show_cli(config_path)
        return

    token = os.environ.get("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_TOKEN not set — add it to .env or environment.")

    shows = {}
    shows_path = Path(args.shows_config) if args.shows_config else Path("shows.toml")
    if shows_path.exists():
        shows = _load_shows_config(shows_path)

    config = BotConfig(
        model=args.model,
        chunker=args.chunking,
        top_k=args.top_k,
        qdrant_url=args.qdrant_url,
        merge_strategy=args.merge_strategy,
        cooldown_seconds=args.cooldown,
        dev_guild_id=args.dev_guild,
        shows=shows,
    )

    bot = PodCodexBot(config, server_config_path=Path(args.server_config))
    logger.info(f"Starting PodCodex bot (model={config.model}, top_k={config.top_k})")
    if shows:
        logger.info(f"Access control ON: {len(shows)} show(s) registered")
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
