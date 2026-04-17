"""
podcodex.bot.bot — Discord bot for podcast transcript search.

Entrypoint:
    podcodex-bot [--model bge-m3] [--chunking semantic] [--top-k 5]
                 [--db PATH] [--server-config FILE]
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
    /unlock         password
                    Unlock a show for this server (password identifies the show).
    /lock           show
                    Remove a show from this server.
    /changepassword show
                    Rotate the password for an already-unlocked show; sends new
                    password via DM.
    /sync           Manually sync the command tree.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import random
import re
import secrets
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
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
    truncate_description,
)
from podcodex.bot.synthesis import synthesize_answer
from podcodex.bot.ui import ExpandView, NoExpandView, PaginatedResultView, SourcesView
from podcodex.rag.defaults import (
    ALPHA,
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING,
    DEFAULT_MODEL,
    MODELS,
    TOP_K,
)
from podcodex.rag.index_store import IndexStore
from podcodex.rag.retriever import Retriever
from podcodex.rag.store import collection_name

# ── Display helpers ───────────────────────────

_STEM_PREFIX_RE = re.compile(r"^\d+_(?:episode_\d+_)?", re.IGNORECASE)


def _humanize_stem(stem: str) -> str:
    s = _STEM_PREFIX_RE.sub("", stem).replace("_", " ").strip()
    return (s[:1].upper() + s[1:]) if s else stem


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


# ── Access control ───────────────────────────


class ShowAccess(Enum):
    """Outcome of resolving an optional show argument against access policy."""

    ALL = "all"  # no explicit show → enumerate all server-visible collections
    LOCKED = "locked"  # explicit show exists but is password-protected & not unlocked
    SPECIFIC = "specific"  # explicit show is accessible; query only that show


@dataclass(frozen=True)
class ResolvedShows:
    access: ShowAccess
    shows: tuple[str, ...] = ()

    @property
    def is_all(self) -> bool:
        return self.access is ShowAccess.ALL

    @property
    def is_locked(self) -> bool:
        return self.access is ShowAccess.LOCKED

    @property
    def is_specific(self) -> bool:
        return self.access is ShowAccess.SPECIFIC


# ── Autocomplete cache ───────────────────────


@dataclass
class _AutocompleteCache:
    episodes: dict[str, list[str]]  # collection -> episode stems
    episode_titles: dict[str, dict[str, str]]  # collection -> {stem: rss_title}
    sources: dict[str, list[str]]  # collection -> source values
    speakers: dict[str, list[str]]  # collection -> speaker names
    col_info: dict[str, dict] | None = None  # {collection: {show, model, chunker, dim}}
    timestamp: float = 0.0
    ttl: float = 300.0  # 5 minutes

    def is_stale(self) -> bool:
        return (time.monotonic() - self.timestamp) > self.ttl


# ── Config dataclasses ────────────────────────


@dataclass
class ShowEntry:
    """A single password-protected show (stored in the IndexStore)."""

    name: str
    password_hash: str  # "sha256:<hex>"


@dataclass
class BotConfig:
    """Global bot configuration (set via CLI flags, immutable at runtime)."""

    model: str = DEFAULT_MODEL
    chunker: str = DEFAULT_CHUNKING
    top_k: int = TOP_K
    index_path: str | None = None
    merge_strategy: str = "roundrobin"
    cooldown_seconds: float = 5.0
    dev_guild_id: int | None = None
    # /ask LLM config — defaults for all servers
    ask_provider: str = ""
    ask_model: str = ""
    ask_api_url: str = ""
    ask_api_key: str | None = None
    ask_cooldown_seconds: float = 30.0


@dataclass
class ServerSettings:
    """Per-server overrides persisted to server_config.json."""

    model: str = DEFAULT_MODEL
    chunker: str = DEFAULT_CHUNKING
    top_k: int = TOP_K
    allowed_shows: list[str] = field(default_factory=list)
    default_source: str = ""
    compact: bool = False
    # /ask LLM config (API key never stored — read from env at call time)
    ask_provider: str = ""
    ask_model: str = ""
    ask_api_url: str = ""
    ask_cooldown_seconds: float = 30.0


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored 'sha256:<hex>' hash."""
    # Hash before format check so malformed rows take the same time as valid ones.
    actual = hashlib.sha256(password.encode()).hexdigest()
    if not stored_hash.startswith("sha256:"):
        return False
    expected = stored_hash.removeprefix("sha256:")
    return hmac.compare_digest(actual, expected)


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
    description = truncate_description(speaker_lines(chunk, query=query))
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
        self._ask_cooldown = CooldownManager(seconds=config.ask_cooldown_seconds)

        # Shows loaded from IndexStore — {lower(name): ShowEntry}
        # Populated in setup_hook once the index is open; refreshable via /admin.
        self._shows: dict[str, ShowEntry] = {}

        self._local: IndexStore | None = None
        self._retrievers: dict[str, Retriever] = {}
        self._server_cfg: dict[int, ServerSettings] = self._load_server_config()
        self._ac_cache = _AutocompleteCache(
            episodes={}, episode_titles={}, sources={}, speakers={}
        )

        self._register_commands()

    @property
    def _locked_show_names(self) -> set[str]:
        return {e.name for e in self._shows.values()}

    def _reload_shows(self) -> None:
        """Refresh password-protected shows from IndexStore."""
        raw = self.local.get_show_passwords()  # {name: hash}
        self._shows = {
            name.lower(): ShowEntry(name=name, password_hash=pw_hash)
            for name, pw_hash in raw.items()
        }
        logger.info(f"Shows loaded: {len(self._shows)} password-protected")

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
            ask_provider=self.config.ask_provider,
            ask_model=self.config.ask_model,
            ask_api_url=self.config.ask_api_url,
            ask_cooldown_seconds=self.config.ask_cooldown_seconds,
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
            ask_provider=base.ask_provider,
            ask_model=base.ask_model,
            ask_api_url=base.ask_api_url,
            ask_cooldown_seconds=base.ask_cooldown_seconds,
        )

    # ── Access control helpers ────────────────

    def _resolve_shows(
        self, settings: ServerSettings, explicit_show: str = ""
    ) -> "ResolvedShows":
        """Resolve which shows a command may query.

        Shows are public by default. A show becomes password-protected only
        when the bot owner sets a password via ``--manage-passwords``.
        """
        if not explicit_show:
            return ResolvedShows(ShowAccess.ALL)
        if (
            explicit_show in self._locked_show_names
            and explicit_show not in settings.allowed_shows
        ):
            return ResolvedShows(ShowAccess.LOCKED)
        return ResolvedShows(ShowAccess.SPECIFIC, (explicit_show,))

    def _filter_collections(
        self,
        collections: list[str],
        settings: ServerSettings,
        col_info: dict[str, dict] | None = None,
    ) -> list[str]:
        """Filter collections to those the server may access.

        Public shows (no password set) are always included.
        Password-protected shows are included only if unlocked on this server.

        Args:
            col_info: Pre-fetched ``{name: {show, ...}}`` map from
                ``local.get_all_collection_info()``. If omitted, fetched on
                demand (adds N+1 queries; prefer passing it from the caller).
        """
        if not self._locked_show_names:
            return collections  # nothing is password-protected
        info_map = (
            col_info if col_info is not None else self.local.get_all_collection_info()
        )
        allowed = set(settings.allowed_shows)
        return [
            col
            for col in collections
            if (info_map.get(col) or {}).get("show", "") not in self._locked_show_names
            or (info_map.get(col) or {}).get("show", "") in allowed
        ]

    # ── Lazy singletons ──────────────────────

    @property
    def local(self) -> IndexStore:
        if self._local is None:
            self._local = IndexStore(self.config.index_path)
        return self._local

    def retriever(self, model: str) -> Retriever:
        if model not in self._retrievers:
            self._retrievers[model] = Retriever(model=model, local=self.local)
        return self._retrievers[model]

    # ── Lifecycle ─────────────────────────────

    async def setup_hook(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.local)
        await loop.run_in_executor(None, self._reload_shows)
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

    async def _check_cooldown_for(
        self,
        interaction: discord.Interaction,
        manager: CooldownManager,
        action: str,
        *,
        seconds: float | None = None,
    ) -> bool:
        """Return True if the user may proceed; sends an ephemeral notice if not."""
        remaining = manager.check(interaction.user.id, seconds=seconds)
        if remaining > 0:
            await interaction.response.send_message(
                f"⏳ Please wait **{remaining:.1f}s** before {action} again.",
                ephemeral=True,
            )
            return False
        manager.consume(interaction.user.id)
        return True

    async def _check_cooldown(self, interaction: discord.Interaction) -> bool:
        return await self._check_cooldown_for(interaction, self._cooldown, "searching")

    async def _check_ask_cooldown(self, interaction: discord.Interaction) -> bool:
        settings = self._server_settings(interaction.guild_id)
        return await self._check_cooldown_for(
            interaction,
            self._ask_cooldown,
            "asking",
            seconds=settings.ask_cooldown_seconds,
        )

    # ── Autocomplete ──────────────────────────

    async def _show_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
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
        cache = self._ac_cache
        if cache.is_stale():
            cache.episodes.clear()
            cache.episode_titles.clear()
            cache.sources.clear()
            cache.speakers.clear()
            cache.col_info = None
            cache.timestamp = time.monotonic()

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
        self, settings: "ServerSettings", model: str, chunker: str
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
            cols = [collection_name(s, model, chunker) for s in resolved.shows]
        else:
            cols, _ = await self._visible_collections(settings, model, chunker)
        for col in cols:
            stems = await self._cached_episodes(col)
            titles = await self._cached_episode_titles(col)
            for stem in stems:
                all_episodes[stem] = titles.get(stem) or _humanize_stem(stem)

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
                col = collection_name(s, model, chunker)
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
        available = [entry.name for entry in self._shows.values()]
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
            source="Where to search: corrected, transcript, etc.",
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

        # /ask ────────────────────────────────
        @self.tree.command(
            name="ask",
            description="Ask a question — retrieves relevant passages and synthesizes an answer",
        )
        @app_commands.describe(
            question="What do you want to know?",
            show="Pick a show (searches all if empty)",
            episode="Pick an episode",
            speaker="Filter by speaker name",
            alpha="0 = keywords only → 1 = meaning only (default 0.5)",
            top_k="How many passages to retrieve (leave empty for server default)",
            source="Where to search: corrected, transcript, etc.",
        )
        async def ask(
            interaction: discord.Interaction,
            question: str,
            show: str = "",
            episode: str = "",
            speaker: str = "",
            alpha: app_commands.Range[float, 0.0, 1.0] = ALPHA,
            top_k: app_commands.Range[int, 1, 25] = 0,
            source: str = "",
        ) -> None:
            if not await self._check_ask_cooldown(interaction):
                return
            await self._handle_ask(
                interaction,
                question,
                show=show or None,
                episode=episode or None,
                speaker=speaker or None,
                alpha=alpha,
                top_k=top_k,
                source=source or None,
            )

        ask.autocomplete("show")(self._show_autocomplete)
        ask.autocomplete("episode")(self._episode_autocomplete)
        ask.autocomplete("source")(self._source_autocomplete)
        ask.autocomplete("speaker")(self._speaker_autocomplete)

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
            source="Where to search: corrected, transcript, etc.",
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
            source="Where to search: corrected, transcript, etc.",
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
            default_source="Default source to search: corrected, transcript, etc.",
            compact="Use compact results by default",
            ask_provider="LLM provider for /ask (openai, mistral, anthropic, custom)",
            ask_model="LLM model name for /ask (leave empty for provider default)",
            ask_api_url="API base URL for /ask (required for 'custom' provider)",
            ask_cooldown="Per-user cooldown for /ask in seconds (default 30)",
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
            ask_provider: str = "",
            ask_model: str = "",
            ask_api_url: str = "",
            ask_cooldown: app_commands.Range[float, 1.0, 300.0] = 0.0,
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
                ask_provider=ask_provider,
                ask_model=ask_model,
                ask_api_url=ask_api_url,
                ask_cooldown=ask_cooldown or None,
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
            description="Unlock a show for this server using a password (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(
            password="Access password provided by the bot owner",
        )
        async def unlock(
            interaction: discord.Interaction,
            password: str,
        ) -> None:
            await self._handle_unlock(interaction, password)

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

        # /changepassword ─────────────────────
        @self.tree.command(
            name="changepassword",
            description="Rotate the password for a show you have already unlocked on this server",
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(show="Show to rotate the password for")
        async def changepassword(
            interaction: discord.Interaction,
            show: str,
        ) -> None:
            await self._handle_changepassword(interaction, show)

        changepassword.autocomplete("show")(self._pinned_show_autocomplete)

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
            ask_settings = self._server_settings(interaction.guild_id)
            ask_configured = bool(ask_settings.ask_provider or ask_settings.ask_model)
            ask_status = (
                "" if ask_configured else " *(LLM not configured — admin: use /setup)*"
            )
            embed.add_field(
                name="/ask `question`",
                value=(
                    f"Ask a question and get a synthesized answer grounded in the transcripts.{ask_status}\n"
                    "Attaches a **Show sources** button to reveal the retrieved passages."
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
            if self._locked_show_names:
                embed.add_field(
                    name="/unlock *(admin)*",
                    value="Unlock a show for this server — provide the password, the bot identifies the show automatically.",
                    inline=False,
                )
                embed.add_field(
                    name="/lock *(admin)*",
                    value="Remove a show from this server.",
                    inline=False,
                )
                embed.add_field(
                    name="/changepassword *(admin)*",
                    value="Rotate the password for an already-unlocked show. New password is sent via DM.",
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
        shows: ResolvedShows,
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
        shows: ResolvedShows,
        settings: ServerSettings,
        alpha: float,
        *,
        source: str | None = None,
        episode: str | None = None,
        speaker: str | None = None,
    ) -> list[tuple[dict, str]]:
        """Run hybrid retrieval across collections and merge results."""
        model, chunker = settings.model, settings.chunker
        if shows.is_specific:
            collections = [collection_name(s, model, chunker) for s in shows.shows]
        elif shows.is_locked:
            collections = []
        else:
            col_info = self.local.get_all_collection_info()
            collections = [
                name
                for name, info in col_info.items()
                if info["model"] == model and info["chunker"] == chunker
            ]
            collections = self._filter_collections(collections, settings, col_info)
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

    # ── /ask handler ─────────────────────────

    async def _handle_ask(
        self,
        interaction: discord.Interaction,
        question: str,
        *,
        show: str | None = None,
        episode: str | None = None,
        speaker: str | None = None,
        alpha: float = ALPHA,
        top_k: int = 0,
        source: str | None = None,
    ) -> None:
        await interaction.response.defer()
        settings = self._effective_settings(interaction.guild_id, "", top_k)
        effective_source = source or settings.default_source or None

        # Validate LLM configured
        if not (settings.ask_provider or settings.ask_model):
            await interaction.followup.send(
                "❌ /ask is not configured for this server.\n"
                "An admin must set `ask_provider` and `ask_model` via `/setup`.",
                ephemeral=True,
            )
            return

        shows = self._resolve_shows(settings, show or "")
        loop = asyncio.get_running_loop()

        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._hybrid_search(
                    question,
                    shows,
                    settings,
                    alpha,
                    source=effective_source,
                    episode=episode,
                    speaker=speaker,
                ),
            )
        except Exception:
            logger.exception(f"ask: retrieval error for {question!r}")
            await interaction.followup.send(
                "❌ Retrieval failed. Please try again later.", ephemeral=True
            )
            return

        if not results:
            await interaction.followup.send(
                "No relevant passages found.", ephemeral=True
            )
            return

        chunks = [chunk for chunk, _col in results]

        try:
            answer = await loop.run_in_executor(
                None,
                lambda: synthesize_answer(
                    question,
                    chunks,
                    provider=settings.ask_provider,
                    model=settings.ask_model,
                    api_base_url=settings.ask_api_url,
                    api_key=self.config.ask_api_key,
                ),
            )
        except ValueError as exc:
            logger.warning(f"ask: config error — {exc}")
            await interaction.followup.send(f"❌ {exc}", ephemeral=True)
            return
        except Exception:
            logger.exception(f"ask: synthesis error for {question!r}")
            await interaction.followup.send(
                "❌ Answer synthesis failed. Please try again later.", ephemeral=True
            )
            return

        label = f"α={alpha:.2f} • {MODELS[settings.model].label}"
        source_pages = [
            _result_embed(chunk, rank, len(results), col, label, question=question)
            for rank, (chunk, col) in enumerate(results, 1)
        ]

        provider_label = settings.ask_provider or settings.ask_model or "LLM"
        embed = discord.Embed(
            description=truncate_description(answer),
            color=discord.Color.gold(),
        )
        embed.set_author(name=f'❓ "{question}"')
        embed.set_footer(
            text=f"{len(chunks)} source{'s' if len(chunks) != 1 else ''} · {provider_label}"
        )

        view = SourcesView(source_pages)
        await interaction.followup.send(embed=embed, view=view)

    # ── /exact handler ────────────────────────

    async def _run_exact(
        self,
        interaction: discord.Interaction,
        query: str,
        shows: ResolvedShows,
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
            if shows.is_specific:
                collections = [
                    collection_name(s, settings.model, settings.chunker)
                    for s in shows.shows
                ]
            elif shows.is_locked:
                collections = []
            else:
                col_info = await loop.run_in_executor(
                    None, self.local.get_all_collection_info
                )
                collections = [
                    name
                    for name, info in col_info.items()
                    if info["model"] == settings.model
                    and info["chunker"] == settings.chunker
                ]
                collections = self._filter_collections(collections, settings, col_info)
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

        # Phrase results (exact + accent) sorted chronologically; fuzzy by BM25 score
        phrase = sorted(
            [r for r in all_results if not r[0].get("fuzzy_match")],
            key=lambda x: (
                x[0].get("score", 1.0) < 1.0,
                x[0].get("episode", ""),
                x[0].get("start", 0.0),
            ),
        )
        fuzzy = sorted(
            [r for r in all_results if r[0].get("fuzzy_match")],
            key=lambda x: -x[0].get("score", 0.6),
        )
        all_results = (phrase + fuzzy)[:top_k]

        n_exact = sum(
            1
            for c, _ in all_results
            if not c.get("accent_match") and not c.get("fuzzy_match")
        )
        n_accent = sum(1 for c, _ in all_results if c.get("accent_match"))
        n_fuzzy = sum(1 for c, _ in all_results if c.get("fuzzy_match"))
        total_mentions = sum(
            count_occurrences(c.get("text", ""), query) for c, _ in all_results
        )
        label_parts = [f"exact match 🔍 · {n_exact} exact"]
        if n_accent:
            label_parts.append(f"{n_accent} variant{'s' if n_accent != 1 else ''}")
        if n_fuzzy:
            label_parts.append(f"{n_fuzzy} near-typo{'s' if n_fuzzy != 1 else ''}")
        label_parts.append(
            f"{total_mentions} mention{'s' if total_mentions != 1 else ''} "
            f"in {len(all_results)} chunk{'s' if len(all_results) != 1 else ''}"
        )
        label = " · ".join(label_parts)
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
        shows: ResolvedShows,
        *,
        source: str | None = None,
        episode: str | None = None,
        speaker: str | None = None,
    ) -> None:
        await interaction.response.defer()
        settings = self._server_settings(interaction.guild_id)
        loop = asyncio.get_running_loop()

        try:
            if shows.is_specific:
                collections = [
                    collection_name(s, settings.model, settings.chunker)
                    for s in shows.shows
                ]
            elif shows.is_locked:
                collections = []
            else:
                col_info = await loop.run_in_executor(
                    None, self.local.get_all_collection_info
                )
                collections = [
                    name
                    for name, info in col_info.items()
                    if info["model"] == settings.model
                    and info["chunker"] == settings.chunker
                ]
                collections = self._filter_collections(collections, settings, col_info)
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
        ask_provider: str = "",
        ask_model: str = "",
        ask_api_url: str = "",
        ask_cooldown: float | None = None,
    ) -> None:
        guild_id = interaction.guild_id
        current = self._server_settings(guild_id)

        # Password-protected shows are managed via /unlock + /lock, not /setup
        if self._locked_show_names and (show_add or show_remove or show_clear):
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
                ask_provider,
                ask_model,
                ask_api_url,
                ask_cooldown,
            ]
        )
        if not has_change:
            if self._locked_show_names:
                shows_str = (
                    ", ".join(f"`{s}`" for s in current.allowed_shows)
                    or "*(none — use /unlock)*"
                )
            else:
                shows_str = (
                    ", ".join(f"`{s}`" for s in current.allowed_shows)
                    or "*(all public)*"
                )
            ask_status = (
                current.ask_provider or current.ask_model or "*(not configured)*"
            )
            await interaction.response.send_message(
                f"**Current settings**\n"
                f"Model: `{current.model}`\n"
                f"Chunker: `{current.chunker}`\n"
                f"Top-k: `{current.top_k}`\n"
                f"Shows: {shows_str}\n"
                f"Default source: `{current.default_source or '(any)'}`\n"
                f"Compact: `{current.compact}`\n"
                f"Merge: `{self.config.merge_strategy}`\n"
                f"Ask provider: `{ask_status}`\n"
                f"Ask model: `{current.ask_model or '(provider default)'}`\n"
                f"Ask cooldown: `{current.ask_cooldown_seconds:.0f}s`",
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
            ask_provider=ask_provider or current.ask_provider,
            ask_model=ask_model or current.ask_model,
            ask_api_url=ask_api_url or current.ask_api_url,
            ask_cooldown_seconds=ask_cooldown
            if ask_cooldown
            else current.ask_cooldown_seconds,
        )
        self._server_cfg[guild_id] = updated
        self._save_server_config()
        logger.info(f"Guild {guild_id} updated: {updated}")

        shows_str = (
            ", ".join(f"`{s}`" for s in updated.allowed_shows) or "*(all public)*"
        )
        ask_status = updated.ask_provider or updated.ask_model or "*(not configured)*"
        await interaction.response.send_message(
            f"✅ Settings updated\n"
            f"Model: `{updated.model}`\n"
            f"Chunker: `{updated.chunker}`\n"
            f"Top-k: `{updated.top_k}`\n"
            f"Shows: {shows_str}\n"
            f"Default source: `{updated.default_source or '(any)'}`\n"
            f"Compact: `{updated.compact}`\n"
            f"Ask provider: `{ask_status}`\n"
            f"Ask model: `{updated.ask_model or '(provider default)'}`\n"
            f"Ask cooldown: `{updated.ask_cooldown_seconds:.0f}s`",
            ephemeral=True,
        )

    # ── /unlock + /lock handlers ────────────────

    async def _handle_unlock(
        self,
        interaction: discord.Interaction,
        password: str,
    ) -> None:
        # Find show by password — intentionally no show name in the command
        # so available show names are never exposed to users.
        entry = next(
            (
                e
                for e in self._shows.values()
                if _verify_password(password, e.password_hash)
            ),
            None,
        )
        if entry is None:
            logger.warning(f"Failed unlock attempt in guild {interaction.guild_id}")
            await interaction.response.send_message("Invalid password.", ephemeral=True)
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

    # ── /changepassword handler ───────────────

    async def _handle_changepassword(
        self,
        interaction: discord.Interaction,
        show: str,
    ) -> None:
        guild_id = interaction.guild_id
        settings = self._server_settings(guild_id)

        if show not in settings.allowed_shows:
            await interaction.response.send_message(
                f"Show **{show}** is not unlocked on this server. Unlock it first with /unlock.",
                ephemeral=True,
            )
            return

        # Generate new password and update the index.
        password = secrets.token_urlsafe(16)
        h = hashlib.sha256(password.encode()).hexdigest()
        self.local.set_show_password(show, f"sha256:{h}")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._reload_shows)

        # Send via DM to keep the password out of the guild audit log.
        try:
            await interaction.user.send(
                f"New password for **{show}**:\n```\n{password}\n```\n"
                "Share this with the show owner. It cannot be recovered after this message."
            )
            await interaction.response.send_message(
                f"Password for **{show}** has been rotated. Check your DMs.",
                ephemeral=True,
            )
        except discord.HTTPException:
            # DM failed (blocked, rate-limited, server error) — fall back to ephemeral
            # so the caller still receives the new password.
            await interaction.response.send_message(
                f"Password for **{show}** rotated.\n"
                f"**Could not send DM** (enable DMs from server members, or try again).\n"
                f"New password: `{password}`",
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
            col_info = await loop.run_in_executor(
                None, self.local.get_all_collection_info
            )
            collections = [
                name
                for name, info in col_info.items()
                if (not show or info["show"] == show)
                and info["model"] == settings.model
                and info["chunker"] == settings.chunker
            ]
            collections = self._filter_collections(collections, settings, col_info)
            if not collections:
                await interaction.followup.send(
                    "No indexed shows found.", ephemeral=True
                )
                return

            all_stats: list[dict] = []
            for col in collections:
                stats = await loop.run_in_executor(
                    None,
                    lambda c=col: self.local.get_episode_stats(c),
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

        show_names = sorted(
            {col_info.get(col, {}).get("show") or col for col in collections}
        )

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

        # Auto-resolve show: explicit > unlocked > single accessible show > ask
        show_auto_resolved = not show
        # Check access control for explicit show
        if show and self._resolve_shows(settings, show).is_locked:
            await interaction.followup.send("No indexed shows found.", ephemeral=True)
            return

        if not show:
            if settings.allowed_shows:
                show = settings.allowed_shows[0]
            else:
                col_info = await loop.run_in_executor(
                    None, self.local.get_all_collection_info
                )
                collections = [
                    name
                    for name, info in col_info.items()
                    if info["model"] == settings.model
                    and info["chunker"] == settings.chunker
                ]
                collections = self._filter_collections(collections, settings, col_info)
                shows = sorted(
                    {col_info.get(c, {}).get("show") or c for c in collections}
                )
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
                lambda: self.local.get_episode_stats(col),
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
                ep_display = ep.get("episode_title") or _humanize_stem(ep["episode"])
                embed.add_field(
                    name=ep_display,
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


def _manage_passwords_cli(index_path: str | None) -> None:
    """Interactive CLI to manage show passwords stored in the IndexStore.

    Lists all indexed shows, shows their current password status, and
    lets the operator set or auto-generate a password for each show.
    Auto-generated passwords use 16 URL-safe random bytes (22 chars).
    """
    import getpass

    from podcodex.rag.index_store import IndexStore

    store = IndexStore(index_path)
    col_info = store.get_all_collection_info()
    if not col_info:
        print("No indexed shows found.")
        return

    show_names = sorted({info.get("show") or name for name, info in col_info.items()})
    existing = store.get_show_passwords()

    print(f"\nFound {len(show_names)} show(s):\n")
    for name in show_names:
        status = "🔒 password set" if name in existing else "🔓 no password (public)"
        print(f"  {name}  —  {status}")

    print("\nEnter a show name to set/update its password, or press Enter to quit.")
    while True:
        name = input("\nShow name (or Enter to quit): ").strip()
        if not name:
            break
        if name not in show_names:
            print(f"  Unknown show. Options: {', '.join(show_names)}")
            continue

        choice = (
            input("  [g]enerate strong password, [s]et manually, [r]emove, [skip]: ")
            .strip()
            .lower()
        )
        if choice == "g":
            password = secrets.token_urlsafe(16)
            h = hashlib.sha256(password.encode()).hexdigest()
            store.set_show_password(name, f"sha256:{h}")
            print(f"  Password for '{name}': {password}")
            print("  (copy this — it cannot be recovered from the stored hash)")
        elif choice == "s":
            password = getpass.getpass("  Password: ")
            if not password:
                print("  Empty — skipped.")
                continue
            if len(password) < 16:
                print(
                    "  Too short (min 16 chars). Use [g]enerate for a strong random password."
                )
                continue
            confirm = getpass.getpass("  Confirm: ")
            if password != confirm:
                print("  Mismatch — skipped.")
                continue
            h = hashlib.sha256(password.encode()).hexdigest()
            store.set_show_password(name, f"sha256:{h}")
            print(f"  Password set for '{name}'.")
        elif choice == "r":
            store.delete_show_password(name)
            print(f"  Password removed — '{name}' is now public.")
        else:
            print("  Skipped.")


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
    parser.add_argument(
        "--index",
        default=None,
        help="Path to LanceDB index directory (default: ~/.local/share/podcodex/index)",
    )
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
    parser.add_argument(
        "--ask-provider",
        default="",
        help="LLM provider for /ask: openai, mistral, anthropic, custom",
    )
    parser.add_argument("--ask-model", default="", help="LLM model name for /ask")
    parser.add_argument(
        "--ask-api-url", default="", help="API base URL for /ask (custom provider)"
    )
    parser.add_argument(
        "--ask-api-key",
        default=None,
        help="API key for /ask (overrides env variable)",
    )
    parser.add_argument(
        "--ask-cooldown",
        default=30.0,
        type=float,
        help="Per-user cooldown for /ask in seconds (default 30)",
    )
    parser.add_argument(
        "--manage-passwords",
        action="store_true",
        help="Interactively manage show passwords in the index and exit",
    )
    args = parser.parse_args()

    if args.manage_passwords:
        _manage_passwords_cli(args.index)
        return

    token = os.environ.get("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_TOKEN not set — add it to .env or environment.")

    config = BotConfig(
        model=args.model,
        chunker=args.chunking,
        top_k=args.top_k,
        index_path=args.index,
        merge_strategy=args.merge_strategy,
        cooldown_seconds=args.cooldown,
        dev_guild_id=args.dev_guild,
        ask_provider=args.ask_provider,
        ask_model=args.ask_model,
        ask_api_url=args.ask_api_url,
        ask_api_key=args.ask_api_key,
        ask_cooldown_seconds=args.ask_cooldown,
    )

    bot = PodCodexBot(config, server_config_path=Path(args.server_config))
    logger.info(f"Starting PodCodex bot (model={config.model}, top_k={config.top_k})")
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
