"""
podcodex.bot.bot — Discord bot for podcast transcript search.

Entrypoint:
    podcodex-bot [--model bge-m3] [--chunking semantic] [--top-k 5]
                 [--index PATH] [--server-config FILE] [--dev-guild ID]
    podcodex-bot --manage-passwords [--index PATH]


Slash commands (user-facing):
    /search   question [show] [episode] [speaker] [alpha] [model] [top_k] [source] [compact]
              Hybrid search: alpha blends keyword (0) ↔ semantic (1).
    /exact    query [show] [episode] [speaker] [top_k] [source] [compact]
              Literal substring match (case-insensitive, like Ctrl+F).
    /random   [show] [episode] [speaker] [source]
              Pull a random quote from the transcripts.
    /stats    [show] [model]
              Index overview: shows, episodes, excerpts, duration.
    /episodes show [model]
              List episodes for a show with excerpt counts.
    /speakers [show] [model]
              Per-speaker chunk counts and airtime, ranked.

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
import secrets
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import tasks
from loguru import logger

from podcodex.bot.announce import (
    AnnounceStore,
    build_new_episodes_embed,
    build_version_embed,
)
from podcodex.bot.formatting import (
    CooldownManager,
    count_occurrences,
    display_speaker,
    episode_display,
    fmt_time,
    fmt_timestamp,
    format_filter_suffix,
    humanize_stem,
    set_chunk_thumbnail,
)
from podcodex.bot.result_store import CachedSearch, ResultRef, SearchCacheStore
from podcodex.bot.ui import (
    DYNAMIC_ITEMS,
    ExpandResult,
    build_compact_view,
    build_episodes_embeds,
    build_list_view,
    build_listen_button,
    build_results_view,
    build_stats_embed,
)
from podcodex.rag.defaults import (
    ALPHA,
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING,
    DEFAULT_MODEL,
    MODELS,
    TOP_K,
)
from podcodex.rag.index_store import IndexStore, _normalize_pub_date, get_index_store
from podcodex.rag.retriever import Retriever, get_retriever, merge_results
from podcodex.rag.store import collection_name

# Throttle for the per-call mtime check. Discord fires autocomplete on every
# keystroke; without throttling, a 10-char query would walk the index dir 10
# times. 2s is well below any realistic rate of out-of-process index changes.
_MTIME_CHECK_INTERVAL = 2.0

# ── Slash-command choices ─────────────────────

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
    announce_interval_minutes: int = 10


@dataclass
class ServerSettings:
    """Per-server overrides persisted to server_config.json."""

    model: str = DEFAULT_MODEL
    chunker: str = DEFAULT_CHUNKING
    top_k: int = TOP_K
    allowed_shows: list[str] = field(default_factory=list)
    default_source: str = ""
    compact: bool = False
    announce_channel_id: int = 0  # 0 = announcements disabled for this server


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored 'sha256:<hex>' hash."""
    # Hash before format check so malformed rows take the same time as valid ones.
    actual = hashlib.sha256(password.encode()).hexdigest()
    if not stored_hash.startswith("sha256:"):
        return False
    expected = stored_hash.removeprefix("sha256:")
    return hmac.compare_digest(actual, expected)


# ── Embed builder ─────────────────────────────


def pick_show_collection(
    cols: list[tuple[str, dict]], pref_model: str, pref_chunker: str
) -> tuple[str, str] | None:
    """Choose one ``(collection, model)`` for a show from its collections.

    Priority: the server's preferred model+chunker, then the global default
    combo, then the first remaining collection by name — so a show indexed
    only under a non-default model stays reachable without the user ever
    naming a model. Returns ``None`` for a show with no collections.
    """
    for name, info in cols:
        if info.get("model") == pref_model and info.get("chunker") == pref_chunker:
            return name, info.get("model", pref_model)
    for name, info in cols:
        if (
            info.get("model") == DEFAULT_MODEL
            and info.get("chunker") == DEFAULT_CHUNKING
        ):
            return name, info["model"]
    if not cols:
        return None
    name, info = min(cols, key=lambda c: c[0])
    return name, info.get("model", DEFAULT_MODEL)


def _chunk_to_ref(chunk: dict, collection: str) -> ResultRef:
    """Distill a search-result chunk into a cacheable :class:`ResultRef`.

    Stores only the pointer (``collection``/``episode``/``chunk_index``) plus the
    search-time scalars LanceDB does not carry; the transcript text is re-fetched
    on click. See :mod:`podcodex.bot.result_store`.
    """
    return ResultRef(
        collection=collection,
        episode=chunk.get("episode", ""),
        chunk_index=int(chunk.get("chunk_index", 0)),
        score=float(chunk.get("score", 0.0)),
        fuzzy_match=bool(chunk.get("fuzzy_match")),
        accent_match=bool(chunk.get("accent_match")),
        match_text=chunk.get("match_text"),
        episode_title=episode_display(chunk),
    )


# ── Bot ───────────────────────────────────────


class PodCodexBot(discord.Client):
    """Discord client with slash commands for searching podcast transcripts."""

    def __init__(self, config: BotConfig, server_config_path: Path) -> None:
        super().__init__(intents=discord.Intents.default())
        self.config = config
        self.server_config_path = server_config_path
        self.tree = app_commands.CommandTree(self)
        self.tree.on_error = self._on_app_command_error
        self._cooldown = CooldownManager(seconds=config.cooldown_seconds)

        # Shows loaded from IndexStore — {lower(name): ShowEntry}
        # Populated in setup_hook once the index is open; refreshable via /admin.
        self._shows: dict[str, ShowEntry] = {}

        self._local: IndexStore | None = None
        self._retrievers: dict[str, Retriever] = {}
        self._server_cfg: dict[int, ServerSettings] = self._load_server_config()
        # Durable + in-RAM cache backing the persistent pagination buttons.
        self.results = SearchCacheStore(server_config_path.parent / "search_cache.db")
        # Diff state for new-episode + version announcements.
        self.announce = AnnounceStore(server_config_path.parent / "announce_state.db")
        self._ac_cache = _AutocompleteCache(
            episodes={}, episode_titles={}, sources={}, speakers={}
        )

        # Newest mtime seen across index-dir entries. Rising value means the
        # on-disk index changed (rsync, password set via API, new show
        # indexed) and bot state must be reloaded.
        self._index_mtime_seen: float = 0.0
        self._last_mtime_check: float = 0.0
        # The announcer keeps its OWN mtime watermark, separate from
        # ``_index_mtime_seen`` — otherwise a user command's refresh would
        # advance the shared value and the loop would never see the change.
        self._announce_mtime_seen: float = 0.0

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
        )

    def _effective_settings(
        self,
        guild_id: int | None,
        model: str = "",
        top_k: int = 0,
        chunker: str = "",
    ) -> ServerSettings:
        """Merge per-query overrides with server defaults."""
        base = self._server_settings(guild_id)
        return replace(
            base,
            model=model or base.model,
            chunker=chunker or base.chunker,
            top_k=top_k or base.top_k,
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

    def _empty_collections_message(
        self,
        col_info: dict[str, dict],
        settings: ServerSettings,
        shows: "ResolvedShows | None" = None,
    ) -> str:
        """Explain to the user why no collections matched.

        Distinguishes: empty index, wrong model, locked/no-unlock, missing show.
        """
        if not col_info:
            return (
                "Nothing has been indexed yet. "
                "Add a show in the desktop app and run the **Index** step."
            )

        model_label = self._model_label(settings.model)
        same_model = {
            info["model"]
            for info in col_info.values()
            if info["chunker"] == settings.chunker
        }

        if settings.model not in same_model:
            others = sorted(m for m in same_model if m)
            hint = (
                f"Available models: {', '.join(others)}. "
                f"Switch with `/setup model:{others[0]}` or pass `model:` to this command."
                if others
                else "No other models available either — index something first."
            )
            return (
                f"No shows are indexed with the **{model_label}** embedding model. "
                f"{hint}"
            )

        if shows and shows.is_locked:
            return (
                "No shows are unlocked for this Discord server. "
                "An admin can unlock one with `/unlock password:****`."
            )

        if shows and shows.is_specific:
            missing = ", ".join(f"**{s}**" for s in shows.shows)
            return f"{missing} is not indexed with the **{model_label}** model on this server."

        return "No shows are available to search here."

    @staticmethod
    def _model_label(model: str) -> str:
        """Human label for a model key; a stale/unknown key passes through raw
        instead of raising (server configs can outlive the MODELS registry)."""
        return MODELS[model].label if model in MODELS else model

    def _show_allowed(self, show_name: str, settings: ServerSettings) -> bool:
        """Whether this server may see ``show_name``: public, or unlocked here."""
        return (
            show_name not in self._locked_show_names
            or show_name in settings.allowed_shows
        )

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
        return [
            col
            for col in collections
            if self._show_allowed((info_map.get(col) or {}).get("show", ""), settings)
        ]

    def _resolve_show_collections(
        self,
        shows: ResolvedShows,
        settings: ServerSettings,
        col_info: dict[str, dict],
    ) -> list[tuple[str, str]]:
        """One ``(collection, model)`` per accessible show.

        The single collection-resolution path for /search, /exact, /random and
        the stats commands. Each show maps to exactly one collection (see
        :func:`pick_show_collection`), so a query never needs a model or
        chunker and a show indexed under any model stays reachable.
        ``is_locked`` yields ``[]`` (no preview leaks); locked-but-unlocked
        shows pass the access filter.
        """
        if shows.is_locked:
            return []
        by_show: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for name, info in col_info.items():
            by_show[info.get("show", "")].append((name, info))

        if shows.is_specific:
            wanted = {s.lower() for s in shows.shows}
            show_keys = [s for s in by_show if s.lower() in wanted]
        else:
            show_keys = list(by_show)

        picked: list[tuple[str, str]] = []
        for show_name in sorted(show_keys):
            choice = pick_show_collection(
                by_show[show_name], settings.model, settings.chunker
            )
            if choice is not None:
                picked.append(choice)

        if not self._locked_show_names:
            return picked
        return [
            (col, model)
            for col, model in picked
            if self._show_allowed((col_info.get(col) or {}).get("show", ""), settings)
        ]

    async def _refresh_if_stale(self) -> None:
        """Detect external index changes and reload bot state.

        Called at the top of every privileged handler so out-of-process
        writes (rsync, desktop-app indexing, API password change) can't
        leak ACL state. Filesystem checks are throttled to one sweep per
        ``_MTIME_CHECK_INTERVAL`` so autocomplete bursts don't compound.
        """
        now = time.monotonic()
        if now - self._last_mtime_check < _MTIME_CHECK_INTERVAL:
            return
        self._last_mtime_check = now
        loop = asyncio.get_running_loop()
        current = await loop.run_in_executor(None, self.local.index_mtime)
        if current <= self._index_mtime_seen:
            return
        self._index_mtime_seen = current
        await loop.run_in_executor(None, self.local.reconnect)
        self._ac_cache.reset()
        await loop.run_in_executor(None, self._reload_shows)
        logger.info("Index refresh: external change detected, bot state reloaded.")

    # ── Lazy singletons ──────────────────────

    @property
    def local(self) -> IndexStore:
        if self._local is None:
            # When no custom path is set, share the process-wide singleton so
            # embedder / Retriever caches are reused across bot + API + MCP.
            self._local = (
                IndexStore(self.config.index_path)
                if self.config.index_path
                else get_index_store()
            )
        return self._local

    def retriever(self, model: str) -> Retriever:
        if self.config.index_path is None:
            return get_retriever(model)
        if model not in self._retrievers:
            self._retrievers[model] = Retriever(model=model, local=self.local)
        return self._retrievers[model]

    # ── Lifecycle ─────────────────────────────

    async def _on_app_command_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        """Last-resort handler: never leave an interaction stuck on 'thinking'.

        Any exception that escapes a command handler after ``defer()`` would
        otherwise keep the 'Bot is thinking…' spinner forever with no reply
        (e.g. a Discord 400 for an over-limit embed). Log it, tell the user.
        """
        logger.opt(exception=error).error(
            f"Unhandled app-command error ({interaction.command and interaction.command.name})"
        )
        msg = "❌ Something went wrong — please try again."
        try:
            if interaction.response.is_done():
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=True)
        except discord.HTTPException:
            pass  # interaction expired; nothing left to notify

    async def setup_hook(self) -> None:
        loop = asyncio.get_running_loop()
        # Re-register persistent pagination handlers so buttons on messages from
        # before a restart keep working.
        for item in DYNAMIC_ITEMS:
            self.add_dynamic_items(item)
        await loop.run_in_executor(None, lambda: self.local)
        await loop.run_in_executor(None, self._reload_shows)
        self._index_mtime_seen = await loop.run_in_executor(
            None, self.local.index_mtime
        )
        if self.config.dev_guild_id:
            guild = discord.Object(id=self.config.dev_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info(f"Dev: commands synced to guild {self.config.dev_guild_id}")
        else:
            await self.tree.sync()
            logger.info("Commands synced globally")

        # Start the announcement poller. before_loop waits until ready.
        self._announce_loop.change_interval(
            minutes=self.config.announce_interval_minutes
        )
        self._announce_loop.start()

    async def on_ready(self) -> None:
        cmds = [c.name for c in self.tree.get_commands()]
        logger.success(
            f"Logged in as {self.user} (id={self.user.id}) — commands: {cmds}"
        )
        await self._announce_version_if_changed()

    # ── Announcements ─────────────────────────

    @tasks.loop(minutes=10)
    async def _announce_loop(self) -> None:
        """Poll the index and announce newly-added episodes. Never raises."""
        try:
            await self._run_announce_tick()
        except Exception:
            logger.exception("Announce loop tick failed")

    @_announce_loop.before_loop
    async def _before_announce(self) -> None:
        await self.wait_until_ready()

    async def _run_announce_tick(self) -> None:
        """One poll: detect index change, diff episodes, post per guild.

        State (baseline + seen episodes) advances regardless of whether any
        server has a channel configured, so enabling announcements later never
        replays the back-catalogue. Posting is best-effort per channel.
        """
        loop = asyncio.get_running_loop()
        current_mtime = await loop.run_in_executor(None, self.local.index_mtime)
        if current_mtime <= self._announce_mtime_seen:
            return
        self._announce_mtime_seen = current_mtime
        await loop.run_in_executor(None, self.local.reconnect)
        await loop.run_in_executor(None, self._reload_shows)
        col_info = await loop.run_in_executor(None, self.local.get_all_collection_info)

        # Diff each collection; observe() advances the seen-state.
        new_cols: dict[str, list[str]] = {}
        for col in col_info:
            stems = await loop.run_in_executor(
                None, lambda c=col: self.local.list_episodes(c)
            )
            new = self.announce.observe(col, set(stems))
            if new:
                new_cols[col] = new
        if not new_cols:
            return

        # Fetch display rows for the new stems, newest first.
        per_col_rows: dict[str, list[dict]] = {}
        for col, new_stems in new_cols.items():
            stats = await loop.run_in_executor(
                None, lambda c=col: self.local.get_episode_stats(c)
            )
            newset = set(new_stems)
            rows = [s for s in stats if s.get("episode") in newset]
            rows.sort(
                key=lambda e: (e.get("pub_date") or "", e.get("episode", "")),
                reverse=True,
            )
            if rows:
                per_col_rows[col] = rows
        if not per_col_rows:
            return

        async for guild_id, settings, channel in self._iter_announce_channels():
            accessible = {
                c
                for c, _ in self._resolve_show_collections(
                    ResolvedShows(ShowAccess.ALL), settings, col_info
                )
            }
            for col, rows in per_col_rows.items():
                if col not in accessible:
                    continue
                show = (col_info.get(col, {}) or {}).get("show", "") or col
                try:
                    await channel.send(embed=build_new_episodes_embed(show, rows))
                except discord.HTTPException:
                    logger.warning(
                        f"Announce send failed (guild {guild_id}, "
                        f"channel {settings.announce_channel_id})"
                    )

    async def _announce_channel(self, channel_id: int):
        """Resolve a configured channel id to a sendable channel, or None."""
        channel = self.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(channel_id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                return None
        return channel if hasattr(channel, "send") else None

    async def _iter_announce_channels(self):
        """Yield ``(guild_id, settings, channel)`` for every server with a
        configured, resolvable announcement channel."""
        for guild_id, settings in self._server_cfg.items():
            if not settings.announce_channel_id:
                continue
            channel = await self._announce_channel(settings.announce_channel_id)
            if channel is not None:
                yield guild_id, settings, channel

    async def _announce_version_if_changed(self) -> None:
        """Announce the bot version once when it changes from the last one.

        First ever run baselines silently (no announce). Idempotent across the
        repeated ``on_ready`` fired on gateway reconnects.
        """
        from importlib.metadata import version as _pkg_version

        try:
            current = _pkg_version("podcodex")
        except Exception:
            return
        if not current:
            return
        stored = self.announce.get_meta("announced_version")
        if stored is None:
            self.announce.set_meta("announced_version", current)
            return
        if stored == current:
            return
        async for guild_id, _settings, channel in self._iter_announce_channels():
            try:
                await channel.send(embed=build_version_embed(current))
            except discord.HTTPException:
                logger.warning(f"Version announce send failed (guild {guild_id})")
        self.announce.set_meta("announced_version", current)

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

    async def _reject_bad_date(
        self,
        interaction: discord.Interaction,
        after: str,
        before: str,
    ) -> bool:
        """Validate ``after``/``before`` from an advanced slash command.

        Returns ``True`` when one of the dates is malformed and an
        ephemeral error was sent — the caller should bail out. Returns
        ``False`` when both are empty or valid.
        """
        for label, value in (("after", after), ("before", before)):
            if value and not _normalize_pub_date(value):
                await interaction.response.send_message(
                    f"❌ Invalid `{label}` date: `{value}`. Use YYYY-MM-DD.",
                    ephemeral=True,
                )
                return True
        return False

    # ── Autocomplete ──────────────────────────

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
            cols = [collection_name(s, model, chunker) for s in resolved.shows]
        else:
            cols, _ = await self._visible_collections(settings, model, chunker)
        for col in cols:
            stems = await self._cached_episodes(col)
            titles = await self._cached_episode_titles(col)
            for stem in stems:
                all_episodes[stem] = titles.get(stem) or humanize_stem(stem)

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
        """Autocomplete from password-protected shows in the index (for /unlock)."""
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
            description="Search podcast transcripts (uses server defaults)",
        )
        @app_commands.describe(query="What are you looking for?")
        async def search(
            interaction: discord.Interaction,
            query: str,
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            await self._refresh_if_stale()
            settings = self._effective_settings(interaction.guild_id)
            shows = self._resolve_shows(settings, "")
            label = f"α={ALPHA:.2f} • {self._model_label(settings.model)}"
            await self._run_search(
                interaction,
                query,
                shows,
                settings,
                ALPHA,
                label,
                source=settings.default_source or None,
                compact=settings.compact,
            )

        # /search-advanced ────────────────────
        @self.tree.command(
            name="search-advanced",
            description="Search with full control over retrieval tuning",
        )
        @app_commands.describe(
            query="What are you looking for?",
            show="Pick a show (searches all if empty)",
            episode="Pick an episode",
            speaker="Filter by speaker name",
            source="Transcript version: corrected, transcript, or a language code",
            after="Oldest publication date to include, YYYY-MM-DD (inclusive)",
            before="Newest publication date to include, YYYY-MM-DD (inclusive)",
            alpha="0 = keywords only → 1 = meaning only (default 0.5)",
            model="Embedding model (leave empty for server default)",
            chunker="Chunking strategy (leave empty for server default)",
            top_k="How many results to show (leave empty for server default)",
            compact="Show results in a single compact embed",
        )
        @app_commands.choices(compact=_BOOL_CHOICES)
        async def search_advanced(
            interaction: discord.Interaction,
            query: str,
            show: str = "",
            episode: str = "",
            speaker: str = "",
            source: str = "",
            after: str = "",
            before: str = "",
            alpha: app_commands.Range[float, 0.0, 1.0] = ALPHA,
            model: str = "",
            chunker: str = "",
            top_k: app_commands.Range[int, 1, 25] = 0,
            compact: str = "",
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            if await self._reject_bad_date(interaction, after, before):
                return
            await self._refresh_if_stale()
            settings = self._effective_settings(
                interaction.guild_id, model, top_k, chunker
            )
            effective_source = source or settings.default_source or None
            use_compact = compact == "true" if compact else settings.compact
            shows = self._resolve_shows(settings, show)
            label = f"α={alpha:.2f} • {self._model_label(settings.model)}"
            await self._run_search(
                interaction,
                query,
                shows,
                settings,
                alpha,
                label,
                source=effective_source,
                episode=episode or None,
                speaker=speaker or None,
                pub_date_min=after or None,
                pub_date_max=before or None,
                compact=use_compact,
            )

        search_advanced.autocomplete("show")(self._show_autocomplete)
        search_advanced.autocomplete("episode")(self._episode_autocomplete)
        search_advanced.autocomplete("source")(self._source_autocomplete)
        search_advanced.autocomplete("speaker")(self._speaker_autocomplete)
        search_advanced.autocomplete("model")(self._model_autocomplete)
        search_advanced.autocomplete("chunker")(self._chunker_autocomplete)

        # /exact ──────────────────────────────
        @self.tree.command(
            name="exact",
            description="Literal substring search — case-insensitive, like Ctrl+F",
        )
        @app_commands.describe(query="Text to find (not case-sensitive)")
        async def exact(
            interaction: discord.Interaction,
            query: str,
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            await self._refresh_if_stale()
            settings = self._server_settings(interaction.guild_id)
            shows = self._resolve_shows(settings, "")
            await self._run_exact(interaction, query, shows)

        # /exact-advanced ─────────────────────
        @self.tree.command(
            name="exact-advanced",
            description="Literal substring search with source and date filters",
        )
        @app_commands.describe(
            query="Text to find (not case-sensitive)",
            show="Pick a show (searches all if empty)",
            episode="Pick an episode",
            speaker="Filter by speaker name",
            source="Transcript version: corrected, transcript, or a language code",
            after="Oldest publication date to include, YYYY-MM-DD (inclusive)",
            before="Newest publication date to include, YYYY-MM-DD (inclusive)",
        )
        async def exact_advanced(
            interaction: discord.Interaction,
            query: str,
            show: str = "",
            episode: str = "",
            speaker: str = "",
            source: str = "",
            after: str = "",
            before: str = "",
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            if await self._reject_bad_date(interaction, after, before):
                return
            await self._refresh_if_stale()
            settings = self._server_settings(interaction.guild_id)
            effective_source = source or settings.default_source or None
            shows = self._resolve_shows(settings, show)
            await self._run_exact(
                interaction,
                query,
                shows,
                source=effective_source,
                episode=episode or None,
                speaker=speaker or None,
                pub_date_min=after or None,
                pub_date_max=before or None,
            )

        exact_advanced.autocomplete("show")(self._show_autocomplete)
        exact_advanced.autocomplete("episode")(self._episode_autocomplete)
        exact_advanced.autocomplete("source")(self._source_autocomplete)
        exact_advanced.autocomplete("speaker")(self._speaker_autocomplete)

        # /random ─────────────────────────────
        @self.tree.command(
            name="random",
            description="Pull a random quote from the transcripts",
        )
        async def random_cmd(interaction: discord.Interaction) -> None:
            if not await self._check_cooldown(interaction):
                return
            await self._refresh_if_stale()
            settings = self._server_settings(interaction.guild_id)
            shows = self._resolve_shows(settings, "")
            await self._run_random(interaction, shows)

        # /random-advanced ────────────────────
        @self.tree.command(
            name="random-advanced",
            description="Pull a random quote with source and date filters",
        )
        @app_commands.describe(
            show="Pick a show (random from all if empty)",
            episode="Pick an episode",
            speaker="Filter by speaker name",
            source="Transcript version: corrected, transcript, or a language code",
            after="Oldest publication date to include, YYYY-MM-DD (inclusive)",
            before="Newest publication date to include, YYYY-MM-DD (inclusive)",
        )
        async def random_advanced(
            interaction: discord.Interaction,
            show: str = "",
            episode: str = "",
            speaker: str = "",
            source: str = "",
            after: str = "",
            before: str = "",
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            if await self._reject_bad_date(interaction, after, before):
                return
            await self._refresh_if_stale()
            settings = self._server_settings(interaction.guild_id)
            effective_source = source or settings.default_source or None
            shows = self._resolve_shows(settings, show)
            await self._run_random(
                interaction,
                shows,
                source=effective_source,
                episode=episode or None,
                speaker=speaker or None,
                pub_date_min=after or None,
                pub_date_max=before or None,
            )

        random_advanced.autocomplete("show")(self._show_autocomplete)
        random_advanced.autocomplete("episode")(self._episode_autocomplete)
        random_advanced.autocomplete("source")(self._source_autocomplete)
        random_advanced.autocomplete("speaker")(self._speaker_autocomplete)

        # /stats ──────────────────────────────
        @self.tree.command(
            name="stats",
            description="Index overview: shows, episodes, duration",
        )
        @app_commands.describe(
            show="Pick a show (shows all if empty)",
            model="Search model (leave empty for server default)",
        )
        async def stats(
            interaction: discord.Interaction,
            show: str = "",
            model: str = "",
        ) -> None:
            await self._handle_stats(interaction, show or None, model or None)

        stats.autocomplete("show")(self._show_autocomplete)
        stats.autocomplete("model")(self._model_autocomplete)

        # /episodes ───────────────────────────
        @self.tree.command(
            name="episodes",
            description="List episodes for a show with excerpt count and duration",
        )
        @app_commands.describe(
            show="Pick a show (auto-selected if only one)",
            model="Search model (leave empty for server default)",
        )
        async def episodes(
            interaction: discord.Interaction,
            show: str = "",
            model: str = "",
        ) -> None:
            await self._handle_episodes(interaction, show or None, model or None)

        episodes.autocomplete("show")(self._show_autocomplete)
        episodes.autocomplete("model")(self._model_autocomplete)

        # /speakers ───────────────────────────
        @self.tree.command(
            name="speakers",
            description="Who speaks the most — excerpt count and airtime per speaker",
        )
        @app_commands.describe(
            show="Pick a show (aggregates all if empty)",
            model="Search model (leave empty for server default)",
        )
        async def speakers(
            interaction: discord.Interaction,
            show: str = "",
            model: str = "",
        ) -> None:
            await self._handle_speakers(interaction, show or None, model or None)

        speakers.autocomplete("show")(self._show_autocomplete)
        speakers.autocomplete("model")(self._model_autocomplete)

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
        )
        @app_commands.choices(
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
        setup.autocomplete("model")(self._model_autocomplete)
        setup.autocomplete("chunker")(self._chunker_autocomplete)

        # /announcements ──────────────────────
        @self.tree.command(
            name="announcements",
            description="Set the channel for new-episode and version updates (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(
            channel="Channel to post announcements in",
            off="Turn announcements off for this server",
        )
        @app_commands.choices(off=_BOOL_CHOICES)
        async def announcements(
            interaction: discord.Interaction,
            channel: discord.TextChannel | None = None,
            off: str = "",
        ) -> None:
            await self._handle_announcements(interaction, channel, off == "true")

        # /sync ───────────────────────────────
        @self.tree.command(
            name="sync",
            description="Manually sync slash commands (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        async def sync(interaction: discord.Interaction) -> None:
            await interaction.response.defer(ephemeral=True)
            await self.tree.sync()
            await interaction.followup.send(
                "✅ Command tree synced. New or renamed commands may take up to "
                "an hour to appear in this server (Discord cache).",
                ephemeral=True,
            )

        # /admin-reload ───────────────────────
        @self.tree.command(
            name="admin-reload",
            description="Reconnect to the index and reload show passwords (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        async def admin_reload(interaction: discord.Interaction) -> None:
            await self._handle_admin_reload(interaction)

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
                value=(
                    "Overview of what's indexed: episodes and duration per "
                    "show. A single show (one indexed, or picked via the "
                    "show filter) also gets its top speakers."
                ),
                inline=False,
            )
            embed.add_field(
                name="/speakers",
                value="Who speaks the most — excerpt counts and airtime per speaker.",
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
            embed.add_field(
                name="/announcements *(admin)*",
                value=(
                    "Pick a channel where the bot posts new episodes and version "
                    "updates. Pass `off:True` to disable."
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
        query: str,
        shows: ResolvedShows,
        settings: ServerSettings,
        alpha: float,
        label: str,
        *,
        source: str | None = None,
        episode: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
        compact: bool = False,
    ) -> None:
        await interaction.response.defer()
        loop = asyncio.get_running_loop()

        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._hybrid_search(
                    query,
                    shows,
                    settings,
                    alpha,
                    source=source,
                    episode=episode,
                    speaker=speaker,
                    pub_date_min=pub_date_min,
                    pub_date_max=pub_date_max,
                ),
            )
        except ValueError as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        except Exception:
            logger.exception(f"Search error: {query!r}")
            await interaction.followup.send(
                "❌ Search failed — please try again in a moment.",
                ephemeral=True,
            )
            return

        if not results:
            suffix = format_filter_suffix(
                episode=episode, speaker=speaker, source=source
            )
            await interaction.followup.send(
                f'No results for **"{query}"**{suffix}.\n'
                "Try simpler wording, drop filters, or use `/exact` for literal matches.",
                ephemeral=True,
            )
            return

        await self._send_results(
            "search", label, query, results, interaction, prefer_list=compact
        )

    async def _send_results(
        self,
        kind: str,
        label: str,
        query: str,
        results: list[tuple[dict, str]],
        interaction: discord.Interaction,
        *,
        prefer_list: bool = False,
    ) -> None:
        """Cache result refs and post the first persistent page.

        ``prefer_list`` opens on the compact list instead of the paged card
        (the /exact default, or /search under the server's compact setting);
        either view can flip to the other via its toggle button.
        """
        refs = [_chunk_to_ref(chunk, col) for chunk, col in results]
        sid = self.results.save(CachedSearch(kind, label, query, refs))
        built = (
            await build_compact_view(self, sid)
            if prefer_list
            else await build_results_view(self, sid, 0)
        )
        if built is None:  # episode vanished between search and render
            await interaction.followup.send(
                "❌ Results could not be loaded — please try again.", ephemeral=True
            )
            return
        embed, view = built
        await interaction.followup.send(embed=embed, view=view)

    def _hybrid_search(
        self,
        query: str,
        shows: ResolvedShows,
        settings: ServerSettings,
        alpha: float,
        *,
        source: str | None = None,
        episode: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
    ) -> list[tuple[dict, str]]:
        """Run hybrid retrieval, one collection per show, and merge results.

        Shows may resolve to collections under different embedding models, so
        retrievers are grouped by model (each encodes the query once).
        """
        col_info = self.local.get_all_collection_info()
        pairs = self._resolve_show_collections(shows, settings, col_info)
        if not pairs:
            logger.warning("No collections resolved for this query")
            return []

        by_model: dict[str, list[str]] = defaultdict(list)
        for col, model in pairs:
            by_model[model].append(col)

        hits_by_col: dict[str, list[dict]] = {}
        for model, cols in by_model.items():
            ret = self.retriever(model)
            for col in cols:
                hits = ret.retrieve(
                    query,
                    col,
                    top_k=settings.top_k,
                    alpha=alpha,
                    source=source,
                    episode=episode,
                    speaker=speaker,
                    pub_date_min=pub_date_min,
                    pub_date_max=pub_date_max,
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
        shows: ResolvedShows,
        *,
        source: str | None = None,
        episode: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
    ) -> None:
        await interaction.response.defer()
        settings = self._server_settings(interaction.guild_id)
        loop = asyncio.get_running_loop()

        try:
            col_info = await self._cached_col_info()
            pairs = self._resolve_show_collections(shows, settings, col_info)
            if not pairs:
                await interaction.followup.send(
                    self._empty_collections_message(col_info, settings, shows),
                    ephemeral=True,
                )
                return

            all_results: list[tuple[dict, str]] = []
            for col, model in pairs:
                hits = await loop.run_in_executor(
                    None,
                    lambda c=col, m=model: self.retriever(m).exact(
                        query,
                        c,
                        source=source,
                        episode=episode,
                        speaker=speaker,
                        pub_date_min=pub_date_min,
                        pub_date_max=pub_date_max,
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
        all_results = phrase + fuzzy

        total_mentions = sum(
            count_occurrences(c.get("text", ""), query) for c, _ in all_results
        )
        # Fuzzy-only hits contain no literal occurrence; never show "0 matches"
        # above a non-empty result list.
        n = total_mentions or len(all_results)
        label = f"{n} match{'es' if n != 1 else ''}"
        # /exact is survey-shaped (many positional hits): open on the list.
        await self._send_results(
            "exact", label, query, all_results, interaction, prefer_list=True
        )

    # ── /random handler ───────────────────────

    async def _run_random(
        self,
        interaction: discord.Interaction,
        shows: ResolvedShows,
        *,
        source: str | None = None,
        episode: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
    ) -> None:
        await interaction.response.defer()
        settings = self._server_settings(interaction.guild_id)
        loop = asyncio.get_running_loop()

        try:
            col_info = await self._cached_col_info()
            pairs = self._resolve_show_collections(shows, settings, col_info)
            if not pairs:
                await interaction.followup.send(
                    self._empty_collections_message(col_info, settings, shows),
                    ephemeral=True,
                )
                return

            col, model = random.choice(pairs)
            retriever = self.retriever(model)
            chunk = await loop.run_in_executor(
                None,
                lambda: retriever.random(
                    col,
                    episode=episode,
                    source=source,
                    speaker=speaker,
                    pub_date_min=pub_date_min,
                    pub_date_max=pub_date_max,
                ),
            )
        except Exception:
            logger.exception("Random quote error")
            await interaction.followup.send(
                "❌ Could not fetch a random quote.", ephemeral=True
            )
            return

        if chunk is None:
            suffix = format_filter_suffix(
                episode=episode, speaker=speaker, source=source
            )
            await interaction.followup.send(
                f"No excerpts found{suffix}. Try without filters.",
                ephemeral=True,
            )
            return

        show = chunk.get("show", "")
        ep_display = episode_display(chunk)
        spk = display_speaker(chunk.get("speaker") or chunk.get("dominant_speaker"))
        start = chunk.get("start", 0.0)
        end = chunk.get("end", 0.0)
        text = chunk.get("text", "")

        embed = discord.Embed(
            description=f'"{text}"',
            color=discord.Color.blurple(),
        )
        if show:
            embed.set_author(name=show)
        embed.title = ep_display or "(untitled)"
        set_chunk_thumbnail(embed, chunk)
        embed.add_field(name="Speaker", value=spk, inline=True)
        timed = chunk.get("timed", True)
        ts_label = fmt_timestamp(start, end, timed=timed)
        if ts_label:
            embed.add_field(name="Timestamp", value=ts_label, inline=True)
        embed.set_footer(text="🎲 Random quote")

        # Cache a one-result search so the "Show context" button is persistent
        # and survives restarts like the /search and /exact ones.
        sid = self.results.save(
            CachedSearch("random", "", "", [_chunk_to_ref(chunk, col)])
        )
        view = discord.ui.View(timeout=None)
        listen = build_listen_button(chunk)
        if listen is not None:
            view.add_item(listen)
        view.add_item(ExpandResult(sid, 0))
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

        updated = replace(
            current,
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

        shows_str = (
            ", ".join(f"`{s}`" for s in updated.allowed_shows) or "*(all public)*"
        )
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

    # ── /announcements handler ────────────────

    async def _handle_announcements(
        self,
        interaction: discord.Interaction,
        channel: "discord.TextChannel | None",
        off: bool,
    ) -> None:
        guild_id = interaction.guild_id
        if guild_id is None:
            await interaction.response.send_message(
                "Use this command in a server.", ephemeral=True
            )
            return
        settings = self._server_cfg.get(guild_id) or self._server_settings(guild_id)

        if off:
            settings.announce_channel_id = 0
            self._server_cfg[guild_id] = settings
            self._save_server_config()
            await interaction.response.send_message(
                "🔕 Announcements are off for this server.", ephemeral=True
            )
            return

        if channel is not None:
            settings.announce_channel_id = channel.id
            self._server_cfg[guild_id] = settings
            self._save_server_config()
            await interaction.response.send_message(
                f"📣 New episodes and version updates will post in {channel.mention}.",
                ephemeral=True,
            )
            return

        # No args: report current state.
        if settings.announce_channel_id:
            await interaction.response.send_message(
                f"📣 Announcements post in <#{settings.announce_channel_id}>. "
                "Pass `off:True` to disable.",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "🔕 Announcements are off. Pass a `channel` to enable them.",
                ephemeral=True,
            )

    # ── /admin-reload handler ─────────────────

    async def _handle_admin_reload(self, interaction: discord.Interaction) -> None:
        """Force-refresh bot state against the current index on disk.

        The auto-refresh in every privileged handler already picks up
        external changes via mtime, but this gives admins an explicit
        escape hatch if mtime detection is defeated (e.g. ``cp -p`` that
        preserves timestamps, a network mount with coarse mtime, etc.).
        """
        await interaction.response.defer(ephemeral=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.local.reconnect)
        self._ac_cache.reset()
        await loop.run_in_executor(None, self._reload_shows)
        self._index_mtime_seen = await loop.run_in_executor(
            None, self.local.index_mtime
        )
        self._last_mtime_check = time.monotonic()
        col_info = await loop.run_in_executor(None, self.local.get_all_collection_info)
        shows = sorted(
            {info.get("show", "") for info in col_info.values() if info.get("show")}
        )
        protected = len(self._shows)
        await interaction.followup.send(
            f"✅ Reloaded. {len(col_info)} collection(s), {len(shows)} show(s), "
            f"{protected} password-protected.",
            ephemeral=True,
        )

    # ── /unlock + /lock handlers ────────────────

    async def _handle_unlock(
        self,
        interaction: discord.Interaction,
        password: str,
    ) -> None:
        # Refresh from disk so passwords set via the desktop app (while the
        # bot is already running) are picked up without a restart. The
        # staleness check also reconnects LanceDB if the index was rsynced
        # under the running process.
        await self._refresh_if_stale()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._reload_shows)

        # Strip: Discord mobile clients sometimes append a space on paste.
        password = password.strip()

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
            logger.warning(
                f"Failed unlock attempt in guild {interaction.guild_id} "
                f"(bot sees {len(self._shows)} protected show(s): "
                f"{sorted(e.name for e in self._shows.values())})"
            )
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
        await self._refresh_if_stale()
        settings = self._effective_settings(interaction.guild_id, model or "", 0)
        loop = asyncio.get_running_loop()

        try:
            col_info = await self._cached_col_info()
            resolved = (
                self._resolve_shows(settings, show)
                if show
                else ResolvedShows(ShowAccess.ALL)
            )
            collections = [
                c
                for c, _ in self._resolve_show_collections(resolved, settings, col_info)
            ]
            if not collections:
                stats_shows = resolved if show else None
                await interaction.followup.send(
                    self._empty_collections_message(col_info, settings, stats_shows),
                    ephemeral=True,
                )
                return

            per_show: dict[str, list[dict]] = {}
            for col in collections:
                stats = await loop.run_in_executor(
                    None,
                    lambda c=col: self.local.get_episode_stats(c),
                )
                name = col_info.get(col, {}).get("show") or col
                per_show.setdefault(name, []).extend(stats)

            # Speaker detail only for a single-show scope; the global
            # overview stays a per-show table (mixing speakers across
            # shows is /speakers' job).
            speakers = (
                await loop.run_in_executor(
                    None, self.local.speaker_stats_multi, collections
                )
                if len(collections) == 1
                else []
            )

        except Exception:
            logger.exception("Stats error")
            await interaction.followup.send(
                "❌ Could not retrieve stats.", ephemeral=True
            )
            return

        # Artwork only when the scope is a single show (mirrors /episodes).
        artwork = (
            col_info.get(collections[0], {}).get("artwork_url", "")
            if len(collections) == 1
            else ""
        )
        embed = build_stats_embed(per_show, speakers, artwork_url=artwork)
        await interaction.followup.send(embed=embed)

    # ── /speakers handler ─────────────────────

    async def _handle_speakers(
        self,
        interaction: discord.Interaction,
        show: str | None,
        model: str | None,
    ) -> None:
        await interaction.response.defer()
        await self._refresh_if_stale()
        settings = self._effective_settings(interaction.guild_id, model or "", 0)
        loop = asyncio.get_running_loop()

        try:
            col_info = await self._cached_col_info()
            resolved = (
                self._resolve_shows(settings, show)
                if show
                else ResolvedShows(ShowAccess.ALL)
            )
            collections = [
                c
                for c, _ in self._resolve_show_collections(resolved, settings, col_info)
            ]
            if not collections:
                speaker_shows = resolved if show else None
                await interaction.followup.send(
                    self._empty_collections_message(col_info, settings, speaker_shows),
                    ephemeral=True,
                )
                return

            ranked = await loop.run_in_executor(
                None, self.local.speaker_stats_multi, collections
            )

        except Exception:
            logger.exception("Speakers error")
            await interaction.followup.send(
                "❌ Could not retrieve speaker stats.", ephemeral=True
            )
            return

        if not ranked:
            await interaction.followup.send(
                "No speaker attribution found in these transcripts.",
                ephemeral=True,
            )
            return

        top = ranked[:15]
        total_duration = sum(r["total_duration"] for r in ranked)

        lines = []
        for i, r in enumerate(top, start=1):
            share = (
                (r["total_duration"] / total_duration * 100) if total_duration else 0
            )
            lines.append(
                f"`{i:>2}.` **{display_speaker(r['speaker'])}** — `{fmt_time(r['total_duration'])}` "
                f"({share:.0f}%) · {r['chunk_count']} excerpt{'s' if r['chunk_count'] != 1 else ''} "
                f"· {r['episodes']} episode{'s' if r['episodes'] != 1 else ''}"
            )

        scope = show or f"{len(collections)} show{'s' if len(collections) > 1 else ''}"
        embed = discord.Embed(
            title=f"🎙 Speakers — {scope}",
            description="\n".join(lines),
            color=discord.Color.blurple(),
        )
        if len(ranked) > len(top):
            embed.set_footer(text=f"Showing top {len(top)} of {len(ranked)}")

        await interaction.followup.send(embed=embed)

    # ── /episodes handler ─────────────────────

    async def _handle_episodes(
        self,
        interaction: discord.Interaction,
        show: str | None,
        model: str | None,
    ) -> None:
        await interaction.response.defer()
        await self._refresh_if_stale()
        settings = self._effective_settings(interaction.guild_id, model or "", 0)
        loop = asyncio.get_running_loop()

        # Auto-resolve show: explicit > unlocked > single accessible show > ask
        show_auto_resolved = not show
        # Check access control for explicit show
        if show:
            resolved = self._resolve_shows(settings, show)
            if resolved.is_locked:
                col_info = await self._cached_col_info()
                await interaction.followup.send(
                    self._empty_collections_message(col_info, settings, resolved),
                    ephemeral=True,
                )
                return

        if not show:
            if settings.allowed_shows:
                show = settings.allowed_shows[0]
            else:
                col_info = await self._cached_col_info()
                pairs = self._resolve_show_collections(
                    ResolvedShows(ShowAccess.ALL), settings, col_info
                )
                shows = sorted({col_info.get(c, {}).get("show") or c for c, _ in pairs})
                if len(shows) == 1:
                    show = shows[0]
                elif not shows:
                    await interaction.followup.send(
                        self._empty_collections_message(col_info, settings),
                        ephemeral=True,
                    )
                    return
                else:
                    names = "\n".join(f"🎙 {s}" for s in shows)
                    await interaction.followup.send(
                        f"Multiple shows available — please specify one:\n{names}",
                        ephemeral=True,
                    )
                    return

        col_info = await self._cached_col_info()
        ep_pairs = self._resolve_show_collections(
            ResolvedShows(ShowAccess.SPECIFIC, (show,)), settings, col_info
        )
        if not ep_pairs:
            await interaction.followup.send(
                f"No episodes found for **{show}**.", ephemeral=True
            )
            return
        col = ep_pairs[0][0]

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

        footer = f"{len(ep_stats)} episodes"
        if show_auto_resolved:
            footer += f" (auto-selected: {show})"

        show_art = col_info.get(col, {}).get("artwork_url", "")
        embeds = build_episodes_embeds(show, ep_stats, footer, artwork_url=show_art)

        if len(embeds) == 1:
            await interaction.followup.send(embed=embeds[0])
        else:
            # Self-contained pages: cache the rendered embeds verbatim so the
            # nav buttons stay persistent and survive restarts.
            sid = self.results.save(
                CachedSearch("list", "", "", embeds=[e.to_dict() for e in embeds])
            )
            built = await build_list_view(self, sid, 0)
            assert built is not None  # just saved; cannot miss
            embed, view = built
            await interaction.followup.send(embed=embed, view=view)


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
    from podcodex.bootstrap import bootstrap_for_dev

    bootstrap_for_dev()

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(prog="podcodex-bot")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()))
    parser.add_argument(
        "--chunking", default=DEFAULT_CHUNKING, choices=list(CHUNKING_STRATEGIES.keys())
    )
    parser.add_argument("--top-k", default=TOP_K, type=int)
    parser.add_argument(
        "--index",
        default=None,
        help="Path to LanceDB index directory (default: <data_dir>/index)",
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
        "--announce-interval",
        default=10,
        type=int,
        help="Minutes between checks for new episodes to announce",
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
        announce_interval_minutes=args.announce_interval,
    )

    bot = PodCodexBot(config, server_config_path=Path(args.server_config))
    logger.info(f"Starting PodCodex bot (model={config.model}, top_k={config.top_k})")
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
