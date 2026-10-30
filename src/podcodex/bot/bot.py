"""
podcodex.bot.bot — Discord bot for podcast transcript search.

Entrypoint (``podcodex-bot --help`` for every flag):
    podcodex-bot [--model bge-m3] [--models bge-m3,e5-small] [--chunking semantic]
                 [--top-k 5] [--index PATH] [--server-config FILE] [--dev-guild ID]
    podcodex-bot --manage-passwords [--index PATH]
    podcodex-bot --claim-index [--index PATH]

Environment: DISCORD_TOKEN (required), PODCODEX_INDEX, PODCODEX_ALLOWED_GUILDS.


Slash commands. The registered tree is in ``registration.py``; that file is
the source of truth, and this list exists so an operator reading the entry
point sees the real surface.

  Everyone:
    /search   query
    /exact    query
    /random
              The three plain forms take no options beyond the query and use
              this server's defaults.
    /search-advanced query [show] [episode] [speaker] [source] [after]
              [before] [alpha] [model] [chunker] [top_k] [compact]
    /exact-advanced  query [show] [episode] [speaker] [source] [after]
              [before]
    /random-advanced [show] [episode] [speaker] [source] [after] [before]
              The -advanced forms carry the filters; /search-advanced also
              the retrieval tuning.
    /stats    [show] [model]
    /episodes [show] [model]
    /speakers [show] [model]
    /help     Show available commands and how to use them.

  Admin (manage_guild):
    /setup    [model] [chunker] [top_k] [show_add] [show_remove] [show_clear]
              [default_source] [compact]
              Server defaults, including the pinned default shows.
    /announcements [channel] [off]
              Channel for new-episode and version announcements.
    /unlock         password
              Unlock a password-protected show here (the password identifies
              the show, so no name is ever exposed).
    /lock           show
              Revoke a show unlocked here.
    /changepassword show
              Rotate the password for an unlocked show; DMs the new one.
    /admin-reload   Reconnect to the index and reload show passwords.
    /sync           Manually sync the command tree.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import shutil
import time
from pathlib import Path

import discord
from discord import app_commands
from loguru import logger

from podcodex.bot.access import (  # noqa: F401 (ResolvedShows/ShowAccess re-exported)
    AccessMixin,
    ResolvedShows,
    ShowAccess,
    ShowEntry,
)
from podcodex.bot.announce import AnnounceStore
from podcodex.bot.announce_tasks import AnnounceMixin
from podcodex.bot.autocomplete import AutocompleteMixin, _AutocompleteCache
from podcodex.bot.config import BotConfig, ServerSettings
from podcodex.core.app_paths import data_dir
from podcodex.core.show_passwords import hash_show_password
from podcodex.bot.registration import RegistrationMixin
from podcodex.bot.resolution import ResolutionMixin
from podcodex.bot.search_commands import SearchCommandsMixin
from podcodex.bot.settings import SettingsMixin
from podcodex.bot.stats_commands import StatsCommandsMixin
from podcodex.bot.formatting import CooldownManager
from podcodex.bot.result_store import SearchCacheStore
from podcodex.bot.ui import DYNAMIC_ITEMS, clear_chunk_cache
from podcodex.rag.defaults import (
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING,
    DEFAULT_MODEL,
    MODELS,
    TOP_K,
)
from podcodex.rag.index_store import IndexStore, get_index_store
from podcodex.rag.retriever import Retriever, get_retriever

# Throttle for the per-call mtime check. Discord fires autocomplete on every
# keystroke; without throttling, a 10-char query would walk the index dir 10
# times. 2s is well below any realistic rate of out-of-process index changes.
_MTIME_CHECK_INTERVAL = 2.0


# ── Bot ───────────────────────────────────────


class _GuildGatedTree(app_commands.CommandTree):
    """Command tree that refuses commands outside the allowed guilds."""

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self.client.guild_allowed(interaction.guild_id):
            return True
        if interaction.type is discord.InteractionType.autocomplete:
            return False  # cannot carry a message; no suggestions is the answer
        await interaction.response.send_message(
            "This bot is not available here.", ephemeral=True
        )
        return False


class PodCodexBot(
    AccessMixin,
    SettingsMixin,
    ResolutionMixin,
    AutocompleteMixin,
    SearchCommandsMixin,
    StatsCommandsMixin,
    AnnounceMixin,
    RegistrationMixin,
    discord.Client,
):
    """Discord client with slash commands for searching podcast transcripts."""

    def __init__(self, config: BotConfig, server_config_path: Path) -> None:
        super().__init__(intents=discord.Intents.default())
        self.config = config
        self.server_config_path = server_config_path
        self.tree = _GuildGatedTree(self)
        self.tree.on_error = self._on_app_command_error
        self._cooldown = CooldownManager(seconds=config.cooldown_seconds)

        # Password-protected shows from IndexStore: {lower(show id): ShowEntry}.
        # Populated in setup_hook once the index is open; /admin-reload and
        # every external index change reload it.
        self._shows: dict[str, ShowEntry] = {}

        self._local: IndexStore | None = None
        self._retrievers: dict[str, Retriever] = {}
        self._server_cfg: dict[int, ServerSettings] = self._load_server_config()
        # Durable + in-RAM cache backing the persistent pagination buttons.
        self.results = SearchCacheStore(server_config_path.parent / "search_cache.db")
        # Diff state for new-episode + version announcements.
        self.announce = AnnounceStore(server_config_path.parent / "announce_state.db")
        self._ac_cache = _AutocompleteCache()

        # Newest mtime seen across index-dir entries. Rising value means the
        # on-disk index changed (rsync, password set via API, new show
        # indexed) and bot state must be reloaded.
        self._index_mtime_seen: float = 0.0
        self._last_mtime_check: float = 0.0
        # The announcer keeps its OWN mtime watermark, separate from
        # ``_index_mtime_seen`` — otherwise a user command's refresh would
        # advance the shared value and the loop would never see the change.
        self._announce_mtime_seen: float = 0.0
        # Collections retried after a transient failure: {name: tries}.
        self._announce_retry: dict[str, int] = {}
        # Episodes waiting for stats rows: {name: (stems, index changes seen)}.
        self._announce_pending: dict[str, tuple[frozenset[str], int]] = {}
        # Announcement channels already reported unreachable.
        self._announce_unreachable: set[int] = set()

        self._register_commands()

    async def _refresh_if_stale(self) -> bool:
        """Detect external index changes and reload bot state. True if it did.

        Called at the top of every privileged handler so out-of-process
        writes (rsync, desktop-app indexing, API password change) can't
        leak ACL state. Filesystem checks are throttled to one sweep per
        ``_MTIME_CHECK_INTERVAL`` so autocomplete bursts don't compound.
        """
        now = time.monotonic()
        if now - self._last_mtime_check < _MTIME_CHECK_INTERVAL:
            return False
        self._last_mtime_check = now
        loop = asyncio.get_running_loop()
        current = await loop.run_in_executor(None, self.local.index_mtime)
        # Inequality, not ">": the value can move backwards. ``rsync -a``
        # stamps destination mtimes from the source, so syncing an older
        # dev-machine copy over a table last written on the bot host (a
        # password change, say) lowers it. Any movement means a change.
        if current == self._index_mtime_seen:
            return False
        await loop.run_in_executor(None, self.local.reconnect)
        self._ac_cache.reset()
        clear_chunk_cache()
        await self._reload_access()
        # Advanced only once the reload succeeded. A sweep landing mid-rsync
        # can open a manifest whose data files haven't arrived; advancing
        # first would mark that torn state as seen and never retry it.
        self._index_mtime_seen = current
        logger.info("Index refresh: external change detected, bot state reloaded.")
        return True

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
        await self._reload_access()
        self._index_mtime_seen = await loop.run_in_executor(
            None, self.local.index_mtime
        )
        loaded = self._loadable_models()
        stray = sorted({s.model for s in self._server_cfg.values()} - loaded)
        if stray:
            logger.warning(
                f"Server default model(s) {stray} are not loaded (loading "
                f"{sorted(loaded)}); /search on those servers uses the loaded "
                "ones. Pass --models to keep one."
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

    def guild_allowed(self, guild_id: int | None) -> bool:
        """Whether this bot serves a guild (``None``: a DM).

        Unrestricted unless PODCODEX_ALLOWED_GUILDS is set. Then only the
        listed guilds: a bot left public in the developer portal can be
        invited by anyone who has its id, and would serve every unprotected
        show to their server.
        """
        allowed = self.config.allowed_guild_ids
        return allowed is None or guild_id in allowed

    async def _leave_if_not_allowed(self, guild: discord.Guild) -> None:
        if self.guild_allowed(guild.id):
            return
        logger.warning(
            f"Leaving guild {guild.id} ({guild.name}): not in PODCODEX_ALLOWED_GUILDS"
        )
        try:
            await guild.leave()
        except discord.HTTPException:
            logger.exception(f"Could not leave guild {guild.id}")

    async def on_guild_join(self, guild: discord.Guild) -> None:
        await self._leave_if_not_allowed(guild)

    async def on_ready(self) -> None:
        cmds = [c.name for c in self.tree.get_commands()]
        logger.success(
            f"Logged in as {self.user} (id={self.user.id}) — commands: {cmds}"
        )
        for guild in list(self.guilds):
            await self._leave_if_not_allowed(guild)
        await self._announce_version_if_changed()

    # ── Admin reload ──────────────────────────

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
        clear_chunk_cache()
        await self._reload_access()
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


def _manage_passwords_cli(index_path: str | None) -> None:
    """Interactive CLI to manage show passwords stored in the IndexStore.

    Lists all indexed shows, shows their current password status, and
    lets the operator set or auto-generate a password for each show.
    Auto-generated passwords use 16 URL-safe random bytes (22 chars).
    """
    import getpass

    from podcodex.rag.index_store import IndexStore

    store = IndexStore(index_path)
    from podcodex.rag.index_origin import is_replica, read_origin

    if is_replica(store.path):
        print(
            f"\nThis index is a replica: it is owned by machine "
            f"{read_origin(store.path)!r}, not this one.\n"
            "Passwords set here would be destroyed by the next rsync from the owner.\n"
            "Set them on the owning machine (the PodCodex app's bot access screen),\n"
            "or re-run with --claim-index to take ownership here."
        )
        return
    col_info = store.get_all_collection_info()
    if not col_info:
        print("No indexed shows found.")
        return

    # Display name -> stable id. The CLI speaks names because that is what an
    # operator knows; the password table is keyed by id.
    keys: dict[str, str] = {}
    for col_name, info in col_info.items():
        keys.setdefault(info.get("show") or col_name, store.show_key(info))
    show_names = sorted(keys)
    existing = store.get_show_passwords()

    print(f"\nFound {len(show_names)} show(s):\n")
    for name in show_names:
        protected = keys[name] in existing
        status = "🔒 password set" if protected else "🔓 no password (public)"
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
            store.set_show_password(
                keys[name], hash_show_password(password), show_label=name
            )
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
            store.set_show_password(
                keys[name], hash_show_password(password), show_label=name
            )
            print(f"  Password set for '{name}'.")
        elif choice == "r":
            store.delete_show_password(keys[name])
            print(f"  Password removed — '{name}' is now public.")
        else:
            print("  Skipped.")


def _claim_index_cli(index_path: str | None) -> None:
    """Take ownership of the index on this machine, then exit.

    For the case where an index was copied here once and is now maintained
    here. Syncing again from the previous owner overwrites it, passwords
    included, so the previous owner is named explicitly.
    """
    from podcodex.rag.index_origin import claim_origin
    from podcodex.rag.index_store import IndexStore

    store = IndexStore(index_path)
    previous = claim_origin(store.path)
    print(
        f"Index at {store.path} is now owned by this machine "
        f"(previous owner: {previous or 'none'})."
    )
    if previous:
        print(
            "Syncing this index again from the previous owner will overwrite "
            "it, including any passwords set here."
        )


# Per-guild state written by the bot: /setup defaults, unlocked shows,
# announcement channels, the durable search cache and the announce baseline.
_STATE_FILES = ("server_config.json", "search_cache.db", "announce_state.db")


def _checkpoint_sqlite(path: Path) -> None:
    """Fold a WAL-mode database's ``-wal`` into its main file.

    Closing the last connection after a full checkpoint also removes the
    ``-wal`` and ``-shm`` files, so the database is one file again.
    """
    import sqlite3

    try:
        conn = sqlite3.connect(path)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning(f"Could not checkpoint {path.name} before moving it: {exc}")


def _resolve_state_path(override: str | None) -> Path:
    """Return the server-config path, defaulting to ``<data_dir>/bot/``.

    The default used to be the relative ``server_config.json``, which put
    every guild's settings in the working directory. Under Docker that is
    ``/app`` in the container layer, so ``docker compose up -d --build``
    (the documented update command) silently dropped every server's
    ``/setup`` defaults, unlocked shows and announce channel. ``data_dir()``
    resolves to the ``podcodex_data`` volume there, and to the platform
    app-data dir everywhere else.

    A bare-metal deploy started before this change keeps its state: the
    legacy files in the working directory are moved into the new location
    once, on the first run that finds them. ``shutil.move``, not ``replace``,
    because the two can sit on different filesystems (a container layer and
    a mounted volume). Both databases run in WAL mode, so committed rows not
    yet checkpointed live only in the ``-wal`` file: each is checkpointed
    into its main file before the move, and nothing is left behind to pair
    with a fresh database later. ``server_config.json`` moves last, since
    its presence at the target is what marks the migration done: a run that
    failed partway retries on the next start.
    """
    if override:
        return Path(override)
    target_dir = data_dir() / "bot"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "server_config.json"
    legacy = Path.cwd() / "server_config.json"
    if not target.exists() and legacy.is_file():
        try:
            for filename in sorted(_STATE_FILES, key=lambda f: f == target.name):
                source = Path.cwd() / filename
                if source.is_file() and not (target_dir / filename).exists():
                    if filename.endswith(".db"):
                        _checkpoint_sqlite(source)
                    shutil.move(str(source), str(target_dir / filename))
            logger.info(f"Migrated bot state from {Path.cwd()} to {target_dir}")
        except OSError as exc:
            logger.warning(f"Could not migrate bot state to {target_dir}: {exc}")
    return target


def _allowed_guilds_from_env() -> frozenset[int] | None:
    """PODCODEX_ALLOWED_GUILDS: comma-separated guild ids, or None if unset."""
    raw = os.environ.get("PODCODEX_ALLOWED_GUILDS", "").strip()
    if not raw:
        return None
    try:
        ids = frozenset(int(part) for part in raw.split(",") if part.strip())
    except ValueError:
        ids = frozenset()
    if not ids:
        # Empty would make the bot leave every server it is in.
        raise SystemExit(
            "PODCODEX_ALLOWED_GUILDS must be comma-separated guild ids "
            "(numbers), e.g. 123456789012345678,234567890123456789."
        )
    return ids


def main() -> None:
    from podcodex.bootstrap import bootstrap_for_bot

    bootstrap_for_bot()

    parser = argparse.ArgumentParser(prog="podcodex-bot")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help="Default embedding model for queries",
    )
    parser.add_argument(
        "--models",
        default="",
        help=(
            "Comma-separated embedding models the bot may load (default: just "
            "--model). Shows indexed only under other models are not searchable."
        ),
    )
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
    parser.add_argument(
        "--server-config",
        default=None,
        help=(
            "Path to server_config.json (default: <data_dir>/bot/server_config.json; "
            "search_cache.db and announce_state.db live beside it)"
        ),
    )
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
        "--claim-index",
        action="store_true",
        help=(
            "Take ownership of the index on this machine and exit. Only needed "
            "when an index copied from another machine is now maintained here; "
            "syncing again from the old owner will overwrite it."
        ),
    )
    parser.add_argument(
        "--manage-passwords",
        action="store_true",
        help="Interactively manage show passwords in the index and exit",
    )
    args = parser.parse_args()

    if args.claim_index:
        _claim_index_cli(args.index)
        return

    if args.manage_passwords:
        _manage_passwords_cli(args.index)
        return

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(
            f"Unknown model(s) in --models: {', '.join(unknown)}. "
            f"Known: {', '.join(MODELS)}."
        )
    if models and args.model not in models:
        models = (args.model, *models)

    token = os.environ.get("DISCORD_TOKEN", "").strip()
    if not token:
        # The most common first-run misconfiguration, so answer it with a
        # sentence rather than a traceback, the way the other entry points do.
        raise SystemExit(
            "DISCORD_TOKEN not set. Export it, or put it in a .env file in the "
            "directory you start podcodex-bot from (Docker reads deploy/.env), "
            "and start podcodex-bot again."
        )

    allowed = _allowed_guilds_from_env()
    if allowed is not None and args.dev_guild and args.dev_guild not in allowed:
        # Commands are synced there; leaving it at on_ready would be absurd.
        logger.warning(f"--dev-guild {args.dev_guild} added to PODCODEX_ALLOWED_GUILDS")
        allowed = allowed | {args.dev_guild}

    config = BotConfig(
        model=args.model,
        chunker=args.chunking,
        top_k=args.top_k,
        index_path=args.index,
        merge_strategy=args.merge_strategy,
        cooldown_seconds=args.cooldown,
        dev_guild_id=args.dev_guild,
        announce_interval_minutes=args.announce_interval,
        models=models,
        allowed_guild_ids=allowed,
    )

    bot = PodCodexBot(
        config, server_config_path=_resolve_state_path(args.server_config)
    )
    logger.info(f"Starting PodCodex bot (model={config.model}, top_k={config.top_k})")
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
