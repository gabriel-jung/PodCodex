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
import os
import secrets
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
from podcodex.core.show_passwords import hash_show_password
from podcodex.bot.registration import RegistrationMixin
from podcodex.bot.resolution import ResolutionMixin
from podcodex.bot.search_commands import SearchCommandsMixin
from podcodex.bot.settings import SettingsMixin
from podcodex.bot.stats_commands import StatsCommandsMixin
from podcodex.bot.formatting import CooldownManager
from podcodex.bot.result_store import SearchCacheStore
from podcodex.bot.ui import DYNAMIC_ITEMS
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

        self._register_commands()

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
        # Inequality, not ">": the value can move backwards. ``rsync -a``
        # stamps destination mtimes from the source, so syncing an older
        # dev-machine copy over a table last written on the bot host (a
        # password change, say) lowers it. Any movement means a change.
        if current == self._index_mtime_seen:
            return
        await loop.run_in_executor(None, self.local.reconnect)
        self._ac_cache.reset()
        await loop.run_in_executor(None, self._reload_shows)
        # Advanced only once the reload succeeded. A sweep landing mid-rsync
        # can open a manifest whose data files haven't arrived; advancing
        # first would mark that torn state as seen and never retry it.
        self._index_mtime_seen = current
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
            store.set_show_password(name, hash_show_password(password))
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
            store.set_show_password(name, hash_show_password(password))
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
