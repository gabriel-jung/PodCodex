"""Show access control: password-protected shows, unlock state, filters."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from enum import Enum

import discord
from loguru import logger

from podcodex.bot.config import ServerSettings
from podcodex.core.show_passwords import hash_show_password, verify_show_password

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
    def is_locked(self) -> bool:
        return self.access is ShowAccess.LOCKED

    @property
    def is_specific(self) -> bool:
        return self.access is ShowAccess.SPECIFIC


@dataclass
class ShowEntry:
    """A single password-protected show (stored in the IndexStore)."""

    name: str
    password_hash: str  # "sha256:<hex>"


class AccessMixin:
    """Access-control methods mixed into PodCodexBot (bot.py).

    Expects on self: ``_shows``, ``_server_cfg``, ``local``,
    ``_server_settings``, ``_save_server_config``, ``_refresh_if_stale``.
    """

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

    def _resolve_shows(
        self, settings: ServerSettings, explicit_show: str = ""
    ) -> ResolvedShows:
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
                if verify_show_password(password, e.password_hash)
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
        self.local.set_show_password(show, hash_show_password(password))
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
