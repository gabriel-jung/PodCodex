"""Show access control: password-protected shows, unlock state, filters."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from enum import Enum

import discord
from loguru import logger

from podcodex.bot.config import ServerSettings
from podcodex.core.show_passwords import hash_show_password, verify_show_password
from podcodex.bot.guards import require_guild

_UNLOCK_KEY_BYTES = 32


def _write_secret(path, data: bytes) -> None:
    """Write ``data`` atomically, owner-only from the moment the file exists
    (atomic_write would publish the default mode until a chmod)."""
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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

    show_id: str  # stable identity, or the legacy display name pre-migration
    name: str  # display label, for anything a user reads
    password_hash: str  # "sha256:<hex>"


class AccessMixin:
    """Access-control methods mixed into PodCodexBot (bot.py).

    Expects on self: ``_shows``, ``_server_cfg``, ``local``,
    ``_server_settings``, ``_save_server_config``, ``_refresh_if_stale``.
    """

    _unlock_key_cache: bytes | None = None
    _unlock_key_is_new = False
    _access_lock_for: tuple[object, asyncio.Lock] | None = None

    def _access_lock(self) -> asyncio.Lock:
        """Serializes reloads with each other and with a rotation done here,
        so the newest read always wins and none judges a stale hash. One per
        event loop (tests run several)."""
        loop = asyncio.get_running_loop()
        if self._access_lock_for is None or self._access_lock_for[0] is not loop:
            self._access_lock_for = (loop, asyncio.Lock())
        return self._access_lock_for[1]

    def _password_fingerprint(self, password_hash: str) -> str:
        """Stand-in for a show's password hash, kept with each unlock.

        Enough to tell that the password changed since the unlock. Keyed with
        a secret kept beside, not inside, ``server_config.json``: an unkeyed
        digest of the hash would let anyone holding the settings file test
        password guesses offline. Prefixed with the hash scheme, so moving
        every hash to a new scheme (same passwords) reads as a format change,
        not as every password changing.
        """
        scheme = password_hash.partition(":")[0]
        digest = hmac.new(self._unlock_key(), password_hash.encode(), hashlib.sha256)
        return f"{scheme}:{digest.hexdigest()[:16]}"

    def _unlock_key(self) -> bytes:
        if self._unlock_key_cache is None:
            path = self.server_config_path.parent / "unlock_key"
            try:
                key = path.read_bytes()
            except FileNotFoundError:
                key = b""
            if len(key) < _UNLOCK_KEY_BYTES:
                # Missing, or cut short by a crash mid-write: an empty key
                # would make the fingerprint an unkeyed digest again.
                key = secrets.token_bytes(_UNLOCK_KEY_BYTES)

                try:
                    _write_secret(path, key)
                except OSError as exc:
                    # A read-only state dir must not stop the bot. The key
                    # then lives for this run only: unlocks re-adopt at each
                    # start, so rotations are caught only while it runs.
                    logger.warning(
                        f"Could not write {path} ({exc}); unlock revocation "
                        "will not survive a restart"
                    )
                self._unlock_key_is_new = True
            self._unlock_key_cache = key
        return self._unlock_key_cache

    @property
    def _locked_show_ids(self) -> set[str]:
        return {e.show_id for e in self._shows.values()}

    def _settings_entries(self) -> frozenset[str]:
        """Every show key the guilds hold, copied (taken on the loop)."""
        return frozenset(
            e
            for settings in list(self._server_cfg.values())
            for e in (
                *settings.unlocked_shows,
                *settings.pinned_shows,
                *settings.allowed_shows,
                *settings.unlock_fingerprints,
            )
        )

    def _load_access(
        self, entries: frozenset[str]
    ) -> tuple[dict[str, ShowEntry], dict[str, str]]:
        """Read the protected shows and resolve ``entries`` to show ids.

        Index reads only, safe in an executor: it touches no bot state, so
        the loop installs the result (and only the newest one).
        """
        raw = self.local.get_show_password_entries()
        shows = {
            key.lower(): ShowEntry(
                show_id=key, name=entry["label"], password_hash=entry["password_hash"]
            )
            for key, entry in raw.items()
        }
        logger.info(f"Shows loaded: {len(shows)} password-protected")
        ids = {e: self._show_id_for_label(e, shows) for e in entries}
        return shows, ids

    async def _reload_access(self) -> None:
        """Reload protected shows off the loop, then settle guild settings on it.

        Both follow-ups edit guild settings, which handlers on the loop copy,
        replace and save; doing them in the executor raced those handlers (a
        revoked unlock coming back through a /setup save built from the old
        list, a save serializing a dict the loop was changing).
        """
        async with self._access_lock():
            (
                self._shows,
                self._show_id_map,
            ) = await asyncio.get_running_loop().run_in_executor(
                None, self._load_access, self._settings_entries()
            )
            self._normalize_show_ids()
            self._revoke_stale_unlocks()

    @property
    def _fingerprint_history(self) -> dict[str, list[str]]:
        """Per show, the fingerprints read as current, oldest first (this run)."""
        if not hasattr(self, "_fp_history"):
            self._fp_history: dict[str, list[str]] = {}
        return self._fp_history

    def _is_older_password(self, show_id: str, current: str, stored: str) -> bool:
        history = self._fingerprint_history.get(show_id, [])
        return (
            current in history
            and stored in history
            and history.index(current) < history.index(stored)
        )

    def _revoke_stale_unlocks(self) -> None:
        """Drop unlocks granted under a password that has since changed.

        Only shows protected right now are judged: one that is public at
        this moment (or a password table read empty mid-rsync) keeps its
        entry and fingerprint, so a reset later is still caught. An unlock
        with no fingerprint (from before fingerprints, or a legacy entry)
        adopts the current one, so upgrading revokes nobody.
        """
        current = {
            e.show_id: self._password_fingerprint(e.password_hash)
            for e in self._shows.values()
        }
        for show_id, fingerprint in current.items():
            history = self._fingerprint_history.setdefault(show_id, [])
            if fingerprint not in history:
                history.append(fingerprint)
        # A key minted just now cannot have produced the stored fingerprints
        # (lost or never copied with the settings): adopt, never revoke.
        adopt_all = self._unlock_key_is_new
        self._unlock_key_is_new = False
        changed = False
        for guild_id, settings in list(self._server_cfg.items()):
            # Deduped: a show can sit in both lists (or twice after
            # normalization), and a second pass after the drop would find no
            # fingerprint and adopt the new password's.
            for show_id in dict.fromkeys(self._unlocked_ids(settings)):
                fingerprint = current.get(show_id)
                if fingerprint is None:
                    continue
                seen = settings.unlock_fingerprints.get(show_id)
                rescheme = (
                    seen is not None
                    and seen.partition(":")[0] != fingerprint.partition(":")[0]
                )
                if seen is None or rescheme or (adopt_all and seen != fingerprint):
                    settings.unlock_fingerprints[show_id] = fingerprint
                    changed = True
                elif seen != fingerprint:
                    if self._is_older_password(show_id, fingerprint, seen):
                        # A password this process saw replaced came back: a
                        # sync read mid-way showing the previous table, not a
                        # change. (A deliberate revert to an old password is
                        # treated the same; rare, and it revokes nothing.)
                        continue
                    self._drop_unlock(settings, show_id)
                    changed = True
                    logger.info(
                        f"Guild {guild_id}: unlock of {show_id!r} revoked, "
                        "its password changed"
                    )
        if changed:
            self._save_server_config()

    # ── Label <-> id ─────────────────────────────
    #
    # The bot only ever receives display names: from a slash-command argument,
    # from autocomplete, from its own config file. Identity lives in the index
    # it reads, so these two are the only places a conversion happens.

    def _show_id_for_label(
        self, label: str, shows: dict[str, ShowEntry] | None = None
    ) -> str:
        """Id of the show with this display name, or the label itself.

        Index-backed (``IndexStore.show_id_for_label``), because the bot has
        no show folders. The extra pass over ``shows`` (default: the loaded
        ones) covers a show that is password-protected but not yet indexed,
        which has no collection row.
        """
        resolved = self.local.show_id_for_label(label)
        if resolved != label:
            return resolved
        target = (label or "").strip().lower()
        for entry in (self._shows if shows is None else shows).values():
            if entry.name.strip().lower() == target:
                return entry.show_id
        return resolved

    def _label_for_show_id(self, show_id: str) -> str:
        """Display name for a show id, or the id itself when unknown."""
        resolved = self.local.label_for_show_id(show_id)
        if resolved != show_id:
            return resolved
        entry = self._shows.get(show_id.lower())
        return entry.name if entry else show_id

    def _normalize_show_ids(self) -> None:
        """Rewrite guild show lists from display names to show ids.

        Names only; nothing is reclassified or removed. Without this the
        first rename after the index migration would silently re-lock every
        guild that had unlocked the show, which is the same bug one layer
        out.
        """
        ids = getattr(self, "_show_id_map", {})
        changed = False
        for settings in list(self._server_cfg.values()):
            for field in ("unlocked_shows", "pinned_shows", "allowed_shows"):
                current = getattr(settings, field)
                migrated = [ids.get(e, e) for e in current]
                if migrated != current:
                    setattr(settings, field, migrated)
                    changed = True
            # Fingerprints follow their unlock, or a rekeyed show would find
            # none under its id and adopt the current password's, skipping
            # the revocation a rotation is meant to cause.
            rekeyed = {
                ids.get(k, k): v for k, v in settings.unlock_fingerprints.items()
            }
            if rekeyed != settings.unlock_fingerprints:
                settings.unlock_fingerprints = rekeyed
                changed = True
        if changed:
            self._save_server_config()
            logger.info("Migrated guild show lists to show ids")

    # ── The two meanings, and the legacy list that holds both ────
    #
    # `allowed_shows` predates the split and cannot be classified after the
    # fact: nothing in it records whether `/unlock` or `/setup show_add`
    # wrote an entry. Guessing is not safe in either direction — reading a
    # legacy unlock as a pin silently revokes access, and reading a legacy
    # pin as an unlock hands out access nobody entered a password for (a
    # public show could be pinned, then protected later).
    #
    # So it is not classified at write time. It is read the way the code
    # before the split read it: a legacy entry counts as an unlock exactly
    # while its show is password-protected, and as a pin exactly while it is
    # not. That is what `_show_allowed` did with the single list, so no
    # upgrade changes anyone's access, and it is re-derived on every read —
    # a reload that lands mid-rsync with an empty password set persists
    # nothing and corrects itself on the next one.
    #
    # It leaves one case the data cannot resolve: a public show pinned via
    # `/setup`, then password-protected later, keeps granting that guild
    # access. Identical before the split for the same reason, so it is
    # preserved rather than introduced; `/lock` settles it.
    #
    # The list drains only when an admin acts on a show by name, which is
    # the one moment its meaning is known.

    def _unlocked_ids(self, settings: ServerSettings) -> list[str]:
        """Protected shows this guild may see."""
        locked = self._locked_show_ids
        legacy = [s for s in settings.allowed_shows if s in locked]
        return [*settings.unlocked_shows, *legacy]

    def _pinned_ids(self, settings: ServerSettings) -> list[str]:
        """Shows pinned as this guild's defaults.

        A protected legacy entry is not offered as a pin: it is an unlock as
        far as anything can tell, and counting it here made
        `/setup show_remove` report success while the entry stayed.
        """
        locked = self._locked_show_ids
        legacy = [s for s in settings.allowed_shows if s not in locked]
        return [*settings.pinned_shows, *legacy]

    def _drop_unlock(self, settings: ServerSettings, show_id: str) -> bool:
        """Revoke an unlock, legacy entry included. True if anything went.

        The legacy entry goes only when the show is actually protected: for
        a public one `_pinned_ids` is reading that entry as a pin, and
        deleting it here would silently drop the guild's search default
        while reporting an access change — the exact confusion the split
        exists to end.
        """
        fields = ["unlocked_shows"]
        if show_id in self._locked_show_ids:
            fields.append("allowed_shows")
        removed = False
        for name in fields:
            lst = getattr(settings, name)
            if show_id in lst:
                lst.remove(show_id)
                removed = True
        settings.unlock_fingerprints.pop(show_id, None)
        return removed

    def _resolve_shows(
        self, settings: ServerSettings, explicit_show: str = ""
    ) -> ResolvedShows:
        """Resolve which shows a command may query.

        Shows are public by default. A show becomes password-protected only
        when the bot owner sets a password via ``--manage-passwords``.
        """
        if not explicit_show:
            return ResolvedShows(ShowAccess.ALL)
        if not self._locked_show_ids:
            # Nothing protected: no index read on the event loop.
            return ResolvedShows(ShowAccess.SPECIFIC, (explicit_show,))
        # Every collection carrying the name, judged by the same rule the
        # collection filter applies (``_row_allowed``): the name is locked only
        # if none is visible here, and a visible twin never exposes the locked
        # one since collections are filtered by that rule downstream too. A
        # protected show not indexed yet has no rows; its password entry
        # still locks the name.
        rows = self.local.rows_for_label(explicit_show)
        visible = (
            any(
                self._row_allowed(m.get("show_id") or "", m.get("show") or "", settings)
                for m in rows
            )
            if rows
            else self._show_allowed(self._show_id_for_label(explicit_show), settings)
        )
        if not visible:
            return ResolvedShows(ShowAccess.LOCKED)
        # Downstream resolves collections by label, so the label travels.
        return ResolvedShows(ShowAccess.SPECIFIC, (explicit_show,))

    def _show_allowed(self, show_id: str, settings: ServerSettings) -> bool:
        """Whether this server may see a show: public, or unlocked here.

        Takes the show's id, not its name: a rename must not silently make a
        protected show public.
        """
        return show_id not in self._locked_show_ids or show_id in self._unlocked_ids(
            settings
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
        if not self._locked_show_ids:
            return collections  # nothing is password-protected
        info_map = (
            col_info if col_info is not None else self.local.get_all_collection_info()
        )
        return [
            col
            for col in collections
            if self._row_allowed(
                (info_map.get(col) or {}).get("show_id") or "",
                (info_map.get(col) or {}).get("show") or col,
                settings,
            )
        ]

    def _row_allowed(self, show_id: str, label: str, settings: ServerSettings) -> bool:
        """Whether this server may see one collection: the one access rule.

        A collection is protected by its id and, until the migration rekeys
        it, by a legacy password row keyed by its display name; an index read
        mid-migration can hold either shape for either side. So a row must
        pass under its id (or, with none, the id its label resolves to) and
        under its label. This fails closed when two shows share a name and one
        is protected by a legacy row: both stay hidden until the migration
        separates them, rather than one exposing the other.
        """
        ident = show_id or self._show_id_for_label(label)
        return self._show_allowed(ident, settings) and self._show_allowed(
            label, settings
        )

    async def _handle_unlock(
        self,
        interaction: discord.Interaction,
        password: str,
    ) -> None:
        if await require_guild(interaction) is None:
            return
        # Before the index work below, which can outlast Discord's 3 seconds.
        await interaction.response.defer(ephemeral=True)
        # Reconnect if the index was rsynced under the running process, then
        # read the passwords fresh (set from the desktop app while the bot
        # runs), whether or not the staleness check saw a change.
        if not await self._refresh_if_stale():
            await self._reload_access()

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
            await interaction.followup.send("Invalid password.", ephemeral=True)
            return

        guild_id = interaction.guild_id
        settings = self._server_settings(guild_id)
        fingerprint = self._password_fingerprint(entry.password_hash)
        newly = entry.show_id not in self._unlocked_ids(settings)
        if newly:
            settings.unlocked_shows.append(entry.show_id)
        if newly or settings.unlock_fingerprints.get(entry.show_id) != fingerprint:
            settings.unlock_fingerprints[entry.show_id] = fingerprint
            self._server_cfg[guild_id] = settings
            self._save_server_config()
        if newly:
            logger.info(f"Guild {guild_id} unlocked show {entry.name!r}")

        await interaction.followup.send(
            f"Show **{entry.name}** is now available on this server.",
            ephemeral=True,
        )

    async def _handle_lock(
        self,
        interaction: discord.Interaction,
        show: str,
    ) -> None:
        guild_id = await require_guild(interaction)
        if guild_id is None:
            return
        settings = self._server_settings(guild_id)
        show_id = self._show_id_for_label(show)
        if self._drop_unlock(settings, show_id):
            self._server_cfg[guild_id] = settings
            self._save_server_config()
            logger.info(f"Guild {guild_id} locked show {show!r}")
            await interaction.response.send_message(
                f"Show **{show}** has been removed from this server.",
                ephemeral=True,
            )
        elif show_id not in self._locked_show_ids:
            # Nothing to revoke: the show is public. Saying "removed" here is
            # what the old shared list made this branch do, and it read as an
            # access change when only a search default had moved.
            await interaction.response.send_message(
                f"Show **{show}** is not password-protected, so there is "
                "nothing to lock. Use `/setup show_remove` to drop it as a "
                "default.",
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
        guild_id = await require_guild(interaction)
        if guild_id is None:
            return
        # Before the rotation, the save and the reload, which can outlast
        # Discord's 3 seconds; a late reply would lose the new password.
        await interaction.response.defer(ephemeral=True)
        # Fresh access first: a password changed outside the bot must revoke
        # this guild before it can rotate (and keep) the show.
        if not await self._refresh_if_stale():
            await self._reload_access()
        settings = self._server_settings(guild_id)

        show_id = self._show_id_for_label(show)
        if show_id not in self._locked_show_ids:
            # `unlocked_shows` is never pruned when the app makes a show
            # public, so a stale entry (which the unlocked picker still
            # offers) would otherwise let this call `set_show_password` on a
            # public show — re-protecting it index-wide and locking every
            # other guild out, with the new password DM'd only to whoever
            # ran the command.
            await interaction.followup.send(
                f"Show **{show}** is not password-protected, so there is no "
                "password to rotate.",
                ephemeral=True,
            )
            return
        if show_id not in self._unlocked_ids(settings):
            await interaction.followup.send(
                f"Show **{show}** is not unlocked on this server. Unlock it first with /unlock.",
                ephemeral=True,
            )
            return

        # Generate new password and update the index.
        password = secrets.token_urlsafe(16)
        new_hash = hash_show_password(password)
        from podcodex.rag.index_origin import IndexOwnershipError

        # Under the access lock, so no reload reads the table between the
        # write and this guild's new fingerprint.
        async with self._access_lock():
            try:
                self.local.set_show_password(show_id, new_hash, show_label=show)
            except IndexOwnershipError:
                await interaction.followup.send(
                    "This bot reads a copy of the index that is maintained on "
                    "another machine, so a new password set here would be lost on "
                    "the next sync. Rotate it from the PodCodex app instead.",
                    ephemeral=True,
                )
                return
            # This guild keeps its unlock; every other guild that unlocked with
            # the old password loses it on the next reload.
            settings.unlock_fingerprints[show_id] = self._password_fingerprint(new_hash)
            self._server_cfg[guild_id] = settings

        # The password goes out before the save and the reload: it now exists
        # only here. Whatever happens to the delivery, the settings are saved
        # and the other guilds revoked.
        try:
            await self._deliver_new_password(interaction, show, password)
        except Exception:
            logger.exception(
                f"Password for {show!r} rotated, but it could not be delivered"
            )
        finally:
            try:
                self._save_server_config()
                await self._reload_access()
            except Exception:
                logger.exception(
                    "Password rotated, but saving or reloading guild settings failed"
                )

    async def _deliver_new_password(
        self, interaction: discord.Interaction, show: str, password: str
    ) -> None:
        # Send via DM to keep the password out of the guild audit log.
        try:
            await interaction.user.send(
                f"New password for **{show}**:\n```\n{password}\n```\n"
                "Share this with the show owner. It cannot be recovered after this message."
            )
            await interaction.followup.send(
                f"Password for **{show}** has been rotated. Check your DMs.",
                ephemeral=True,
            )
        except discord.HTTPException:
            # DM failed (blocked, rate-limited, server error) — fall back to ephemeral
            # so the caller still receives the new password.
            await interaction.followup.send(
                f"Password for **{show}** rotated.\n"
                f"**Could not send DM** (enable DMs from server members, or try again).\n"
                f"New password: `{password}`",
                ephemeral=True,
            )
