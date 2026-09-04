"""Per-server settings persistence and the /setup, /announcements handlers."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from difflib import get_close_matches

import discord
from loguru import logger

from podcodex.bot.config import ServerSettings
from podcodex.bot.guards import require_guild
from podcodex.rag.defaults import (
    MODELS,
)


class SettingsMixin:
    """Server-settings methods mixed into PodCodexBot (bot.py).

    Expects on self: ``config``, ``server_config_path``, ``_server_cfg``,
    ``_locked_show_ids``, ``_label_for_show_id``, ``_show_id_for_label``,
    ``_known_show_labels``.
    """

    def _load_server_config(self) -> dict[int, ServerSettings]:
        if not self.server_config_path.exists():
            return {}
        raw = json.loads(self.server_config_path.read_text(encoding="utf-8"))
        valid_keys = {f.name for f in fields(ServerSettings)}
        result: dict[int, ServerSettings] = {}
        for sid, d in raw.items():
            # A guild-less write (a DM before the handlers guarded against it)
            # left the literal key "None" behind. Skip anything that is not a
            # guild id rather than failing the whole bot start on it.
            try:
                guild_id = int(sid)
            except (TypeError, ValueError):
                logger.warning(f"Ignoring non-guild key {sid!r} in server config")
                continue
            # Backward compat: rename old "default_shows" → "allowed_shows"
            if "default_shows" in d and "allowed_shows" not in d:
                d["allowed_shows"] = d.pop("default_shows")
            filtered = {k: v for k, v in d.items() if k in valid_keys}
            result[guild_id] = ServerSettings(**filtered)
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

    def _settings_and_explicit(
        self, guild_id: int | None, model: str | None
    ) -> tuple[ServerSettings, ServerSettings, tuple[str, str] | None]:
        """Merged settings, unmerged guild settings, explicit override.

        `settings` (merged) drives messaging so the empty-collections text
        still names the model the user actually typed. Resolution takes the
        unmerged `base` as its default tier plus `explicit` on top; feeding
        the merged settings there would let a failed explicit combo collapse
        past the guild's real default (see /search-advanced).
        """
        settings = self._effective_settings(guild_id, model or "", 0)
        base = self._server_settings(guild_id)
        explicit = (model, base.chunker) if model else None
        return settings, base, explicit

    @staticmethod
    def _model_label(model: str) -> str:
        """Human label for a model key; a stale/unknown key passes through raw
        instead of raising (server configs can outlive the MODELS registry)."""
        return MODELS[model].label if model in MODELS else model

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
        guild_id = await require_guild(interaction)
        if guild_id is None:
            return
        # Deferred up front: validating `show_add` reads the collection list,
        # which on a cold autocomplete cache and a large index can outrun
        # Discord's 3-second first-response window and show the admin "The
        # application did not respond" whether or not the save happened.
        await interaction.response.defer(ephemeral=True)
        # Same two steps the show autocompletes run, so a name the picker
        # offered is never rejected here: reconnect after an external index
        # change, then drop a cached collection list past its TTL.
        await self._refresh_if_stale()
        self._cache_clear_if_stale()
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
            lines = [
                "**Current settings**",
                f"Model: `{current.model}`",
                f"Chunker: `{current.chunker}`",
                f"Top-k: `{current.top_k}`",
                f"Pinned shows: {self._pinned_shows_str(current)}",
            ]
            # Only worth a line where something is actually protected;
            # otherwise it is a row of "(none)" that means nothing.
            if self._locked_show_ids:
                unlocked = self._show_labels_str(self._unlocked_ids(current))
                lines.append(f"Unlocked shows: {unlocked or '*(none — use /unlock)*'}")
            lines += [
                f"Default source: `{current.default_source or '(any)'}`",
                f"Compact: `{current.compact}`",
                f"Merge: `{self.config.merge_strategy}`",
            ]
            await interaction.followup.send("\n".join(lines), ephemeral=True)
            return

        # Pins are this guild's default shows; they grant no access, so a
        # protected show has to be unlocked before it can be pinned (it is
        # simply not in the visible set below, which is also why an unknown
        # name never confirms that a protected show exists).
        # `pinned_shows` only. The legacy entries `_pinned_ids` also reports
        # stay where they are and are settled separately below — seeding this
        # list from `_pinned_ids` copied them in while leaving the originals,
        # so one show ended up in both lists and `_pinned_ids` returned it
        # twice.
        new_shows = [] if show_clear else list(current.pinned_shows)
        remove_id = ""
        if show_add:
            add_id = await self._resolve_pinnable_show(interaction, current, show_add)
            if add_id is None:
                return
            # Against the full set, so re-pinning a legacy entry is a no-op
            # rather than a second copy of it.
            if add_id not in self._pinned_ids(current):
                new_shows.append(add_id)
        if show_remove:
            # Validated against the guild's own pins, not the index: a show
            # that has since left the index must still be un-pinnable.
            remove_id = self._show_id_for_label(show_remove)
            # Against the guild's pins as they were, not `new_shows`: with
            # `show_clear` also set, the list is already empty here and a
            # redundant removal would fail the whole call instead of being
            # the no-op it is.
            if remove_id not in self._pinned_ids(current):
                await interaction.followup.send(
                    f"**{show_remove}** is not pinned on this server. "
                    f"Pinned: {self._pinned_shows_str(current)}",
                    ephemeral=True,
                )
                return
            if remove_id in new_shows:
                new_shows.remove(remove_id)

        # The legacy list is rewritten only by a command that names what to
        # do with it. It used to be recomputed on every `/setup`, classifying
        # against `_locked_show_ids` at that instant — so a `/setup top_k:10`
        # issued while the password table was unreadable (an index rsynced
        # mid-transfer reports no protected shows without erroring) filed
        # every legacy unlock as a pin and cleared the list, and the guild
        # lost its access for good. The read path re-derives instead, which
        # is why it survives that window; this must not undo it.
        new_legacy = list(current.allowed_shows)
        if show_clear:
            # Settles the entries `_pinned_ids` was offering as pins; a
            # protected one is an unlock, so `/lock` settles that instead.
            new_legacy = [s for s in new_legacy if s in self._locked_show_ids]
        if show_remove and remove_id:
            new_legacy = [s for s in new_legacy if s != remove_id]
        updated = replace(
            current,
            model=model or current.model,
            chunker=chunker or current.chunker,
            top_k=top_k or current.top_k,
            pinned_shows=new_shows,
            allowed_shows=new_legacy,
            default_source=default_source if default_source else current.default_source,
            compact=compact == "true" if compact else current.compact,
        )
        self._server_cfg[guild_id] = updated
        self._save_server_config()
        logger.info(f"Guild {guild_id} updated: {updated}")

        await interaction.followup.send(
            f"✅ Settings updated\n"
            f"Model: `{updated.model}`\n"
            f"Chunker: `{updated.chunker}`\n"
            f"Top-k: `{updated.top_k}`\n"
            f"Pinned shows: {self._pinned_shows_str(updated)}\n"
            f"Default source: `{updated.default_source or '(any)'}`\n"
            f"Compact: `{updated.compact}`",
            ephemeral=True,
        )

    def _show_labels_str(self, show_ids: list[str]) -> str:
        """Ids rendered as the display names users read, or ``""``."""
        return ", ".join(f"`{self._label_for_show_id(s)}`" for s in show_ids)

    def _pinned_shows_str(self, settings: ServerSettings) -> str:
        return self._show_labels_str(self._pinned_ids(settings)) or "*(none)*"

    async def _resolve_pinnable_show(
        self,
        interaction: discord.Interaction,
        settings: ServerSettings,
        label: str,
    ) -> str | None:
        """Show id for a user-typed name, or None after answering the user.

        A name that resolves to nothing used to be stored verbatim and echoed
        back as if it were pinned, so the admin believed the default was set
        while every later command ignored it.
        """
        known = await self._known_show_labels(settings)
        match = next(
            (k for k in known if k.strip().lower() == label.strip().lower()), None
        )
        if match is not None:
            return self._show_id_for_label(match)
        close = get_close_matches(label, known, n=3, cutoff=0.5)
        hint = (
            f" Did you mean {', '.join(f'**{c}**' for c in close)}?"
            if close
            else " Nothing on this server matches that name."
        )
        await interaction.followup.send(
            f"No show called **{label}** is available here.{hint}",
            ephemeral=True,
        )
        return None

    # ── /announcements handler ────────────────

    async def _handle_announcements(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel | None,
        off: bool,
    ) -> None:
        guild_id = await require_guild(interaction)
        if guild_id is None:
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
