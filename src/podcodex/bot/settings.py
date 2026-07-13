"""Per-server settings persistence and the /setup, /announcements handlers."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, replace

import discord
from loguru import logger

from podcodex.bot.config import ServerSettings
from podcodex.rag.defaults import (
    MODELS,
)


class SettingsMixin:
    """Server-settings methods mixed into PodCodexBot (bot.py).

    Expects on self: ``config``, ``server_config_path``, ``_server_cfg``,
    ``_locked_show_names``.
    """

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
        channel: discord.TextChannel | None,
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
