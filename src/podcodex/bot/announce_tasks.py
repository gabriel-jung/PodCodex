"""Background announce loop: new-episode and version announcements."""

from __future__ import annotations

import asyncio

import discord
from discord.ext import tasks
from loguru import logger

from podcodex.bot.access import ResolvedShows, ShowAccess
from podcodex.bot.announce import (
    bot_revision,
    build_new_episodes_embed,
    build_update_embed,
    changelog_section,
    commit_subjects,
    repo_url,
)


class AnnounceMixin:
    """Announce-loop methods mixed into PodCodexBot (bot.py).

    Expects on self: ``announce``, ``local``, ``_server_cfg``,
    ``_resolve_show_collections``, ``_announce_mtime_seen``. Uses its own
    mtime watermark, deliberately independent from the user-path one.
    """

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
                c.name
                for c in self._resolve_show_collections(
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
        """Announce once when the running revision changes.

        The bot is deployed by git pull / rsync, so it updates far more often
        than it is released: the revision is ``version+commit``, and a plain
        code update announces just like a version bump. A new version shows the
        release notes; a same-version update lists the commits since the last
        announcement.

        First ever run baselines silently (no announce), so enabling this never
        backfills. Idempotent across the repeated ``on_ready`` fired on gateway
        reconnects.
        """
        loop = asyncio.get_running_loop()
        version, sha = await loop.run_in_executor(None, bot_revision)
        if not version:
            return

        current = f"{version}+{sha}" if sha else version
        stored = self.announce.get_meta("announced_revision")
        if stored is None:
            self.announce.set_meta("announced_revision", current)
            return
        if stored == current:
            return

        stored_version, _, stored_sha = stored.partition("+")
        notes = changelog_section(version) if stored_version != version else ""
        changes = (
            await loop.run_in_executor(None, commit_subjects, stored_sha, sha)
            if not notes
            else []
        )
        embed = build_update_embed(
            version,
            sha=sha,
            repo=await loop.run_in_executor(None, repo_url),
            notes=notes,
            changes=changes,
        )
        async for guild_id, _settings, channel in self._iter_announce_channels():
            try:
                await channel.send(embed=embed)
            except discord.HTTPException:
                logger.warning(f"Update announce send failed (guild {guild_id})")
        self.announce.set_meta("announced_revision", current)
