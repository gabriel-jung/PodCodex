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
from podcodex.bot.ui import clear_chunk_cache

# Ticks a collection is retried for (one per announce interval) when its new
# episodes could not be announced, before the bot stops trying until the next
# index change.
_ANNOUNCE_RETRIES = 6


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
        # Inequality, not ">": rsync can lower the value by stamping an older
        # source mtime onto a table. See _refresh_if_stale for the long form.
        changed = current_mtime != self._announce_mtime_seen
        if not changed and not self._announce_retry:
            return
        if changed:
            await loop.run_in_executor(None, self.local.reconnect)
            clear_chunk_cache()
            await self._reload_access()
        col_info = await loop.run_in_executor(None, self.local.get_all_collection_info)
        # Advanced only once the reload succeeded, so a tick that dies there
        # (mid-rsync) is retried whole.
        if changed:
            self._announce_mtime_seen = current_mtime
        # A collection that left the index has nothing left to announce.
        for state in (self._announce_retry, self._announce_pending):
            for gone in [c for c in state if c not in col_info]:
                del state[gone]

        # An episode is marked seen once it has been announced (each guild
        # best-effort), never before. A collection whose read or delivery
        # failed is retried on the next ticks on its own (no reload, nothing
        # held back); one whose new episodes have no stats row yet waits for
        # the next index change (`_wait_for_stats`).
        cols = col_info if changed else list(self._announce_retry)
        targets: list | None = None  # resolved once per tick, on first need
        for col in cols:
            new: list[str] = []
            try:
                stems = await loop.run_in_executor(
                    None, lambda c=col: self.local.list_episodes(c)
                )
                new = self.announce.observe(col, set(stems), record=False)
                if new and targets is None:
                    targets = await self._announce_targets(col_info)
                announced, reached = (
                    await self._post_new_episodes(col, new, col_info, targets or [])
                    if new
                    else (set(), True)
                )
            except Exception:
                logger.exception(f"Announce: {col} failed this tick")
                self._retry_announce(col, "read")
                continue
            try:
                self.announce.mark_seen(col, sorted(announced))
            except Exception:
                # Posted already: retrying would post it again.
                logger.exception(f"Announce: {col} posted but not recorded")
            if not reached:
                self._retry_announce(col, "delivery", sorted(new))
            else:
                self._announce_retry.pop(col, None)
                self._wait_for_stats(col, sorted(set(new) - announced), changed)

    def _retry_announce(
        self, col: str, why: str, stems: list[str] | None = None
    ) -> None:
        """A transient failure (``why``: "read" or "delivery"): retried every
        tick, index change or not, up to ``_ANNOUNCE_RETRIES``. Then a delivery
        failure marks its stems seen (dropped, not posted late), and a read
        failure re-baselines the collection, so whatever it holds once it
        reads again is taken silently rather than posted as a burst."""
        attempts = self._announce_retry.get(col, 0) + 1
        if attempts <= _ANNOUNCE_RETRIES:
            self._announce_retry[col] = attempts
            return
        self._announce_retry.pop(col, None)
        if why == "delivery" and stems:
            self.announce.mark_seen(col, stems)
        elif why == "read":
            self.announce.forget(col)
        logger.error(
            f"Announce: gave up on {col} after {_ANNOUNCE_RETRIES} retries ({why})"
        )

    def _wait_for_stats(self, col: str, missing: list[str], changed: bool) -> None:
        """New episodes listed without a stats row yet (a half-synced index).

        Only an index change can fix that, so they are not retried on quiet
        ticks; they come back through ``observe`` on the next change. Counted
        per set of waiting episodes and per index change: after
        ``_ANNOUNCE_RETRIES`` changes still without stats, they are marked seen
        so they neither come back forever nor post weeks late.
        """
        if not missing:
            self._announce_pending.pop(col, None)
            return
        if not changed:
            return
        key = frozenset(missing)
        prev_key, prev_count = self._announce_pending.get(col, (frozenset(), 0))
        count = prev_count + 1 if prev_key == key else 1
        if count > _ANNOUNCE_RETRIES:
            self._announce_pending.pop(col, None)
            self.announce.mark_seen(col, missing)
            logger.error(
                f"Announce: {len(missing)} episode(s) of {col} never got stats; "
                "marked seen without posting"
            )
        else:
            self._announce_pending[col] = (key, count)

    async def _announce_targets(
        self, col_info: dict[str, dict]
    ) -> list[tuple[int, int, object | None, set[str]]]:
        """``(guild_id, channel_id, channel or None, collections it may see)``
        per allowed guild with an announcement channel configured. None marks
        a channel that could not be reached this tick."""
        targets = []
        async for guild_id, settings, channel in self._iter_announce_channels(
            include_unreachable=True
        ):
            try:
                accessible = {
                    c.name
                    for c in self._resolve_show_collections(
                        ResolvedShows(ShowAccess.ALL), settings, col_info
                    )
                }
            except Exception:
                logger.exception(f"Announce: access for guild {guild_id} failed")
                continue
            targets.append(
                (guild_id, settings.announce_channel_id, channel, accessible)
            )
        return targets

    async def _post_new_episodes(
        self,
        col: str,
        new_stems: list[str],
        col_info: dict[str, dict],
        targets: list[tuple[int, int, object | None, set[str]]],
    ) -> tuple[set[str], bool]:
        """Post one collection's new episodes to every guild that may see it.

        Returns ``(stems announced, reached)``: the stems with a stats row,
        once a guild got them or when no guild that may see the show has a
        channel; ``reached`` is False (and nothing announced) while every such
        guild failed or was unreachable.
        """
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(
            None, lambda: self.local.get_episode_stats(col)
        )
        newset = set(new_stems)
        rows = [s for s in stats if s.get("episode") in newset]
        if not rows:
            return set(), True
        # Newest first.
        rows.sort(
            key=lambda e: (e.get("pub_date") or "", e.get("episode", "")),
            reverse=True,
        )
        show = (col_info.get(col, {}) or {}).get("show", "") or col
        embed = build_new_episodes_embed(show, rows)
        audience = [t for t in targets if col in t[3]]
        delivered = False
        for guild_id, channel_id, channel, _accessible in audience:
            if channel is None:
                continue
            try:
                await channel.send(embed=embed)
                delivered = True
            except Exception as exc:
                # One guild's failure (a send, a connection drop) must not
                # abort the others or re-post to those already served.
                logger.warning(
                    f"Announce send failed (guild {guild_id}, channel {channel_id}): {exc!r}"
                )
        if audience and not delivered:
            return set(), False
        return {r["episode"] for r in rows}, True

    async def _announce_channel(self, channel_id: int):
        """Resolve a configured channel id to a sendable channel, or None."""
        channel = self.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(channel_id)
            except Exception as exc:
                # Once per channel, not every tick: a deleted channel or a
                # removed permission stays that way until an admin acts.
                if channel_id not in self._announce_unreachable:
                    self._announce_unreachable.add(channel_id)
                    logger.warning(
                        f"Announce channel {channel_id} unavailable: {exc!r}"
                    )
                return None
        self._announce_unreachable.discard(channel_id)
        return channel if hasattr(channel, "send") else None

    async def _iter_announce_channels(self, *, include_unreachable: bool = False):
        """Yield ``(guild_id, settings, channel)`` for every allowed server with
        an announcement channel configured: the one walk both announcement
        kinds use. An unreachable channel is skipped, or yielded as None with
        ``include_unreachable``."""
        # A snapshot: a guild configured during the awaits below would
        # otherwise change the dict mid-iteration and abort the fan-out.
        for guild_id, settings in list(self._server_cfg.items()):
            if not settings.announce_channel_id or not self.guild_allowed(guild_id):
                continue
            channel = await self._announce_channel(settings.announce_channel_id)
            if channel is not None or include_unreachable:
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
            # Pre-revision deployments stored the bare version under the old
            # key; reading it keeps the pending update announcement instead
            # of silently re-baselining on the first run after the rename.
            stored = self.announce.get_meta("announced_version")
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
