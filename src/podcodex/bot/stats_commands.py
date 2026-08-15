"""/stats, /speakers, /episodes handler bodies."""

from __future__ import annotations

import asyncio

import discord
from loguru import logger

from podcodex.bot.access import ResolvedShows, ShowAccess
from podcodex.bot.formatting import (
    display_speaker,
    fmt_time,
)
from podcodex.bot.result_store import CachedSearch
from podcodex.bot.ui import (
    build_episodes_embeds,
    build_list_view,
    build_stats_embed,
)


class StatsCommandsMixin:
    """Stats-command methods mixed into PodCodexBot (bot.py).

    Expects on self: ``results``, ``local``, ``_refresh_if_stale``,
    ``_resolve_shows``, ``_resolve_show_collections``, ``_settings_and_explicit``,
    ``_empty_collections_message``, ``_cached_col_info``.
    """

    async def _handle_stats(
        self,
        interaction: discord.Interaction,
        show: str | None,
        model: str | None,
    ) -> None:
        await interaction.response.defer()
        await self._refresh_if_stale()
        settings, base, explicit = self._settings_and_explicit(
            interaction.guild_id, model
        )
        loop = asyncio.get_running_loop()

        try:
            col_info = await self._cached_col_info()
            resolved = (
                self._resolve_shows(settings, show)
                if show
                else ResolvedShows(ShowAccess.ALL)
            )
            collections = self._resolve_show_collections(
                resolved, base, col_info, explicit=explicit
            )
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
                    lambda c=col: self.local.get_episode_stats(c.name),
                )
                per_show.setdefault(col.show, []).extend(stats)

            # Speaker detail only for a single-show scope; the global
            # overview stays a per-show table (mixing speakers across
            # shows is /speakers' job).
            speakers = (
                await loop.run_in_executor(
                    None, self.local.speaker_stats_multi, [c.name for c in collections]
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
        artwork = collections[0].artwork_url if len(collections) == 1 else ""
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
        settings, base, explicit = self._settings_and_explicit(
            interaction.guild_id, model
        )
        loop = asyncio.get_running_loop()

        try:
            col_info = await self._cached_col_info()
            resolved = (
                self._resolve_shows(settings, show)
                if show
                else ResolvedShows(ShowAccess.ALL)
            )
            collections = self._resolve_show_collections(
                resolved, base, col_info, explicit=explicit
            )
            if not collections:
                speaker_shows = resolved if show else None
                await interaction.followup.send(
                    self._empty_collections_message(col_info, settings, speaker_shows),
                    ephemeral=True,
                )
                return

            ranked = await loop.run_in_executor(
                None, self.local.speaker_stats_multi, [c.name for c in collections]
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
            name = display_speaker(r["speaker"])
            if not name:
                # Unattributed time has no one to rank; speaker_stats does not
                # filter the placeholder the way speaker_airtime does.
                continue
            lines.append(
                f"`{i:>2}.` **{name}** — `{fmt_time(r['total_duration'])}` "
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
        settings, base, explicit = self._settings_and_explicit(
            interaction.guild_id, model
        )
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
                    ResolvedShows(ShowAccess.ALL), base, col_info, explicit=explicit
                )
                shows = sorted({c.show for c in pairs})
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
            ResolvedShows(ShowAccess.SPECIFIC, (show,)),
            base,
            col_info,
            explicit=explicit,
        )
        if not ep_pairs:
            await interaction.followup.send(
                f"No episodes found for **{show}**.", ephemeral=True
            )
            return
        col = ep_pairs[0].name

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
