"""/search, /exact, /random handler bodies plus shared gates."""

from __future__ import annotations

import asyncio

import discord
from loguru import logger

from podcodex.bot.access import ResolvedShows
from podcodex.bot.config import ServerSettings
from podcodex.bot.formatting import (
    CooldownManager,
    count_occurrences,
    count_word_occurrences,
    display_speaker,
    fmt_timestamp,
    format_filter_suffix,
    set_chunk_thumbnail,
)
from podcodex.bot.result_store import CachedSearch, ResultRef
from podcodex.bot.ui import (
    ExpandResult,
    build_compact_view,
    build_listen_button,
    build_results_view,
)
from podcodex.rag.hit import Hit
from podcodex.core._utils import normalize_pub_date
from podcodex.rag.search_service import (
    exact_search,
    hybrid_search,
    random_quote,
)

# ── Embed builder ─────────────────────────────


def _chunk_to_ref(chunk: Hit, collection: str) -> ResultRef:
    """Distill a search-result chunk into a cacheable :class:`ResultRef`.

    Stores only the pointer (``collection``/``episode``/``chunk_index``) plus the
    search-time scalars LanceDB does not carry; the transcript text is re-fetched
    on click. See :mod:`podcodex.bot.result_store`.
    """
    return ResultRef(
        collection=collection,
        episode=chunk.episode,
        chunk_index=max(0, chunk.chunk_index),
        score=chunk.score or 0.0,
        fuzzy_match=chunk.fuzzy_match,
        accent_match=chunk.accent_match,
        match_text=chunk.match_text,
        episode_title=chunk.display_title,
    )


class SearchCommandsMixin:
    """Search-command methods mixed into PodCodexBot (bot.py).

    Expects on self: ``config``, ``results``, ``retriever``, ``local``,
    ``_cooldown``, ``_resolve_show_collections``,
    ``_empty_collections_message``, ``_server_settings``, ``_cached_col_info``.
    """

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
            if value and not normalize_pub_date(value):
                await interaction.response.send_message(
                    f"❌ Invalid `{label}` date: `{value}`. Use YYYY-MM-DD.",
                    ephemeral=True,
                )
                return True
        return False

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
        explicit: tuple[str, str] | None = None,
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
                    explicit=explicit,
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
        results: list[tuple[Hit, str]],
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
        explicit: tuple[str, str] | None = None,
    ) -> list[tuple[Hit, str]]:
        """Run hybrid retrieval, one collection per show, and merge results.

        Shows may resolve to collections under different embedding models; the
        shared search service groups retrievers by model so each query is
        encoded once per model, not once per collection.
        """
        col_info = self.local.get_all_collection_info()
        cols = self._resolve_show_collections(
            shows, settings, col_info, explicit=explicit
        )
        if not cols:
            logger.warning("No collections resolved for this query")
            return []

        return hybrid_search(
            query,
            cols,
            top_k=settings.top_k,
            alpha=alpha,
            strategy=self.config.merge_strategy,
            score_floor=0.05,
            episode=episode,
            episodes=None,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
            retriever_factory=self.retriever,
        )

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
            cols = self._resolve_show_collections(shows, settings, col_info)
            if not cols:
                await interaction.followup.send(
                    self._empty_collections_message(col_info, settings, shows),
                    ephemeral=True,
                )
                return

            all_results = await loop.run_in_executor(
                None,
                lambda: exact_search(
                    query,
                    cols,
                    order="chronological",
                    source=source,
                    episode=episode,
                    speaker=speaker,
                    pub_date_min=pub_date_min,
                    pub_date_max=pub_date_max,
                    retriever_factory=self.retriever,
                ),
            )

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

        # Split the mention count: whole-word occurrences are "matches",
        # everything else (inside a longer word, accent variants, and fuzzy
        # excerpts, which contain no literal occurrence) is "partial".
        total_mentions = 0
        word_mentions = 0
        fuzzy_excerpts = 0
        for c, _ in all_results:
            total_mentions += count_occurrences(c.text, query)
            word_mentions += count_word_occurrences(c.text, query)
            if c.fuzzy_match:
                fuzzy_excerpts += 1
        partial = (total_mentions - word_mentions) + fuzzy_excerpts
        if word_mentions and partial:
            label = f"{word_mentions} exact · {partial} partial"
        elif word_mentions:
            label = f"{word_mentions} exact"
        elif partial:
            label = f"{partial} partial"
        else:
            # Never show "0 matches" above a non-empty result list.
            n = len(all_results)
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
            cols = self._resolve_show_collections(shows, settings, col_info)
            if not cols:
                await interaction.followup.send(
                    self._empty_collections_message(col_info, settings, shows),
                    ephemeral=True,
                )
                return

            result = await loop.run_in_executor(
                None,
                lambda: random_quote(
                    cols,
                    episode=episode,
                    source=source,
                    speaker=speaker,
                    pub_date_min=pub_date_min,
                    pub_date_max=pub_date_max,
                    retriever_factory=self.retriever,
                ),
            )
        except Exception:
            logger.exception("Random quote error")
            await interaction.followup.send(
                "❌ Could not fetch a random quote.", ephemeral=True
            )
            return

        if result is None:
            suffix = format_filter_suffix(
                episode=episode, speaker=speaker, source=source
            )
            await interaction.followup.send(
                f"No excerpts found{suffix}. Try without filters.",
                ephemeral=True,
            )
            return

        chunk, col = result

        show = chunk.show
        ep_display = chunk.display_title
        spk = display_speaker(chunk.speaker_label)
        start = chunk.start
        end = chunk.end
        text = chunk.text

        embed = discord.Embed(
            description=f'"{text}"',
            color=discord.Color.blurple(),
        )
        if show:
            embed.set_author(name=show)
        embed.title = ep_display or "(untitled)"
        set_chunk_thumbnail(embed, chunk)
        embed.add_field(name="Speaker", value=spk, inline=True)
        ts_label = fmt_timestamp(start, end, timed=chunk.timed)
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
