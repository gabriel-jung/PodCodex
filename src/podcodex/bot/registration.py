"""Slash-command registration: all command closures and the help embed."""

from __future__ import annotations

from dataclasses import replace

import discord
from discord import app_commands

from podcodex.rag.defaults import (
    ALPHA,
)

# ── Slash-command choices ─────────────────────

_BOOL_CHOICES = [
    app_commands.Choice(name="True", value="true"),
    app_commands.Choice(name="False", value="false"),
]


class RegistrationMixin:
    """Command registration mixed into PodCodexBot (bot.py). Called once from
    ``__init__``; every closure delegates to a ``_run_*`` / ``_handle_*``
    method defined on the other mixins.
    """

    def _register_commands(self) -> None:

        # /search ─────────────────────────────
        @self.tree.command(
            name="search",
            description="Search podcast transcripts (uses server defaults)",
        )
        @app_commands.describe(query="What are you looking for?")
        async def search(
            interaction: discord.Interaction,
            query: str,
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            await self._refresh_if_stale()
            settings = self._effective_settings(interaction.guild_id)
            shows = self._resolve_shows(settings, "")
            label = f"α={ALPHA:.2f} • {self._model_label(settings.model)}"
            await self._run_search(
                interaction,
                query,
                shows,
                settings,
                ALPHA,
                label,
                source=settings.default_source or None,
                compact=settings.compact,
            )

        # /search-advanced ────────────────────
        @self.tree.command(
            name="search-advanced",
            description="Search with full control over retrieval tuning",
        )
        @app_commands.describe(
            query="What are you looking for?",
            show="Pick a show (searches all if empty)",
            episode="Pick an episode",
            speaker="Filter by speaker name",
            source="Transcript version: corrected, transcript, or a language code",
            after="Oldest publication date to include, YYYY-MM-DD (inclusive)",
            before="Newest publication date to include, YYYY-MM-DD (inclusive)",
            alpha="0 = keywords only → 1 = meaning only (default 0.5)",
            model="Embedding model (leave empty for server default)",
            chunker="Chunking strategy (leave empty for server default)",
            top_k="How many results to show (leave empty for server default)",
            compact="Show results in a single compact embed",
        )
        @app_commands.choices(compact=_BOOL_CHOICES)
        async def search_advanced(
            interaction: discord.Interaction,
            query: str,
            show: str = "",
            episode: str = "",
            speaker: str = "",
            source: str = "",
            after: str = "",
            before: str = "",
            alpha: app_commands.Range[float, 0.0, 1.0] = ALPHA,
            model: str = "",
            chunker: str = "",
            top_k: app_commands.Range[int, 1, 25] = 0,
            compact: str = "",
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            if await self._reject_bad_date(interaction, after, before):
                return
            await self._refresh_if_stale()
            # Model/chunker are threaded separately as `explicit` rather than
            # pre-merged into `settings`, so a typo'd explicit combo still
            # falls back through show.toml prefs and the server default
            # instead of collapsing straight to first-by-name.
            base = self._server_settings(interaction.guild_id)
            settings = replace(base, top_k=top_k or base.top_k)
            explicit = (
                (model or base.model, chunker or base.chunker)
                if (model or chunker)
                else None
            )
            effective_source = source or settings.default_source or None
            use_compact = compact == "true" if compact else settings.compact
            shows = self._resolve_shows(settings, show)
            label = f"α={alpha:.2f} • {self._model_label(model or base.model)}"
            await self._run_search(
                interaction,
                query,
                shows,
                settings,
                alpha,
                label,
                source=effective_source,
                episode=episode or None,
                speaker=speaker or None,
                pub_date_min=after or None,
                pub_date_max=before or None,
                compact=use_compact,
                explicit=explicit,
            )

        search_advanced.autocomplete("show")(self._show_autocomplete)
        search_advanced.autocomplete("episode")(self._episode_autocomplete)
        search_advanced.autocomplete("source")(self._source_autocomplete)
        search_advanced.autocomplete("speaker")(self._speaker_autocomplete)
        search_advanced.autocomplete("model")(self._model_autocomplete)
        search_advanced.autocomplete("chunker")(self._chunker_autocomplete)

        # /exact ──────────────────────────────
        @self.tree.command(
            name="exact",
            description="Literal substring search — case-insensitive, like Ctrl+F",
        )
        @app_commands.describe(query="Text to find (not case-sensitive)")
        async def exact(
            interaction: discord.Interaction,
            query: str,
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            await self._refresh_if_stale()
            settings = self._server_settings(interaction.guild_id)
            shows = self._resolve_shows(settings, "")
            await self._run_exact(interaction, query, shows)

        # /exact-advanced ─────────────────────
        @self.tree.command(
            name="exact-advanced",
            description="Literal substring search with source and date filters",
        )
        @app_commands.describe(
            query="Text to find (not case-sensitive)",
            show="Pick a show (searches all if empty)",
            episode="Pick an episode",
            speaker="Filter by speaker name",
            source="Transcript version: corrected, transcript, or a language code",
            after="Oldest publication date to include, YYYY-MM-DD (inclusive)",
            before="Newest publication date to include, YYYY-MM-DD (inclusive)",
        )
        async def exact_advanced(
            interaction: discord.Interaction,
            query: str,
            show: str = "",
            episode: str = "",
            speaker: str = "",
            source: str = "",
            after: str = "",
            before: str = "",
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            if await self._reject_bad_date(interaction, after, before):
                return
            await self._refresh_if_stale()
            settings = self._server_settings(interaction.guild_id)
            effective_source = source or settings.default_source or None
            shows = self._resolve_shows(settings, show)
            await self._run_exact(
                interaction,
                query,
                shows,
                source=effective_source,
                episode=episode or None,
                speaker=speaker or None,
                pub_date_min=after or None,
                pub_date_max=before or None,
            )

        exact_advanced.autocomplete("show")(self._show_autocomplete)
        exact_advanced.autocomplete("episode")(self._episode_autocomplete)
        exact_advanced.autocomplete("source")(self._source_autocomplete)
        exact_advanced.autocomplete("speaker")(self._speaker_autocomplete)

        # /random ─────────────────────────────
        @self.tree.command(
            name="random",
            description="Pull a random quote from the transcripts",
        )
        async def random_cmd(interaction: discord.Interaction) -> None:
            if not await self._check_cooldown(interaction):
                return
            await self._refresh_if_stale()
            settings = self._server_settings(interaction.guild_id)
            shows = self._resolve_shows(settings, "")
            await self._run_random(interaction, shows)

        # /random-advanced ────────────────────
        @self.tree.command(
            name="random-advanced",
            description="Pull a random quote with source and date filters",
        )
        @app_commands.describe(
            show="Pick a show (random from all if empty)",
            episode="Pick an episode",
            speaker="Filter by speaker name",
            source="Transcript version: corrected, transcript, or a language code",
            after="Oldest publication date to include, YYYY-MM-DD (inclusive)",
            before="Newest publication date to include, YYYY-MM-DD (inclusive)",
        )
        async def random_advanced(
            interaction: discord.Interaction,
            show: str = "",
            episode: str = "",
            speaker: str = "",
            source: str = "",
            after: str = "",
            before: str = "",
        ) -> None:
            if not await self._check_cooldown(interaction):
                return
            if await self._reject_bad_date(interaction, after, before):
                return
            await self._refresh_if_stale()
            settings = self._server_settings(interaction.guild_id)
            effective_source = source or settings.default_source or None
            shows = self._resolve_shows(settings, show)
            await self._run_random(
                interaction,
                shows,
                source=effective_source,
                episode=episode or None,
                speaker=speaker or None,
                pub_date_min=after or None,
                pub_date_max=before or None,
            )

        random_advanced.autocomplete("show")(self._show_autocomplete)
        random_advanced.autocomplete("episode")(self._episode_autocomplete)
        random_advanced.autocomplete("source")(self._source_autocomplete)
        random_advanced.autocomplete("speaker")(self._speaker_autocomplete)

        # /stats ──────────────────────────────
        @self.tree.command(
            name="stats",
            description="Index overview: shows, episodes, duration",
        )
        @app_commands.describe(
            show="Pick a show (shows all if empty)",
            model="Search model (leave empty for server default)",
        )
        async def stats(
            interaction: discord.Interaction,
            show: str = "",
            model: str = "",
        ) -> None:
            await self._handle_stats(interaction, show or None, model or None)

        stats.autocomplete("show")(self._show_autocomplete)
        stats.autocomplete("model")(self._model_autocomplete)

        # /episodes ───────────────────────────
        @self.tree.command(
            name="episodes",
            description="List episodes for a show with excerpt count and duration",
        )
        @app_commands.describe(
            show="Pick a show (auto-selected if only one)",
            model="Search model (leave empty for server default)",
        )
        async def episodes(
            interaction: discord.Interaction,
            show: str = "",
            model: str = "",
        ) -> None:
            await self._handle_episodes(interaction, show or None, model or None)

        episodes.autocomplete("show")(self._show_autocomplete)
        episodes.autocomplete("model")(self._model_autocomplete)

        # /speakers ───────────────────────────
        @self.tree.command(
            name="speakers",
            description="Who speaks the most — excerpt count and airtime per speaker",
        )
        @app_commands.describe(
            show="Pick a show (aggregates all if empty)",
            model="Search model (leave empty for server default)",
        )
        async def speakers(
            interaction: discord.Interaction,
            show: str = "",
            model: str = "",
        ) -> None:
            await self._handle_speakers(interaction, show or None, model or None)

        speakers.autocomplete("show")(self._show_autocomplete)
        speakers.autocomplete("model")(self._model_autocomplete)

        # /setup ──────────────────────────────
        @self.tree.command(
            name="setup",
            description="Configure bot defaults for this server (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(
            model="Default search model for this server",
            chunker="How transcripts are split up for search",
            top_k="How many results to show by default",
            show_add="Always search this show by default",
            show_remove="Stop searching this show by default",
            show_clear="Remove all default shows (search everything)",
            default_source="Default source to search: corrected, transcript, etc.",
            compact="Use compact results by default",
        )
        @app_commands.choices(
            show_clear=_BOOL_CHOICES,
            compact=_BOOL_CHOICES,
        )
        async def setup(
            interaction: discord.Interaction,
            model: str = "",
            chunker: str = "",
            top_k: app_commands.Range[int, 1, 25] = 0,
            show_add: str = "",
            show_remove: str = "",
            show_clear: str = "",
            default_source: str = "",
            compact: str = "",
        ) -> None:
            await self._handle_setup(
                interaction,
                model or None,
                chunker or None,
                top_k or None,
                show_add=show_add or None,
                show_remove=show_remove or None,
                show_clear=show_clear == "true",
                default_source=default_source,
                compact=compact,
            )

        setup.autocomplete("show_add")(self._show_autocomplete)
        setup.autocomplete("show_remove")(self._pinned_show_autocomplete)
        setup.autocomplete("default_source")(self._source_autocomplete)
        setup.autocomplete("model")(self._model_autocomplete)
        setup.autocomplete("chunker")(self._chunker_autocomplete)

        # /announcements ──────────────────────
        @self.tree.command(
            name="announcements",
            description="Set the channel for new-episode and version updates (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(
            channel="Channel to post announcements in",
            off="Turn announcements off for this server",
        )
        @app_commands.choices(off=_BOOL_CHOICES)
        async def announcements(
            interaction: discord.Interaction,
            channel: discord.TextChannel | None = None,
            off: str = "",
        ) -> None:
            await self._handle_announcements(interaction, channel, off == "true")

        # /sync ───────────────────────────────
        @self.tree.command(
            name="sync",
            description="Manually sync slash commands (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        async def sync(interaction: discord.Interaction) -> None:
            await interaction.response.defer(ephemeral=True)
            await self.tree.sync()
            await interaction.followup.send(
                "✅ Command tree synced. New or renamed commands may take up to "
                "an hour to appear in this server (Discord cache).",
                ephemeral=True,
            )

        # /admin-reload ───────────────────────
        @self.tree.command(
            name="admin-reload",
            description="Reconnect to the index and reload show passwords (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        async def admin_reload(interaction: discord.Interaction) -> None:
            await self._handle_admin_reload(interaction)

        # /unlock ─────────────────────────────
        @self.tree.command(
            name="unlock",
            description="Unlock a show for this server using a password (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(
            password="Access password provided by the bot owner",
        )
        async def unlock(
            interaction: discord.Interaction,
            password: str,
        ) -> None:
            await self._handle_unlock(interaction, password)

        # /lock ───────────────────────────────
        @self.tree.command(
            name="lock",
            description="Remove a show from this server (admin)",
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(show="Name of the show to lock")
        async def lock(
            interaction: discord.Interaction,
            show: str,
        ) -> None:
            await self._handle_lock(interaction, show)

        lock.autocomplete("show")(self._pinned_show_autocomplete)

        # /changepassword ─────────────────────
        @self.tree.command(
            name="changepassword",
            description="Rotate the password for a show you have already unlocked on this server",
        )
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.describe(show="Show to rotate the password for")
        async def changepassword(
            interaction: discord.Interaction,
            show: str,
        ) -> None:
            await self._handle_changepassword(interaction, show)

        changepassword.autocomplete("show")(self._pinned_show_autocomplete)

        # /help ───────────────────────────────
        @self.tree.command(
            name="help",
            description="Show available commands and how to use them",
        )
        async def help_cmd(interaction: discord.Interaction) -> None:
            embed = discord.Embed(
                title="📖 PodCodex Help",
                description="Search and explore your podcast transcripts.",
                color=discord.Color.blurple(),
            )
            embed.add_field(
                name="/search `question`",
                value=(
                    "Find relevant passages using a mix of keyword and semantic search.\n"
                    "`alpha` controls the blend: 0 = keywords only, 1 = meaning only (default 0.5)."
                ),
                inline=False,
            )
            embed.add_field(
                name="/exact `query`",
                value="Find exact text matches, like Ctrl+F across all episodes.",
                inline=False,
            )
            embed.add_field(
                name="/random",
                value="Pull a random quote — optionally filter by show, episode, or speaker.",
                inline=False,
            )
            embed.add_field(
                name="/episodes `show`",
                value="List all indexed episodes for a show.",
                inline=False,
            )
            embed.add_field(
                name="/stats",
                value=(
                    "Overview of what's indexed: episodes and duration per "
                    "show. A single show (one indexed, or picked via the "
                    "show filter) also gets its top speakers."
                ),
                inline=False,
            )
            embed.add_field(
                name="/speakers",
                value="Who speaks the most — excerpt counts and airtime per speaker.",
                inline=False,
            )
            embed.add_field(
                name="/setup *(admin)*",
                value=(
                    "Configure server defaults: model, top-k, "
                    "default source, compact mode."
                ),
                inline=False,
            )
            embed.add_field(
                name="/announcements *(admin)*",
                value=(
                    "Pick a channel where the bot posts new episodes and version "
                    "updates. Pass `off:True` to disable."
                ),
                inline=False,
            )
            if self._locked_show_names:
                embed.add_field(
                    name="/unlock *(admin)*",
                    value="Unlock a show for this server — provide the password, the bot identifies the show automatically.",
                    inline=False,
                )
                embed.add_field(
                    name="/lock *(admin)*",
                    value="Remove a show from this server.",
                    inline=False,
                )
                embed.add_field(
                    name="/changepassword *(admin)*",
                    value="Rotate the password for an already-unlocked show. New password is sent via DM.",
                    inline=False,
                )
            embed.set_footer(
                text="Use the Show context ↕ button on results to see surrounding dialogue."
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
