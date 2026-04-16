"""podcodex.bot.ui — Discord UI components (Views + Buttons)."""

from __future__ import annotations

import asyncio

import discord

from podcodex.bot.formatting import fmt_timestamp, speaker_lines, truncate_description

# ──────────────────────────────────────────────
# Episode chunk cache
# ──────────────────────────────────────────────

_chunk_cache: dict[tuple[str, str], list[dict]] = {}
_cache_lock = asyncio.Lock()
_MAX_CACHE = 64


async def _fetch_episode_chunks(store, collection: str, episode: str) -> list[dict]:
    key = (collection, episode)
    async with _cache_lock:
        if key in _chunk_cache:
            return _chunk_cache[key]

    loop = asyncio.get_running_loop()
    chunks = await loop.run_in_executor(
        None, lambda: store.load_chunks_no_embeddings(collection, episode)
    )

    async with _cache_lock:
        if len(_chunk_cache) >= _MAX_CACHE:
            oldest = next(iter(_chunk_cache))
            del _chunk_cache[oldest]
        _chunk_cache[key] = chunks

    return chunks


# ──────────────────────────────────────────────
# Transcript pagination UI
# ──────────────────────────────────────────────


def _transcript_embed(
    chunk: dict,
    pos: int,
    total: int,
    show: str,
    *,
    is_match: bool = False,
) -> discord.Embed:
    """Build a single embed for one transcript chunk."""
    description = truncate_description(speaker_lines(chunk))
    color = discord.Color.gold() if is_match else discord.Color.dark_gray()
    embed = discord.Embed(description=description, color=color)

    episode_display = chunk.get("episode_title") or chunk.get("episode", "")
    title = episode_display or "(untitled)"
    if show:
        title += f" ({show})"
    embed.title = title

    start = chunk.get("start", 0.0)
    end = chunk.get("end", 0.0)
    timed = chunk.get("timed", True)
    ts = fmt_timestamp(start, end, timed=timed)
    if ts:
        embed.add_field(name="Timestamp", value=ts, inline=True)

    marker = " ◀ matched" if is_match else ""
    embed.set_footer(text=f"Segment {pos + 1} of {total}{marker}")
    return embed


class TranscriptView(discord.ui.View):
    """Paginated view over all segments of an episode, one per page.

    Opens at the segment that was returned by the search result so users
    can immediately read forward/backward through the transcript.
    """

    def __init__(
        self,
        chunks: list[dict],
        match_pos: int,
        show: str,
    ) -> None:
        super().__init__(timeout=300)
        self._chunks = chunks
        self._match_pos = match_pos
        self._pos = match_pos
        self._show = show
        self._update_nav()

    @property
    def current_embed(self) -> discord.Embed:
        return _transcript_embed(
            self._chunks[self._pos],
            self._pos,
            len(self._chunks),
            self._show,
            is_match=(self._pos == self._match_pos),
        )

    def _update_nav(self) -> None:
        self.prev_button.disabled = self._pos == 0
        self.next_button.disabled = self._pos == len(self._chunks) - 1
        self.counter_button.label = f"{self._pos + 1} / {len(self._chunks)}"

    @discord.ui.button(label="◀", style=discord.ButtonStyle.secondary, row=0)
    async def prev_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        self._pos -= 1
        self._update_nav()
        await interaction.response.edit_message(embed=self.current_embed, view=self)

    @discord.ui.button(
        label="1 / ?", style=discord.ButtonStyle.gray, disabled=True, row=0
    )
    async def counter_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await interaction.response.defer()

    @discord.ui.button(label="▶", style=discord.ButtonStyle.secondary, row=0)
    async def next_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        self._pos += 1
        self._update_nav()
        await interaction.response.edit_message(embed=self.current_embed, view=self)


# ──────────────────────────────────────────────
# Context expand button (wired to search results)
# ──────────────────────────────────────────────


class ExpandView(discord.ui.View):
    def __init__(self, collection: str, episode: str, show: str, start: float) -> None:
        super().__init__(timeout=300)
        self.add_item(_ExpandButton(collection, episode, show, start))


class _ExpandButton(discord.ui.Button):
    def __init__(self, collection: str, episode: str, show: str, start: float) -> None:
        super().__init__(label="Show context ↕", style=discord.ButtonStyle.secondary)
        self._collection = collection
        self._episode = episode
        self._show = show
        self._start = start

    async def callback(self, interaction: discord.Interaction) -> None:
        from podcodex.bot.bot import PodCodexBot

        bot: PodCodexBot = interaction.client  # type: ignore[assignment]
        chunks = await _fetch_episode_chunks(bot.local, self._collection, self._episode)

        # Find the matched chunk by start time
        match_pos = next(
            (
                i
                for i, c in enumerate(chunks)
                if abs(c.get("start", -1) - self._start) < 0.1
            ),
            0,
        )

        view = TranscriptView(chunks, match_pos, self._show)
        await interaction.response.send_message(
            embed=view.current_embed, view=view, ephemeral=True
        )


# ──────────────────────────────────────────────
# Paginated results UI
# ──────────────────────────────────────────────


class NoExpandView(discord.ui.View):
    """Placeholder for paginated embeds that have no context button."""

    def __init__(self) -> None:
        super().__init__(timeout=300)


# ──────────────────────────────────────────────
# /ask sources reveal UI
# ──────────────────────────────────────────────


class SourcesView(discord.ui.View):
    """Attaches a 'Show sources' button to a /ask answer embed.

    Clicking reveals an ephemeral paginated view of the retrieved chunks.
    """

    def __init__(self, pages: list[tuple[discord.Embed, discord.ui.View]]) -> None:
        super().__init__(timeout=300)
        self._pages = pages

    @discord.ui.button(label="Show sources ↗", style=discord.ButtonStyle.secondary)
    async def show_sources(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        view = PaginatedResultView(self._pages)
        await interaction.response.send_message(
            embed=view.current_embed, view=view, ephemeral=True
        )


class PaginatedResultView(discord.ui.View):
    """
    Single-message paginated result display with ◀ counter ▶ buttons.
    Optionally shows a 'Show context ↕' button per page if the page has one.
    """

    def __init__(self, results: list[tuple[discord.Embed, discord.ui.View]]) -> None:
        super().__init__(timeout=300)
        self._results = results
        self._index = 0
        # _expand_item holds the dynamically swapped button (row 1)
        self._expand_item: discord.ui.Button | None = None
        self._sync_expand()
        self._update_nav()

    @property
    def current_embed(self) -> discord.Embed:
        return self._results[self._index][0]

    def _sync_expand(self) -> None:
        """Swap in the expand button for the current page, if any."""
        if self._expand_item is not None:
            self.remove_item(self._expand_item)
            self._expand_item = None

        expand_view = self._results[self._index][1]
        if expand_view.children:
            src: _ExpandButton = expand_view.children[0]  # type: ignore
            btn = _ExpandButton(src._collection, src._episode, src._show, src._start)
            btn.row = 1
            self.add_item(btn)
            self._expand_item = btn

    def _update_nav(self) -> None:
        self.prev_button.disabled = self._index == 0
        self.next_button.disabled = self._index == len(self._results) - 1
        self.counter_button.label = f"{self._index + 1} / {len(self._results)}"

    @discord.ui.button(label="◀", style=discord.ButtonStyle.secondary, row=0)
    async def prev_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        self._index -= 1
        self._sync_expand()
        self._update_nav()
        await interaction.response.edit_message(embed=self.current_embed, view=self)

    @discord.ui.button(
        label="1 / ?", style=discord.ButtonStyle.gray, disabled=True, row=0
    )
    async def counter_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await interaction.response.defer()

    @discord.ui.button(label="▶", style=discord.ButtonStyle.secondary, row=0)
    async def next_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        self._index += 1
        self._sync_expand()
        self._update_nav()
        await interaction.response.edit_message(embed=self.current_embed, view=self)
