"""podcodex.bot.ui — Discord UI components (Views + Buttons)."""

from __future__ import annotations

import asyncio

import discord

from podcodex.bot.formatting import format_context

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
# Context expand UI
# ──────────────────────────────────────────────


class ContextView(discord.ui.View):
    def __init__(
        self,
        collection: str,
        episode: str,
        show: str,
        start: float,
        neighbors: list[dict],
        n: int,
        has_more: bool,
    ) -> None:
        super().__init__(timeout=300)
        if has_more:
            self.add_item(
                _ExpandMoreButton(collection, episode, show, start, neighbors, n + 2)
            )


class _ExpandMoreButton(discord.ui.Button):
    def __init__(
        self,
        collection: str,
        episode: str,
        show: str,
        start: float,
        neighbors: list[dict],
        n: int,
    ) -> None:
        super().__init__(
            label=f"Show more ↕ (±{n})", style=discord.ButtonStyle.secondary
        )
        self._collection = collection
        self._episode = episode
        self._show = show
        self._start = start
        self._neighbors = neighbors
        self._n = n

    async def callback(self, interaction: discord.Interaction) -> None:
        content, has_more = format_context(
            self._neighbors, self._start, self._n, self._show, self._episode
        )
        view = ContextView(
            self._collection,
            self._episode,
            self._show,
            self._start,
            self._neighbors,
            n=self._n,
            has_more=has_more,
        )
        await interaction.response.edit_message(content=content, view=view)


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
        neighbors = await _fetch_episode_chunks(
            bot.local, self._collection, self._episode
        )
        content, has_more = format_context(
            neighbors, self._start, 2, self._show, self._episode
        )
        view = ContextView(
            self._collection,
            self._episode,
            self._show,
            self._start,
            neighbors,
            n=2,
            has_more=has_more,
        )
        await interaction.response.send_message(content, view=view, ephemeral=True)


# ──────────────────────────────────────────────
# Paginated results UI
# ──────────────────────────────────────────────


class NoExpandView(discord.ui.View):
    """Placeholder for paginated embeds that have no context button."""

    def __init__(self) -> None:
        super().__init__(timeout=300)


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
        # Remove previous expand item
        if self._expand_item is not None:
            self.remove_item(self._expand_item)
            self._expand_item = None

        expand_view = self._results[self._index][1]
        if expand_view.children:
            # Clone the button so it isn't bound to the other view
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
