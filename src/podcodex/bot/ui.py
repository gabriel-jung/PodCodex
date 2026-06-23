"""podcodex.bot.ui — Discord UI: persistent, stateless paginated views.

Every paginated view here is *stateless and persistent*. Page state lives in the
button custom_ids (a short hex ``search_id`` plus integer indices) and in the
:class:`~podcodex.bot.result_store.SearchCacheStore`, never on a View instance.
This survives both the old 5-minute view timeout and bot restarts: on each click
the bot reloads the refs from the cache, re-fetches the chunk text from LanceDB,
and rebuilds the page on demand. See ``result_store`` for the cache and
``bot.setup_hook`` for the ``add_dynamic_items`` registration.

custom_id grammar (':'-delimited; ``sid`` is hex so it never contains ':'):
    pcx:r:<sid>:<idx>:<p|n>          result prev / next  -> page idx∓1
    pcx:rj:<sid>                     result jump select (target index is the value)
    pcx:rx:<sid>:<idx>               expand result <idx> into its transcript
    pcx:t:<sid>:<ridx>:<pos>:<p|n>   transcript prev / next within result <ridx>
    pcx:l:<sid>:<idx>:<p|n>          list (e.g. /shows) prev / next page
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

import discord

from podcodex.bot.formatting import (
    episode_display,
    fmt_timestamp,
    score_bar,
    speaker_lines,
    truncate_description,
)
from podcodex.bot.result_store import CachedSearch, ResultRef

if TYPE_CHECKING:
    from podcodex.bot.bot import PodCodexBot

EXPIRED_MSG = (
    "⌛ These results have expired. Please re-run the search to get fresh buttons."
)
UNAVAILABLE_MSG = "This result is no longer available (the episode may have changed)."

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


def _locate(chunks: list[dict], chunk_index: int) -> int:
    """Position of the chunk with ``chunk_index``, nearest on drift, 0 if empty.

    The episode may have been re-chunked since the search ran, so the exact
    index can be gone; falling back to the nearest keeps the page rendering.
    """
    for i, c in enumerate(chunks):
        if c.get("chunk_index") == chunk_index:
            return i
    if not chunks:
        return 0
    return min(
        range(len(chunks)),
        key=lambda i: abs(chunks[i].get("chunk_index", 0) - chunk_index),
    )


async def _result_chunk(local, ref: ResultRef) -> dict | None:
    """Re-fetch the chunk a ref points to and re-attach the cached scalars.

    Returns None if the episode is gone.
    """
    chunks = await _fetch_episode_chunks(local, ref.collection, ref.episode)
    if not chunks:
        return None
    # Overlay the search-time scalars LanceDB doesn't carry.
    return {
        **chunks[_locate(chunks, ref.chunk_index)],
        "score": ref.score,
        "fuzzy_match": ref.fuzzy_match,
        "accent_match": ref.accent_match,
        "match_text": ref.match_text,
    }


# ──────────────────────────────────────────────
# Embed builders
# ──────────────────────────────────────────────


def build_result_embed(
    chunk: dict,
    rank: int,
    total: int,
    label: str,
    text: str = "",
    *,
    highlight: bool = False,
) -> discord.Embed:
    """Build a single search-result embed.

    Pure: takes a chunk dict already carrying the search-time scalars
    (``score``, ``fuzzy_match``, ``accent_match``, ``match_text``).

    ``text`` is the search/question string shown on the author line.
    ``highlight`` additionally marks it in-body (set by /exact; /search shows
    the question but does not highlight).
    """
    show = chunk.get("show", "")
    start = chunk.get("start", 0.0)
    end = chunk.get("end", 0.0)
    score = chunk.get("score", 0.0)

    description = truncate_description(
        speaker_lines(chunk, query=text if highlight else "")
    )

    if chunk.get("fuzzy_match"):
        color = discord.Color.orange()
        badge = "〜 near-typo"
    elif chunk.get("accent_match"):
        color = discord.Color.gold()
        badge = "≈ accent variant"
    else:
        color = discord.Color.blurple()
        badge = ""

    embed = discord.Embed(description=description, color=color)
    if text:
        embed.set_author(name=f'🔎 "{text}"')
    title = episode_display(chunk) or "(untitled)"
    if show:
        title += f" ({show})"
    embed.title = title
    if badge:
        embed.add_field(name="Match", value=badge, inline=True)
    timed = chunk.get("timed", True)
    ts_label = fmt_timestamp(start, end, timed=timed)
    if ts_label:
        embed.add_field(name="Timestamp", value=ts_label, inline=True)
    pub_date = (chunk.get("pub_date") or "").strip()
    if pub_date:
        embed.add_field(name="Published", value=pub_date[:10], inline=True)
    clamped = max(0.0, min(1.0, score))
    embed.add_field(
        name="Relevance", value=f"{score_bar(clamped)} {clamped:.0%}", inline=True
    )
    embed.set_footer(text=f"#{rank} of {total} • {label}")
    return embed


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

    title = episode_display(chunk) or "(untitled)"
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


_MARKDOWN_STRIP = re.compile(r"[*_`~]+")


def _option_snippet(embed: discord.Embed) -> str:
    """Build a short snippet for a jump-dropdown option description."""
    ts = ""
    for f in embed.fields:
        if f.name == "Timestamp":
            ts = f.value or ""
            break

    raw = (embed.description or "").strip()
    first_line = raw.split("\n", 1)[0] if raw else ""
    text = _MARKDOWN_STRIP.sub("", first_line).strip()

    if ts and text:
        return f"{ts} • {text}"
    return ts or text


# ──────────────────────────────────────────────
# Persistent dynamic components
# ──────────────────────────────────────────────

# Discord hard cap on Select options per component.
_JUMP_WINDOW = 25


def _counter_button(pos: int, total: int) -> discord.ui.Button:
    """Disabled label-only page counter. Never dispatches (no handler needed)."""
    return discord.ui.Button(
        label=f"{pos} / {total}",
        style=discord.ButtonStyle.gray,
        disabled=True,
        custom_id="pcx:noop",
        row=0,
    )


def _nav_button(custom_id: str, act: str, *, disabled: bool) -> discord.ui.Button:
    """The shared ◀/▶ prev-next button used by every paginated view."""
    return discord.ui.Button(
        label="◀" if act == "p" else "▶",
        style=discord.ButtonStyle.secondary,
        custom_id=custom_id,
        disabled=disabled,
        row=0,
    )


class ResultNav(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"pcx:r:(?P<sid>[0-9a-f]+):(?P<idx>\d+):(?P<act>[pn])$",
):
    """Prev/next button over a cached search's results."""

    def __init__(self, sid: str, idx: int, act: str, *, disabled: bool = False) -> None:
        super().__init__(
            _nav_button(f"pcx:r:{sid}:{idx}:{act}", act, disabled=disabled)
        )
        self.sid = sid
        self.idx = idx
        self.act = act

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["sid"], int(match["idx"]), match["act"])

    async def callback(self, interaction: discord.Interaction) -> None:
        target = self.idx - 1 if self.act == "p" else self.idx + 1
        await _render_results(interaction, self.sid, target)


class ResultJump(
    discord.ui.DynamicItem[discord.ui.Select],
    template=r"pcx:rj:(?P<sid>[0-9a-f]+)$",
):
    """Jump-to-result dropdown. Target index is the selected option value."""

    def __init__(self, sid: str, cached: CachedSearch | None, index: int) -> None:
        super().__init__(self._build_select(sid, cached, index))
        self.sid = sid

    @staticmethod
    def _build_select(
        sid: str, cached: CachedSearch | None, index: int
    ) -> discord.ui.Select:
        options: list[discord.SelectOption] = []
        placeholder = "Jump to result…"
        if cached is not None:
            n = len(cached.refs)
            window = min(_JUMP_WINDOW, n)
            start = max(0, min(n - window, index - window // 2))
            end = start + window
            for i in range(start, end):
                ref = cached.refs[i]
                label = f"#{i + 1} • {ref.episode_title or ref.episode}"
                options.append(
                    discord.SelectOption(
                        label=label[:100],
                        value=str(i),
                        default=(i == index),
                    )
                )
            placeholder = f"Jump to result… ({index + 1} / {n})"
            if n > window:
                placeholder = f"Jump (showing {start + 1}–{end} of {n})"
        if not options:  # reconstructed on click; options are irrelevant then
            options = [discord.SelectOption(label="—", value="0")]
        return discord.ui.Select(
            placeholder=placeholder[:150],
            options=options,
            custom_id=f"pcx:rj:{sid}",
            row=2,
        )

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["sid"], None, 0)

    async def callback(self, interaction: discord.Interaction) -> None:
        target = int(self.item.values[0])
        await _render_results(interaction, self.sid, target)


class ExpandResult(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"pcx:rx:(?P<sid>[0-9a-f]+):(?P<idx>\d+)$",
):
    """Open the full transcript around result <idx>, as an ephemeral message."""

    def __init__(self, sid: str, idx: int) -> None:
        super().__init__(
            discord.ui.Button(
                label="Show context ↕",
                style=discord.ButtonStyle.secondary,
                custom_id=f"pcx:rx:{sid}:{idx}",
                row=1,
            )
        )
        self.sid = sid
        self.idx = idx

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["sid"], int(match["idx"]))

    async def callback(self, interaction: discord.Interaction) -> None:
        await _render_transcript(interaction, self.sid, self.idx, None, ephemeral=True)


class TranscriptNav(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"pcx:t:(?P<sid>[0-9a-f]+):(?P<ridx>\d+):(?P<pos>\d+):(?P<act>[pn])$",
):
    """Prev/next button stepping through the segments of result <ridx>'s episode."""

    def __init__(
        self, sid: str, ridx: int, pos: int, act: str, *, disabled: bool = False
    ) -> None:
        super().__init__(
            _nav_button(f"pcx:t:{sid}:{ridx}:{pos}:{act}", act, disabled=disabled)
        )
        self.sid = sid
        self.ridx = ridx
        self.pos = pos
        self.act = act

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["sid"], int(match["ridx"]), int(match["pos"]), match["act"])

    async def callback(self, interaction: discord.Interaction) -> None:
        target = self.pos - 1 if self.act == "p" else self.pos + 1
        await _render_transcript(interaction, self.sid, self.ridx, target)


class ListNav(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"pcx:l:(?P<sid>[0-9a-f]+):(?P<idx>\d+):(?P<act>[pn])$",
):
    """Prev/next button over a cached list of verbatim embeds (e.g. /shows)."""

    def __init__(self, sid: str, idx: int, act: str, *, disabled: bool = False) -> None:
        super().__init__(
            _nav_button(f"pcx:l:{sid}:{idx}:{act}", act, disabled=disabled)
        )
        self.sid = sid
        self.idx = idx
        self.act = act

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["sid"], int(match["idx"]), match["act"])

    async def callback(self, interaction: discord.Interaction) -> None:
        target = self.idx - 1 if self.act == "p" else self.idx + 1
        await _render_list(interaction, self.sid, target)


# ──────────────────────────────────────────────
# Page builders + renderers
# ──────────────────────────────────────────────


async def build_list_view(
    client: PodCodexBot, sid: str, index: int
) -> tuple[discord.Embed, discord.ui.View] | None:
    """Build the (embed, view) for verbatim-embed page ``index`` of ``sid``."""
    cached = client.results.load(sid)
    if cached is None or not cached.embeds:
        return None
    n = len(cached.embeds)
    index = max(0, min(index, n - 1))
    embed = discord.Embed.from_dict(cached.embeds[index])
    view = discord.ui.View(timeout=None)
    view.add_item(ListNav(sid, index, "p", disabled=index == 0))
    view.add_item(_counter_button(index + 1, n))
    view.add_item(ListNav(sid, index, "n", disabled=index == n - 1))
    return embed, view


async def build_results_view(
    client: PodCodexBot, sid: str, index: int
) -> tuple[discord.Embed, discord.ui.View] | None:
    """Build the (embed, view) for result page ``index`` of search ``sid``."""
    cached = client.results.load(sid)
    if cached is None:
        return None
    n = len(cached.refs)
    if n == 0:
        return None
    index = max(0, min(index, n - 1))
    chunk = await _result_chunk(client.local, cached.refs[index])
    if chunk is None:
        return None

    # /exact highlights the literal query in-text; /search (semantic) shows the
    # question on the author line but does not highlight.
    embed = build_result_embed(
        chunk,
        index + 1,
        n,
        cached.label,
        cached.query,
        highlight=(cached.kind == "exact"),
    )

    view = discord.ui.View(timeout=None)
    view.add_item(ResultNav(sid, index, "p", disabled=index == 0))
    view.add_item(_counter_button(index + 1, n))
    view.add_item(ResultNav(sid, index, "n", disabled=index == n - 1))
    if chunk.get("episode"):
        view.add_item(ExpandResult(sid, index))
    if n > 1:
        view.add_item(ResultJump(sid, cached, index))
    return embed, view


async def build_transcript_view(
    client: PodCodexBot, sid: str, ridx: int, pos: int | None
) -> tuple[discord.Embed, discord.ui.View] | None:
    """Build the transcript (embed, view) for result ``ridx`` at segment ``pos``.

    ``pos=None`` opens at the matched segment (the result's own chunk).
    """
    cached = client.results.load(sid)
    if cached is None or not (0 <= ridx < len(cached.refs)):
        return None
    ref = cached.refs[ridx]
    chunks = await _fetch_episode_chunks(client.local, ref.collection, ref.episode)
    if not chunks:
        return None

    match_pos = _locate(chunks, ref.chunk_index)
    if pos is None:
        pos = match_pos
    pos = max(0, min(pos, len(chunks) - 1))
    show = chunks[pos].get("show", "")

    embed = _transcript_embed(
        chunks[pos], pos, len(chunks), show, is_match=(pos == match_pos)
    )
    view = discord.ui.View(timeout=None)
    view.add_item(TranscriptNav(sid, ridx, pos, "p", disabled=pos == 0))
    view.add_item(_counter_button(pos + 1, len(chunks)))
    view.add_item(TranscriptNav(sid, ridx, pos, "n", disabled=pos == len(chunks) - 1))
    return embed, view


async def _respond(
    interaction: discord.Interaction,
    built: tuple[discord.Embed, discord.ui.View] | None,
    *,
    ephemeral: bool = False,
    miss_msg: str = EXPIRED_MSG,
) -> None:
    """Edit the message to the built page, or report a cache miss."""
    if built is None:
        await interaction.response.send_message(miss_msg, ephemeral=True)
        return
    embed, view = built
    if ephemeral:
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
    else:
        await interaction.response.edit_message(embed=embed, view=view)


async def _render_results(
    interaction: discord.Interaction, sid: str, index: int
) -> None:
    client: PodCodexBot = interaction.client  # type: ignore[assignment]
    await _respond(interaction, await build_results_view(client, sid, index))


async def _render_list(interaction: discord.Interaction, sid: str, index: int) -> None:
    client: PodCodexBot = interaction.client  # type: ignore[assignment]
    await _respond(interaction, await build_list_view(client, sid, index))


async def _render_transcript(
    interaction: discord.Interaction,
    sid: str,
    ridx: int,
    pos: int | None,
    *,
    ephemeral: bool = False,
) -> None:
    client: PodCodexBot = interaction.client  # type: ignore[assignment]
    await _respond(
        interaction,
        await build_transcript_view(client, sid, ridx, pos),
        ephemeral=ephemeral,
        miss_msg=EXPIRED_MSG if ephemeral else UNAVAILABLE_MSG,
    )


# Registered with bot.add_dynamic_items in setup_hook.
DYNAMIC_ITEMS = (ResultNav, ResultJump, ExpandResult, TranscriptNav, ListNav)
