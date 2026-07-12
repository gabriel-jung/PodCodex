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
    build_compact_embed,
    display_speaker,
    episode_display,
    fmt_duration,
    fmt_time,
    fmt_timestamp,
    is_http_url,
    pub_day,
    pub_month,
    score_bar,
    set_chunk_thumbnail,
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
    footer_extra: str = "",
) -> discord.Embed:
    """Build a single search-result embed — lean by default.

    Pure: takes a chunk dict already carrying the search-time scalars
    (``score``, ``fuzzy_match``, ``accent_match``, ``match_text``).

    The card leads with the quote and anchors it to the episode (show on the
    author line, episode title as the title, one quiet ``time · month`` line
    under the body). The engine numbers — relevance, full range, published
    date, match tier, search label — all live behind the Details button
    (:func:`build_details_embed`), so a reader is never shown telemetry.

    ``text`` is the search string; ``highlight`` marks it in-body (set by
    /exact). /search passes the question through for highlighting context but
    leaves ``highlight`` off.
    """
    show = chunk.get("show", "")
    start = chunk.get("start", 0.0)

    description = truncate_description(
        speaker_lines(chunk, query=text if highlight else "")
    )

    # Match tier only tints the card now; the badge text moved to Details.
    if chunk.get("fuzzy_match"):
        color = discord.Color.orange()
    elif chunk.get("accent_match"):
        color = discord.Color.gold()
    else:
        color = discord.Color.blurple()

    # One quiet meta line under the quote: start time · month. Plain text —
    # Discord's `-#` subtext markdown does not render inside embeds.
    meta_bits: list[str] = []
    if start:
        meta_bits.append(f"🕐 {fmt_time(start)}")
    month = pub_month(chunk.get("pub_date"))
    if month:
        meta_bits.append(month)
    if meta_bits:
        description = f"{description}\n\n{' · '.join(meta_bits)}"

    embed = discord.Embed(description=description, color=color)
    if show:
        embed.set_author(name=show)
    embed.title = episode_display(chunk) or "(untitled)"
    set_chunk_thumbnail(embed, chunk)
    # ``footer_extra`` carries /exact's human total ("2444 matches"); /search
    # passes nothing (its label is engine telemetry, which lives in Details).
    if footer_extra:
        embed.set_footer(text=f"{rank} of {total} excerpts · {footer_extra}")
    else:
        embed.set_footer(text=f"{rank} of {total}")
    return embed


def build_listen_button(chunk: dict) -> discord.ui.Button | None:
    """A source-routed link button to reach the audio, or None when unavailable.

    YouTube episodes (``youtube_id`` present) get a timestamped watch link that
    jumps to the moment; RSS episodes (``audio_url``) get a plain 'listen to
    episode' link (raw-audio seek isn't reliable, so no timestamp). Local
    imports carry neither and get no button.
    """
    yt = (chunk.get("youtube_id") or "").strip()
    start = int(chunk.get("start", 0) or 0)
    if yt:
        label = (
            f"▶ Watch on YouTube · {fmt_time(start)}" if start else "▶ Watch on YouTube"
        )
        return discord.ui.Button(
            style=discord.ButtonStyle.link,
            url=f"https://www.youtube.com/watch?v={yt}&t={start}s",
            label=label,
            row=1,
        )
    audio = (chunk.get("audio_url") or "").strip()
    if is_http_url(audio):
        return discord.ui.Button(
            style=discord.ButtonStyle.link,
            url=audio,
            label="🎧 Listen to episode",
            row=1,
        )
    return None


def build_details_embed(chunk: dict, label: str) -> discord.Embed:
    """Build the ephemeral 'Details' card — the engine numbers, opt-in.

    Everything the lean result card deliberately drops: relevance score, the
    full timestamp range, publication date, match tier, and the search label
    (``α`` / model). Sent per-user so opening it never mutates the shared
    public result message.
    """
    show = chunk.get("show", "")
    start = chunk.get("start", 0.0)
    end = chunk.get("end", 0.0)
    score = chunk.get("score", 0.0)

    embed = discord.Embed(
        title=episode_display(chunk) or "(untitled)",
        color=discord.Color.dark_gray(),
    )
    if show:
        embed.set_author(name=show)

    clamped = max(0.0, min(1.0, score))
    embed.add_field(
        name="Relevance", value=f"{score_bar(clamped)} {clamped:.0%}", inline=True
    )
    timed = chunk.get("timed", True)
    ts_label = fmt_timestamp(start, end, timed=timed)
    if ts_label:
        embed.add_field(name="Full range", value=ts_label, inline=True)
    pub_date = (chunk.get("pub_date") or "").strip()
    if pub_date:
        embed.add_field(name="Published", value=pub_date[:10], inline=True)
    if chunk.get("fuzzy_match"):
        embed.add_field(name="Match", value="〜 near-typo", inline=True)
    elif chunk.get("accent_match"):
        embed.add_field(name="Match", value="≈ accent variant", inline=True)
    if label:
        embed.add_field(name="Search", value=label, inline=True)
    return embed


def build_episodes_embeds(
    show: str,
    ep_stats: list[dict],
    footer: str,
    artwork_url: str = "",
) -> list[discord.Embed]:
    """Build the paged `/episodes` embeds: 10 episodes per page, newest first.

    ``ep_stats`` come from ``IndexStore.get_episode_stats``; ordering happens
    here: broadcast/episode number descending when present (the number is the
    show's canonical order, even though it is not displayed), ``pub_date``
    as fallback, unnumbered-undated episodes sink to the bottom.
    ``artwork_url`` is the channel-level show art (from the collection row);
    empty or non-http means no thumbnail, nothing is invented.
    """

    def _order(ep: dict):
        number = ep.get("broadcast_number") or ep.get("episode_number")
        return (
            number is not None,
            number if number is not None else 0,
            ep.get("pub_date") or "",
            ep.get("episode", ""),
        )

    ep_stats = sorted(ep_stats, key=_order, reverse=True)
    pages_data = [ep_stats[i : i + 10] for i in range(0, len(ep_stats), 10)]

    embeds: list[discord.Embed] = []
    for page in pages_data:
        embed = discord.Embed(title=f"🎙 {show}", color=discord.Color.blurple())
        if is_http_url(artwork_url):
            embed.set_thumbnail(url=artwork_url.strip())
        for ep in page:
            # Plain RSS title: feeds usually number their titles already, so
            # a broadcast_number prefix would just duplicate it.
            name = episode_display(ep)

            # Cap the roster so 10 fields stay under Discord's 6000-char
            # total embed budget even on heavily-diarized episodes.
            roster = [display_speaker(s) for s in ep.get("speakers", [])]
            if len(roster) > 5:
                roster = roster[:5] + [f"+{len(roster) - 5}"]
            speakers = ", ".join(roster) or "—"
            date = pub_day(ep.get("pub_date"))
            value = " · ".join(
                b for b in (date, speakers, f"`{fmt_time(ep['duration'])}`") if b
            )
            embed.add_field(name=name[:120], value=value[:400], inline=False)
        embed.set_footer(text=footer)
        embeds.append(embed)
    return embeds


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'s' if n != 1 else ''}"


def build_stats_embed(
    per_show: dict[str, list[dict]],
    speakers: list[dict],
    artwork_url: str = "",
) -> discord.Embed:
    """Build the `/stats` overview embed: totals, top speakers, per-show lines.

    ``per_show`` maps show name to its ``IndexStore.get_episode_stats`` rows;
    ``speakers`` is the ranked list from ``speaker_stats_multi``. The caller
    passes it only for a single-show scope (and it may still be empty:
    subtitle-only indexes carry no attribution); no list, no 🎤 line.
    Shows sort newest-episode first; shows with no ``pub_date`` at all sink
    to the bottom alphabetically and show no "newest" (nothing is invented).
    ``artwork_url`` is set by the caller only when the scope is one show.
    """
    total_eps = sum(len(rows) for rows in per_show.values())
    total_dur = sum(r.get("duration", 0.0) for rows in per_show.values() for r in rows)

    header = [
        f"{_plural(len(per_show), 'show')} · {_plural(total_eps, 'episode')} · "
        f"{fmt_duration(total_dur)} indexed"
    ]

    if speakers:
        ranked = sorted(
            speakers, key=lambda s: s.get("total_duration", 0.0), reverse=True
        )
        parts = [
            f"{display_speaker(s.get('speaker'))} "
            f"({fmt_duration(s.get('total_duration', 0.0))})"
            for s in ranked[:3]
        ]
        rest = len(ranked) - 3
        tail = f", and {_plural(rest, 'other')}" if rest > 0 else ""
        header.append(f"🎤 {', '.join(parts)}{tail}")

    def _newest(rows: list[dict]) -> str:
        return max((r.get("pub_date") or "" for r in rows), default="")

    # Name-ascending first, then a stable date-descending pass, so shows
    # sharing a newest date stay alphabetical.
    dated = sorted(s for s in per_show if _newest(per_show[s]))
    dated.sort(key=lambda s: _newest(per_show[s]), reverse=True)
    dateless = sorted(s for s in per_show if not _newest(per_show[s]))

    blocks: list[str] = []
    for show in [*dated, *dateless]:
        rows = per_show[show]
        meta = f"{_plural(len(rows), 'episode')} · " + fmt_duration(
            sum(r.get("duration", 0.0) for r in rows)
        )
        day = pub_day(_newest(rows))
        if day:
            meta += f" · newest {day}"
        blocks.append(f"🎙 **{show}**\n{meta}")

    embed = discord.Embed(
        title="📊 PodCodex Index",
        description="\n".join(header) + "\n\n" + "\n".join(blocks),
        color=discord.Color.blurple(),
    )
    if is_http_url(artwork_url):
        embed.set_thumbnail(url=artwork_url.strip())
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
    embed.set_footer(text=f"{pos + 1} of {total}{marker}")
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


class ResultDetails(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"pcx:rd:(?P<sid>[0-9a-f]+):(?P<idx>\d+)$",
):
    """Open the engine numbers for result <idx> as an ephemeral card.

    Ephemeral so a viewer opening details never edits the shared public
    result message for everyone else.
    """

    def __init__(self, sid: str, idx: int) -> None:
        super().__init__(
            discord.ui.Button(
                label="Details",
                style=discord.ButtonStyle.secondary,
                custom_id=f"pcx:rd:{sid}:{idx}",
                row=1,
            )
        )
        self.sid = sid
        self.idx = idx

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["sid"], int(match["idx"]))

    async def callback(self, interaction: discord.Interaction) -> None:
        await _render_details(interaction, self.sid, self.idx)


class ViewSwitch(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"pcx:vs:(?P<sid>[0-9a-f]+):(?P<mode>[cl])$",
):
    """Toggle a cached search between the paged card and the compact list.

    ``mode`` is the *target* view: ``c`` renders the card (page 0), ``l`` the
    list. /search opens on the card, /exact on the list; either can flip.
    """

    def __init__(self, sid: str, mode: str) -> None:
        super().__init__(
            discord.ui.Button(
                label="Full view" if mode == "c" else "List view",
                style=discord.ButtonStyle.secondary,
                custom_id=f"pcx:vs:{sid}:{mode}",
                row=1,
            )
        )
        self.sid = sid
        self.mode = mode

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["sid"], match["mode"])

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.mode == "c":
            await _render_results(interaction, self.sid, 0)
        else:
            await _render_list_compact(interaction, self.sid)


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
        footer_extra=cached.label if cached.kind == "exact" else "",
    )

    view = discord.ui.View(timeout=None)
    view.add_item(ResultNav(sid, index, "p", disabled=index == 0))
    view.add_item(_counter_button(index + 1, n))
    view.add_item(ResultNav(sid, index, "n", disabled=index == n - 1))
    listen = build_listen_button(chunk)
    if listen is not None:
        view.add_item(listen)
    if chunk.get("episode"):
        view.add_item(ExpandResult(sid, index))
    view.add_item(ResultDetails(sid, index))
    if n > 1:
        view.add_item(ViewSwitch(sid, "l"))
        view.add_item(ResultJump(sid, cached, index))
    return embed, view


async def build_compact_view(
    client: PodCodexBot, sid: str
) -> tuple[discord.Embed, discord.ui.View] | None:
    """Build the single-embed compact list for search ``sid`` (up to 25 rows).

    Re-fetches each ref's chunk text on demand (same path as the card), so the
    list is persistent and restart-safe like every other view. /exact opens
    here; a 'Full view' button flips to the paged reader.
    """
    cached = client.results.load(sid)
    if cached is None or not cached.refs:
        return None
    chunks: list[dict] = []
    for ref in cached.refs[:25]:
        chunk = await _result_chunk(client.local, ref)
        if chunk is not None:
            chunks.append(chunk)
    if not chunks:
        return None

    # /exact highlights the literal query; /search shows the question, no highlight.
    is_exact = cached.kind == "exact"
    # /exact's label is already human ("12 matches"); /search's carries the
    # α/model telemetry, which belongs in Details — not in the list title.
    title_label = cached.label if is_exact else "Search"
    embed = build_compact_embed(
        [(c, "") for c in chunks],
        title_label,
        query=cached.query if is_exact else "",
        question="" if is_exact else cached.query,
    )
    total = len(cached.refs)
    # The embed may show fewer rows than fetched: 25-field cap plus the
    # 6000-char total budget both trim from the tail.
    shown = len(embed.fields)
    if total > shown:
        # Never hide the cap: /exact is uncapped, the list shows a page at most.
        # For /exact, repeat the human match total next to the excerpt count
        # (the label is telemetry for /search, so it stays out of that footer).
        extra = f" · {cached.label}" if is_exact and cached.label else ""
        embed.set_footer(
            text=f"Showing {shown} of {total} excerpts{extra} · "
            "Full view pages through all"
        )
    view = discord.ui.View(timeout=None)
    view.add_item(ViewSwitch(sid, "c"))
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


async def _render_list_compact(interaction: discord.Interaction, sid: str) -> None:
    client: PodCodexBot = interaction.client  # type: ignore[assignment]
    await _respond(interaction, await build_compact_view(client, sid))


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


async def _render_details(interaction: discord.Interaction, sid: str, idx: int) -> None:
    """Send the ephemeral Details card for result ``idx`` of search ``sid``."""
    client: PodCodexBot = interaction.client  # type: ignore[assignment]
    cached = client.results.load(sid)
    if cached is None or not (0 <= idx < len(cached.refs)):
        await interaction.response.send_message(EXPIRED_MSG, ephemeral=True)
        return
    chunk = await _result_chunk(client.local, cached.refs[idx])
    if chunk is None:
        await interaction.response.send_message(UNAVAILABLE_MSG, ephemeral=True)
        return
    await interaction.response.send_message(
        embed=build_details_embed(chunk, cached.label), ephemeral=True
    )


# Registered with bot.add_dynamic_items in setup_hook.
DYNAMIC_ITEMS = (
    ResultNav,
    ResultJump,
    ExpandResult,
    ResultDetails,
    ViewSwitch,
    TranscriptNav,
    ListNav,
)
