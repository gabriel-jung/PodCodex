"""podcodex.bot.formatting — Pure display and search-merge helpers."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from podcodex.core._utils import episode_display, humanize_stem  # noqa: F401 — re-exported
from podcodex.rag.index_store import fold_text


def format_filter_suffix(
    *,
    episode: str | None = None,
    speaker: str | None = None,
    source: str | None = None,
) -> str:
    """Return ``" (filters: episode=`X`, speaker=`Y`)"`` or ``""`` when empty."""
    parts: list[str] = []
    if episode:
        parts.append(f"episode=`{episode}`")
    if speaker:
        parts.append(f"speaker=`{speaker}`")
    if source:
        parts.append(f"source=`{source}`")
    return f" (filters: {', '.join(parts)})" if parts else ""


if TYPE_CHECKING:
    import discord

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

_MAX_CHARS = 3900  # context sent as embed description (Discord limit: 4096)
_MAX_DESC_CHARS = 4000  # result / answer embed description guard
COOLDOWN_SECONDS = 5.0

# ──────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def fmt_duration(seconds: float) -> str:
    """Humanize a duration for totals: ``"46h 12m"``, ``"59m"``, ``"214h"``.

    Rounds to the nearest minute (clock precision belongs to
    :func:`fmt_time`; totals read better coarse). Zero-minute components
    drop out; a zero total is ``"0m"``.
    """
    minutes = round(seconds / 60)
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"


def fmt_timestamp(start: float, end: float, *, timed: bool = True) -> str | None:
    """Format a timestamp range for display.

    Returns None when there is no meaningful timestamp to show
    (untimed episode with both values at zero).
    """
    if start == 0.0 and end == 0.0:
        return None
    if not timed:
        return f"~{start:.0f}% → ~{end:.0f}%"
    return f"{fmt_time(start)} → {fmt_time(end)}"


_RAW_SPEAKER_RE = re.compile(r"^SPEAKER_0*(\d+)$", re.IGNORECASE)


def display_speaker(name: str | None) -> str:
    """Render a speaker label for humans, never leaking raw diarization tags.

    ``SPEAKER_01`` becomes ``Speaker 1``; an empty/blank label (YouTube
    subtitle imports without ``<v>`` tags leave ``""``) becomes ``Speaker``.
    Any real name passes through unchanged.
    """
    name = (name or "").strip()
    if not name:
        return "Speaker"
    m = _RAW_SPEAKER_RE.match(name)
    if m:
        return f"Speaker {int(m.group(1))}"
    return name


def speaker(chunk: dict) -> str:
    return display_speaker(chunk.get("speaker") or chunk.get("dominant_speaker"))


def is_http_url(value: str | None) -> bool:
    """True when *value* is a non-empty http(s) URL (Discord's link requirement)."""
    return (value or "").strip().startswith(("http://", "https://"))


def set_chunk_thumbnail(embed: "discord.Embed", chunk: dict) -> None:
    """Set the embed thumbnail from a chunk's ``artwork_url`` when it's a real URL."""
    art = (chunk.get("artwork_url") or "").strip()
    if is_http_url(art):
        embed.set_thumbnail(url=art)


_MONTHS = (
    "",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def pub_month(pub_date: str | None) -> str:
    """Format an ISO ``pub_date`` as ``"Apr 2026"``, or ``""`` when unusable.

    Locale-free: reads the ``YYYY-MM`` prefix directly rather than parsing a
    full datetime, so it never trips on the timezone/offset tails that ride
    along on normalized RSS dates.
    """
    p = (pub_date or "").strip()
    if len(p) >= 7 and p[4] == "-":
        try:
            month = int(p[5:7])
        except ValueError:
            return ""
        if 1 <= month <= 12:
            return f"{_MONTHS[month]} {p[:4]}"
    return ""


def pub_day(pub_date: str | None) -> str:
    """Format an ISO ``pub_date`` as ``"8 May 2026"``, or fall back.

    Falls back to :func:`pub_month` ("May 2026") when the day part is
    missing or unparsable, and to ``""`` when even that fails.
    """
    p = (pub_date or "").strip()
    if len(p) >= 10 and p[4] == "-" and p[7] == "-":
        try:
            day = int(p[8:10])
            month = int(p[5:7])
        except ValueError:
            return pub_month(p)
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{day} {_MONTHS[month]} {p[:4]}"
    return pub_month(p)


def count_occurrences(text: str, query: str) -> int:
    """Count accent- and case-insensitive occurrences of *query* in *text*."""
    if not query:
        return 0
    return fold_text(text).count(fold_text(query))


def highlight(text: str, query: str) -> str:
    """Case-insensitive highlight: wrap all occurrences of *query* in bold."""

    if not query:
        return text
    escaped = re.escape(query)
    return re.sub(f"({escaped})", r"**__\1__**", text, flags=re.IGNORECASE)


def speaker_lines(chunk: dict, query: str = "") -> str:
    """Format chunk text with per-turn speaker labels when available.

    Consecutive same-speaker turns are merged into a single block via
    :func:`merge_display_turns`, so readers see one speaker label and
    one paragraph per contiguous run.

    If the chunk carries a ``match_text`` (from /exact's accent/fuzzy tiers),
    highlight that exact substring so the user sees which span matched.
    Otherwise fall back to highlighting the raw query.
    """
    from podcodex.core._utils import merge_display_turns

    turns: list[dict] = chunk.get("speakers") or []
    mark = chunk.get("match_text") or query
    if not turns:
        text = chunk.get("text", "")
        return highlight(text, mark) if mark else text
    merged = merge_display_turns(turns)
    lines = []
    for t in merged:
        spk = display_speaker(t.get("speaker"))
        start = t.get("start", 0)
        ts_part = f"({fmt_time(start)})" if start else ""
        text = highlight(t.get("text", ""), mark) if mark else t.get("text", "")
        sep = " " if ts_part else ""
        lines.append(f"**{spk}**{sep}{ts_part}: {text}")
    return "\n".join(lines)


def score_bar(score: float, width: int = 8) -> str:
    clamped = max(0.0, min(1.0, score))
    filled = round(clamped * width)
    return "█" * filled + "░" * (width - filled)


def safe_truncate(text: str, max_chars: int = _MAX_CHARS) -> tuple[str, bool]:
    """
    Truncate at the last whitespace before max_chars.
    Returns (text, was_truncated) to let callers suppress 'Show more'.
    Never cuts mid-word or inside a markdown ** span.
    """
    if len(text) <= max_chars:
        return text, False
    cut = text.rfind(" ", 0, max_chars)
    cut = cut if cut != -1 else max_chars
    return text[:cut] + "\n\n*…(truncated)*", True


def truncate_description(text: str) -> str:
    """Truncate embed description to Discord's 4096-char limit."""
    return safe_truncate(text, _MAX_DESC_CHARS)[0]


# ──────────────────────────────────────────────
# Per-user cooldown
# ──────────────────────────────────────────────


class CooldownManager:
    """Simple in-memory per-user cooldown tracker."""

    def __init__(self, seconds: float = COOLDOWN_SECONDS) -> None:
        self._seconds = seconds
        self._last_used: dict[int, float] = {}

    def check(self, user_id: int, seconds: float | None = None) -> float:
        """Return 0.0 if the user may proceed, or remaining wait time.

        *seconds* overrides the instance default when provided.
        """
        now = time.monotonic()
        last = self._last_used.get(user_id, 0.0)
        duration = seconds if seconds is not None else self._seconds
        remaining = duration - (now - last)
        return max(0.0, remaining)

    def consume(self, user_id: int) -> None:
        """Record that the user just made a request."""
        self._last_used[user_id] = time.monotonic()


# ──────────────────────────────────────────────
# Compact embed
# ──────────────────────────────────────────────

_COMPACT_TEXT_MAX = 200
# Total-embed character budget: Discord hard-caps an embed at 6000 chars
# (title + description + all field names/values + footer) and 400s above it.
# Reserve headroom for the footer, which is set after the field loop.
_COMPACT_EMBED_BUDGET = 5800


def build_compact_embed(
    results: list[tuple[dict, str]],
    label: str,
    query: str = "",
    question: str = "",
) -> "discord.Embed":
    """Build a single embed with one field per result.

    Capped at 25 fields (Discord's per-embed field limit) AND at
    ``_COMPACT_EMBED_BUDGET`` total characters: Discord rejects any message
    whose embed exceeds 6000 chars with HTTP 400, so rows are dropped from
    the tail once the budget is reached. Callers can read ``len(.fields)``
    for the real shown count.
    """
    import discord

    q = question or query
    title = f"🔎 {label}"
    embed = discord.Embed(
        title=title,
        description=f'*"{q}"*' if q else None,
        color=discord.Color.blurple(),
    )
    for i, (chunk, _col) in enumerate(results[:25], 1):
        show = chunk.get("show", "")
        episode = episode_display(chunk)
        score = chunk.get("score", 0.0)
        start = chunk.get("start", 0.0)
        text = chunk.get("text", "")
        if len(text) > _COMPACT_TEXT_MAX:
            cut = text.rfind(" ", 0, _COMPACT_TEXT_MAX)
            text = text[: cut if cut != -1 else _COMPACT_TEXT_MAX] + "…"
        if query:
            text = highlight(text, query)

        name = f"#{i} {episode}"
        if show:
            name += f" ({show})"
        end = chunk.get("end", 0.0)
        timed = chunk.get("timed", True)
        ts_label = fmt_timestamp(start, end, timed=timed)
        ts_part = f" · {ts_label}" if ts_label else ""
        value = (
            f"{speaker(chunk)}{ts_part} · "
            f"{score_bar(score)} {min(1.0, score):.0%}\n"
            f'*"{text}"*'
        )
        if len(embed) + len(name) + len(value) > _COMPACT_EMBED_BUDGET:
            break
        embed.add_field(name=name, value=value, inline=False)

    n_results = len(results)
    if query:
        total_occ = sum(count_occurrences(c.get("text", ""), query) for c, _ in results)
        footer = (
            f"{n_results} excerpt{'s' if n_results != 1 else ''} · "
            f"{total_occ} mention{'s' if total_occ != 1 else ''}"
        )
    else:
        footer = f"{n_results} result{'s' if n_results != 1 else ''}"
    embed.set_footer(text=footer)
    return embed
