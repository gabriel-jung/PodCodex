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

_MAX_CONTEXT_N = 8
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


def speaker(chunk: dict) -> str:
    return chunk.get("speaker") or chunk.get("dominant_speaker") or "Unknown"


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

    Consecutive turns from the same speaker only render the speaker label
    on the first line — subsequent turns keep just their timestamp and
    text. When the speaker changes, the label is re-emitted. This avoids
    the visual noise of seeing ``UNKNOWN`` repeated for every turn in
    single-speaker (or undiarized) chunks.

    If the chunk carries a ``match_text`` (from /exact's accent/fuzzy tiers),
    highlight that exact substring so the user sees which span matched.
    Otherwise fall back to highlighting the raw query.
    """
    turns: list[dict] = chunk.get("speakers") or []
    mark = chunk.get("match_text") or query
    if not turns:
        text = chunk.get("text", "")
        return highlight(text, mark) if mark else text
    lines = []
    prev_spk: str | None = None
    for t in turns:
        spk = t.get("speaker", "Unknown")
        start = t.get("start", 0)
        ts_part = f"({fmt_time(start)})" if start else ""
        text = highlight(t.get("text", ""), mark) if mark else t.get("text", "")
        if spk != prev_spk:
            sep = " " if ts_part else ""
            lines.append(f"**{spk}**{sep}{ts_part}: {text}")
            prev_spk = spk
        else:
            prefix = f"{ts_part} " if ts_part else ""
            lines.append(f"{prefix}{text}")
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
# Context formatting
# ──────────────────────────────────────────────


def _expand_turns(chunk: dict, bold: bool = False) -> list[str]:
    """Expand a chunk into individual speaker turns.

    If the chunk has a ``speakers`` list (semantic chunks), each turn is
    rendered separately.  Otherwise fall back to a single line with the
    chunk-level speaker.
    """
    turns: list[dict] = chunk.get("speakers") or []
    if not turns:
        spk = speaker(chunk)
        start = chunk.get("start", 0)
        ts_part = f" · {fmt_time(start)}" if start else ""
        text = chunk.get("text", "")
        if bold:
            return [f"**▶ {spk}**{ts_part}\n**{text}**"]
        return [f"*{spk}*{ts_part}\n{text}"]

    lines: list[str] = []
    for t in turns:
        spk = t.get("speaker", "Unknown")
        start = t.get("start", 0)
        ts_part = f" · {fmt_time(start)}" if start else ""
        text = t.get("text", "")
        if bold:
            lines.append(f"**▶ {spk}**{ts_part}\n**{text}**")
        else:
            lines.append(f"*{spk}*{ts_part}\n{text}")
    return lines


def format_context(
    neighbors: list[dict], start: float, n: int, show: str, episode: str
) -> tuple[str, bool]:
    """
    Format transcript context around a chunk at `start`.
    Returns (markdown_text, has_more) where has_more signals
    whether a 'Show more' button makes sense.
    """
    pos = next(
        (i for i, c in enumerate(neighbors) if abs(c.get("start", -1) - start) < 0.1),
        None,
    )
    if pos is None:
        return "Could not locate this chunk in the episode.", False

    lo = max(0, pos - n)
    hi = min(len(neighbors), pos + n + 1)
    has_more = (lo > 0 or hi < len(neighbors)) and n < _MAX_CONTEXT_N

    before = neighbors[lo:pos]
    current = neighbors[pos]
    after = neighbors[pos + 1 : hi]

    ep_display = (
        neighbors[0].get("episode_title") if neighbors else ""
    ) or humanize_stem(episode)
    header = f"**{show} — {ep_display}**" if (show or ep_display) else "*Context*"
    lines = [header + f" · ±{n} turns\n"]

    for c in before:
        lines.extend(_expand_turns(c, bold=False))

    lines.extend(_expand_turns(current, bold=True))

    for c in after:
        lines.extend(_expand_turns(c, bold=False))

    content, truncated = safe_truncate("\n\n".join(lines))
    return content, has_more and not truncated


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


def build_compact_embed(
    results: list[tuple[dict, str]],
    label: str,
    query: str = "",
    question: str = "",
) -> "discord.Embed":
    """Build a single embed with one field per result (max 25)."""
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
        embed.add_field(name=name, value=value, inline=False)

    n_results = len(results)
    if query:
        total_occ = sum(count_occurrences(c.get("text", ""), query) for c, _ in results)
        footer = (
            f"{n_results} chunk{'s' if n_results != 1 else ''} · "
            f"{total_occ} mention{'s' if total_occ != 1 else ''}"
        )
    else:
        footer = f"{n_results} result{'s' if n_results != 1 else ''}"
    embed.set_footer(text=footer)
    return embed
