"""podcodex.bot.formatting — Pure display and search-merge helpers."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import discord

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

_MAX_CONTEXT_N = 8
_MAX_CHARS = 1900
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
    """Count case-insensitive occurrences of *query* in *text*."""
    if not query:
        return 0
    return text.lower().count(query.lower())


def highlight(text: str, query: str) -> str:
    """Case-insensitive highlight: wrap all occurrences of *query* in bold."""
    import re

    if not query:
        return text
    escaped = re.escape(query)
    return re.sub(f"({escaped})", r"**__\1__**", text, flags=re.IGNORECASE)


def speaker_lines(chunk: dict, query: str = "") -> str:
    """Format chunk text with per-turn speaker labels when available."""
    turns: list[dict] = chunk.get("speakers") or []
    if not turns:
        text = chunk.get("text", "")
        return highlight(text, query) if query else text
    lines = []
    for t in turns:
        spk = t.get("speaker", "Unknown")
        start = t.get("start", 0)
        ts_part = f" ({fmt_time(start)})" if start else ""
        text = highlight(t.get("text", ""), query) if query else t.get("text", "")
        lines.append(f"**{spk}**{ts_part}: {text}")
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

    ep_display = (neighbors[0].get("episode_title") if neighbors else None) or episode
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
# Result merging
# ──────────────────────────────────────────────


def merge_results(
    hits_by_collection: dict[str, list[dict]],
    top_k: int,
    strategy: str = "roundrobin",
) -> list[tuple[dict, str]]:
    """
    Merge per-collection hits into a final ranked list of (chunk, collection).

    Strategies:
      - "score"      : global sort by score, slice to top_k (original behaviour,
                       prone to one dominant show flooding results)
      - "roundrobin" : interleave one result per collection in score order,
                       ensures show diversity (default)
    """
    if strategy == "score":
        all_hits = [
            (chunk, col)
            for col, chunks in hits_by_collection.items()
            for chunk in chunks
        ]
        all_hits.sort(key=lambda x: x[0].get("score", 0.0), reverse=True)
        return all_hits[:top_k]

    # Round-robin: sort each collection's hits by score, then interleave
    sorted_cols: dict[str, list[dict]] = {
        col: sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
        for col, chunks in hits_by_collection.items()
    }
    result: list[tuple[dict, str]] = []
    queues = list(sorted_cols.items())
    idx = defaultdict(int)

    while len(result) < top_k:
        advanced = False
        for col, chunks in queues:
            if len(result) >= top_k:
                break
            i = idx[col]
            if i < len(chunks):
                result.append((chunks[i], col))
                idx[col] += 1
                advanced = True
        if not advanced:
            break  # all collections exhausted

    return result


# ──────────────────────────────────────────────
# Per-user cooldown
# ──────────────────────────────────────────────


class CooldownManager:
    """Simple in-memory per-user cooldown tracker."""

    def __init__(self, seconds: float = COOLDOWN_SECONDS) -> None:
        self._seconds = seconds
        self._last_used: dict[int, float] = {}

    def check(self, user_id: int) -> float:
        """
        Returns 0.0 if the user is allowed to proceed,
        or the remaining wait time in seconds if they are on cooldown.
        """
        now = time.monotonic()
        last = self._last_used.get(user_id, 0.0)
        remaining = self._seconds - (now - last)
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
        episode = chunk.get("episode_title") or chunk.get("episode", "")
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
