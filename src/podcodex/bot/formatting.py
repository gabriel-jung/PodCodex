"""podcodex.bot.formatting — Pure display and search-merge helpers."""

from __future__ import annotations

import time
from collections import defaultdict

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


def speaker(chunk: dict) -> str:
    return chunk.get("speaker") or chunk.get("dominant_speaker") or "Unknown"


def score_bar(score: float, width: int = 8) -> str:
    filled = round(score * width)
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

    header = f"**{show} — {episode}**" if (show or episode) else "*Context*"
    lines = [header + f" · ±{n} turns\n"]

    for c in before:
        lines.append(f"*{speaker(c)}* · {fmt_time(c.get('start', 0))}\n{c['text']}")

    lines.append(
        f"**▶ {speaker(current)}** · {fmt_time(current.get('start', 0))}\n"
        f"**{current['text']}**"
    )

    for c in after:
        lines.append(f"*{speaker(c)}* · {fmt_time(c.get('start', 0))}\n{c['text']}")

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
