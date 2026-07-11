"""podcodex.bot.announce — new-episode + version announcements.

The Discord bot sees only the LanceDB index (rsynced from the desktop), so it
learns about new episodes by *polling and diffing*, never by a push signal. This
module owns that diff state and the (pure) embed builders; the polling loop and
Discord I/O live in :mod:`podcodex.bot.bot`.

Provenance rule: every announced field comes from real episode metadata or a
user choice, or it is omitted. Nothing is synthesized (see the design spec).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import discord

from podcodex.bot.formatting import episode_display, is_http_url, pub_month

# Cap the per-show episode list so a large back-to-back batch stays one readable
# embed rather than blowing past Discord's description limit.
_MAX_LISTED = 15


class AnnounceStore:
    """Durable diff state for announcements, in its own SQLite file.

    Isolated from the search-result cache: this file only tracks which episode
    stems have been seen per collection (so new ones can be detected) and the
    last announced bot version. Thread-safe via a single lock — operations are
    sub-millisecond, so contention is a non-issue.
    """

    def __init__(self, db_path: Path) -> None:
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS baselined (
              collection TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS seen_episodes (
              collection TEXT NOT NULL,
              stem       TEXT NOT NULL,
              PRIMARY KEY (collection, stem)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS meta (
              key   TEXT PRIMARY KEY,
              value TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )

    def observe(self, collection: str, current: set[str]) -> list[str]:
        """Record the current episode set and return the newly-appeared stems.

        First observation of a collection is a **silent baseline**: the whole
        back-catalogue is recorded and ``[]`` is returned, so a fresh index (or
        a freshly-added show) never dumps every existing episode. Afterwards,
        only stems not seen before are returned (and then recorded).
        """
        with self._lock:
            baselined = (
                self._conn.execute(
                    "SELECT 1 FROM baselined WHERE collection = ?", (collection,)
                ).fetchone()
                is not None
            )
            if not baselined:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO seen_episodes(collection, stem) VALUES (?, ?)",
                    [(collection, s) for s in current],
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO baselined(collection) VALUES (?)",
                    (collection,),
                )
                return []

            seen = {
                r[0]
                for r in self._conn.execute(
                    "SELECT stem FROM seen_episodes WHERE collection = ?", (collection,)
                )
            }
            new = sorted(current - seen)
            if new:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO seen_episodes(collection, stem) VALUES (?, ?)",
                    [(collection, s) for s in new],
                )
            return new

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ── Pure embed builders (no Discord I/O) ──────────────────────────────


def build_new_episodes_embed(show: str, episodes: list[dict]) -> discord.Embed:
    """A grouped 'new episodes' card for one show.

    ``episodes`` are per-episode stat dicts (``episode_title``, ``pub_date``,
    ``artwork_url``, ...), expected newest-first. Only real fields are shown:
    an episode with no ``pub_date`` gets no date, no ``artwork_url`` means no
    thumbnail — nothing is invented.
    """
    n = len(episodes)
    embed = discord.Embed(
        title=f"📣 {n} new episode{'s' if n != 1 else ''} — {show}",
        color=discord.Color.green(),
    )
    art = next(
        (
            (e.get("artwork_url") or "").strip()
            for e in episodes
            if is_http_url(e.get("artwork_url"))
        ),
        "",
    )
    if art:
        embed.set_thumbnail(url=art)

    lines: list[str] = []
    for e in episodes[:_MAX_LISTED]:
        title = episode_display(e) or "(untitled)"
        month = pub_month(e.get("pub_date"))
        lines.append(f"• {title} · {month}" if month else f"• {title}")
    if n > _MAX_LISTED:
        lines.append(f"…and {n - _MAX_LISTED} more")
    embed.description = "\n".join(lines)
    return embed


def build_version_embed(version: str) -> discord.Embed:
    """A minimal 'bot updated' card. Version is the real ``__version__``."""
    return discord.Embed(
        title=f"🔖 PodCodex bot v{version}",
        color=discord.Color.blurple(),
    )
