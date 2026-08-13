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
import subprocess
import threading
from pathlib import Path

import discord

from podcodex.bot.formatting import (
    episode_display,
    is_http_url,
    pub_day,
    truncate_description,
)

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
        date = pub_day(e.get("pub_date"))
        lines.append(f"• {title} · {date}" if date else f"• {title}")
    if n > _MAX_LISTED:
        lines.append(f"…and {n - _MAX_LISTED} more")
    embed.description = "\n".join(lines)
    return embed


_REPO_ROOT = Path(__file__).resolve().parents[3]
# Cap the listed commits so a long-overdue update stays one readable card.
_MAX_COMMITS = 12


def _git(*args: str) -> str:
    """Run git in the repo root; "" when git or the checkout is unavailable.

    The uv install path clones the repo, so the bot usually knows its commit.
    The Docker image copies only ``src/`` (no ``.git``), and there the commit
    is simply unknown: the card then omits it rather than inventing one.
    """
    # Only trust the computed root when it really is our checkout. For a
    # site-packages install, git would otherwise walk up and could report
    # whatever unrelated repo happens to contain the venv.
    if not (_REPO_ROOT / ".git").exists():
        return ""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def bot_revision() -> tuple[str, str]:
    """``(version, short_sha)`` for the running bot. ``short_sha`` may be ""."""
    try:
        from importlib.metadata import version as _pkg_version

        version = _pkg_version("podcodex")
    except Exception:
        version = ""
    return version, _git("rev-parse", "--short", "HEAD")


def repo_url() -> str:
    """The project's repository URL from package metadata, or ""."""
    try:
        from importlib.metadata import metadata as _pkg_metadata

        for entry in _pkg_metadata("podcodex").get_all("Project-URL") or []:
            name, _, url = entry.partition(",")
            if name.strip().lower() == "repository":
                return url.strip().rstrip("/")
    except Exception:
        pass
    return ""


def commit_subjects(old_sha: str, new_sha: str) -> list[str]:
    """Subjects of the commits between two revisions, newest last.

    Empty when either revision is unknown to this checkout (a fresh clone, or
    a deploy that skipped commits), so the card says nothing rather than
    guessing what changed.
    """
    if not old_sha or not new_sha:
        return []
    out = _git("log", "--format=%s", f"{old_sha}..{new_sha}")
    return [ln for ln in out.splitlines() if ln][:_MAX_COMMITS] if out else []


def changelog_section(version: str) -> str:
    """Return CHANGELOG.md's section for *version*, or "" when unavailable.

    Provenance rule: the notes are read from the shipped CHANGELOG, never
    synthesized. When no copy is found, the caller falls back to the bare
    version card rather than inventing a summary.

    Lookup order: the repo checkout (editable installs), then the process
    cwd (the Docker image copies CHANGELOG.md into /app, its WORKDIR).
    """
    text = None
    for path in (_REPO_ROOT / "CHANGELOG.md", Path.cwd() / "CHANGELOG.md"):
        try:
            text = path.read_text(encoding="utf-8")
            break
        except OSError:
            continue
    if text is None:
        return ""

    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.startswith(f"## [{version}]")), None
    )
    if start is None:
        return ""
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start + 1 : end]).strip()


def build_update_embed(
    version: str,
    *,
    sha: str = "",
    repo: str = "",
    notes: str = "",
    changes: list[str] | None = None,
) -> discord.Embed:
    """The 'bot updated' card.

    The bot is deployed by git pull / rsync, not by release, so its identity is
    the commit it runs, and a version bump is only one kind of update. The card
    states what it is running (version, and the commit when the checkout knows
    it, linked when the repository URL is known), then either the release notes
    for a new version or the list of commits since the last announcement.

    Every part is dropped when it cannot be derived: no commit in the Docker
    image, no link without a repository URL, no notes without a CHANGELOG.
    """
    lines: list[str] = []
    running = f"Now running **v{version}**" if version else "Now running"
    if sha:
        link = f"[`{sha}`]({repo}/commit/{sha})" if repo else f"`{sha}`"
        running += f" · {link}"
    lines.append(running)

    if notes:
        lines.append("")
        lines.append(notes)
    elif changes:
        lines.append("")
        lines.append("**Changes**")
        lines.extend(f"• {c}" for c in changes)

    return discord.Embed(
        title="🔖 PodCodex bot updated",
        description=truncate_description("\n".join(lines)),
        color=discord.Color.blurple(),
    )
