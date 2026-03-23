"""
podcodex.ingest.rss — RSS feed parsing and episode discovery.

Fetches a podcast RSS feed, extracts episode metadata, and caches it
locally as ``.feed_cache.json`` in the show folder.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import feedparser
from loguru import logger

_FEED_CACHE = ".feed_cache.json"
EPISODE_META_FILE = ".episode_meta.json"


@dataclass
class RSSEpisode:
    """Metadata for a single podcast episode from an RSS feed."""

    guid: str
    title: str
    pub_date: str  # ISO 8601 or raw feed date string
    description: str = ""
    audio_url: str = ""
    duration: float = 0.0  # seconds
    episode_number: int | None = None  # from itunes:episode tag only
    season_number: int | None = None  # from itunes:season tag only


def slug_from_title(title: str) -> str:
    """Convert an episode title to a filesystem-safe stem.

    Preserves Unicode letters (accented chars, CJK, etc.) and digits.
    Only strips characters that are unsafe for filenames.

    >>> slug_from_title("Episode 1: Hello World!")
    'episode_1_hello_world'
    >>> slug_from_title("Épisode 116 : Sol'ne et la dot au Moyen Âge")
    'épisode_116_sol_ne_et_la_dot_au_moyen_âge'
    """
    slug = title.lower().strip()
    slug = re.sub(r"[^\w]+", "_", slug, flags=re.UNICODE)
    slug = slug.strip("_")
    slug = unicodedata.normalize("NFC", slug)
    return slug[:120] if slug else "untitled"


def _parse_duration(raw: str) -> float:
    """Parse itunes:duration (HH:MM:SS, MM:SS, or seconds) to float seconds."""
    if not raw:
        return 0.0
    parts = raw.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def _parse_int_tag(value: str) -> int | None:
    """Parse an optional iTunes integer tag (episode/season number)."""
    if not value:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def episode_stem(rss_episode: RSSEpisode) -> str:
    """Return the filesystem stem for an RSS episode.

    Prefixed with the episode number when available (e.g. ``116_épisode_...``),
    otherwise just the slug from the title.
    """
    slug = slug_from_title(rss_episode.title)
    if rss_episode.episode_number is not None:
        return f"{rss_episode.episode_number}_{slug}"
    return slug


def _audio_ext_from_url(url: str) -> str:
    """Extract audio file extension from an enclosure URL."""
    path = urlparse(url).path
    for ext in (".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac"):
        if path.lower().endswith(ext):
            return ext
    return ".mp3"  # safe default


def _extract_audio_url(entry) -> str:
    """Find the first audio enclosure URL in a feed entry."""
    for link in entry.get("enclosures", []):
        if link.get("type", "").startswith("audio/"):
            return link.get("href", "")
    for link in entry.get("links", []):
        if link.get("type", "").startswith("audio/"):
            return link.get("href", "")
    return ""


def fetch_feed(url: str) -> list[RSSEpisode]:
    """Fetch and parse an RSS feed. Returns episodes in feed order."""
    feed = feedparser.parse(url)
    if feed.bozo and not feed.entries:
        logger.warning(f"Feed parse error: {feed.bozo_exception}")
        return []

    episodes: list[RSSEpisode] = []
    for entry in feed.entries:
        guid = entry.get("id") or entry.get("link") or entry.get("title", "")
        title = entry.get("title", "Untitled")

        description = entry.get("summary", "")
        if not description and entry.get("content"):
            description = entry.content[0].get("value", "")

        episodes.append(
            RSSEpisode(
                guid=guid,
                title=title,
                pub_date=entry.get("published", ""),
                description=description,
                audio_url=_extract_audio_url(entry),
                duration=_parse_duration(entry.get("itunes_duration", "")),
                episode_number=_parse_int_tag(entry.get("itunes_episode", "")),
                season_number=_parse_int_tag(entry.get("itunes_season", "")),
            )
        )

    return episodes


# ── Feed cache ────────────────────────────────


def load_feed_cache(show_folder: Path) -> list[RSSEpisode] | None:
    """Load cached feed data from *show_folder*. Returns None if no cache exists."""
    path = Path(show_folder) / _FEED_CACHE
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [RSSEpisode(**ep) for ep in raw]


def save_feed_cache(show_folder: Path, episodes: list[RSSEpisode]) -> Path:
    """Cache feed data to *show_folder*."""
    Path(show_folder).mkdir(parents=True, exist_ok=True)
    path = Path(show_folder) / _FEED_CACHE
    path.write_text(
        json.dumps([asdict(ep) for ep in episodes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ── Per-episode metadata ──────────────────────


def save_episode_meta(episode_dir: Path, rss_episode: RSSEpisode) -> Path:
    """Persist RSS metadata as ``EPISODE_META_FILE`` in the episode output dir.

    Saved alongside processing outputs so that the display title, pub date,
    description, etc. survive independently of the feed cache.
    """
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    path = episode_dir / EPISODE_META_FILE
    path.write_text(
        json.dumps(asdict(rss_episode), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_episode_meta(episode_dir: Path) -> RSSEpisode | None:
    """Load episode metadata from ``EPISODE_META_FILE``, or None if absent."""
    path = Path(episode_dir) / EPISODE_META_FILE
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return RSSEpisode(**raw)
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning(f"Corrupt episode meta: {path}")
        return None


# ── Audio download ────────────────────────────


def download_audio(
    rss_episode: RSSEpisode,
    show_folder: Path,
) -> Path | None:
    """Download audio from *rss_episode.audio_url* into *show_folder*.

    Also persists episode metadata to the episode output directory.
    Returns the local path, or None if no audio URL is available.
    """
    if not rss_episode.audio_url:
        return None

    ext = _audio_ext_from_url(rss_episode.audio_url)
    stem = episode_stem(rss_episode)
    dest = Path(show_folder) / f"{stem}{ext}"

    # Always persist episode metadata (even if audio already downloaded)
    save_episode_meta(Path(show_folder) / stem, rss_episode)

    if dest.exists():
        logger.info(f"Audio already exists: {dest.name}")
        return dest

    import httpx

    logger.info(f"Downloading {rss_episode.audio_url} → {dest.name}")
    with httpx.stream("GET", rss_episode.audio_url, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)

    logger.success(
        f"Downloaded {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)"
    )
    return dest
