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
    artwork_url: str = ""  # per-episode itunes:image


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


def feed_artwork(url: str) -> str:
    """Extract the channel-level artwork URL from an RSS feed."""
    feed = feedparser.parse(url)
    # itunes:image is the most reliable source
    img = feed.feed.get("image", {})
    itunes_img = feed.feed.get("itunes_image", {})
    return itunes_img.get("href", "") or img.get("href", "")


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
                artwork_url=(entry.get("itunes_image") or {}).get("href", "")
                or (entry.get("image") or {}).get("href", ""),
            )
        )

    return episodes


# ── iTunes / Apple Podcasts search ────────────


@dataclass
class PodcastSearchResult:
    """A podcast found via the iTunes Search API."""

    name: str
    artist: str
    feed_url: str
    artwork_url: str = ""


def search_itunes(query: str, limit: int = 8) -> list[PodcastSearchResult]:
    """Search the iTunes/Apple Podcasts directory for podcasts."""
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode({"term": query, "media": "podcast", "limit": limit})
    url = f"https://itunes.apple.com/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "podcodex/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    results = []
    for r in data.get("results", []):
        feed = r.get("feedUrl", "")
        if not feed:
            continue
        results.append(
            PodcastSearchResult(
                name=r.get("collectionName", ""),
                artist=r.get("artistName", ""),
                feed_url=feed,
                artwork_url=r.get("artworkUrl600", "")
                or r.get("artworkUrl100", "")
                or r.get("artworkUrl60", ""),
            )
        )
    return results


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


# ── Context builder ───────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def build_episode_context(
    show_name: str = "",
    episode_dir: Path | str | None = None,
    max_desc: int = 500,
) -> str:
    """Build an LLM context string from show name + RSS episode metadata.

    Returns a multi-line string suitable for the ``context`` parameter of
    :func:`~podcodex.core.correct.correct_segments` and
    :func:`~podcodex.core.translate.translate_segments`.

    Args:
        show_name   : display name of the podcast
        episode_dir : path to the episode output directory (may contain
                      ``.episode_meta.json``)
        max_desc    : maximum characters to keep from the RSS description
    """
    parts: list[str] = []
    if show_name:
        parts.append(f"Podcast: {show_name}")

    if episode_dir is not None:
        meta = load_episode_meta(Path(episode_dir))
        if meta:
            # Title + optional episode/season identifier
            if meta.title:
                ep_id = ""
                if meta.season_number is not None and meta.episode_number is not None:
                    ep_id = f" (S{meta.season_number}E{meta.episode_number})"
                elif meta.episode_number is not None:
                    ep_id = f" (Episode {meta.episode_number})"
                parts.append(f'Episode: "{meta.title}"{ep_id}')
            # HTML-stripped, truncated description
            if meta.description:
                desc = _HTML_TAG_RE.sub("", meta.description).strip()
                if len(desc) > max_desc:
                    desc = desc[:max_desc].rsplit(" ", 1)[0] + "…"
                parts.append(f"Description: {desc}")

    return "\n".join(parts)


# ── Audio download ────────────────────────────


def _cleanup_partial(path: Path) -> None:
    """Remove a partial download, ignoring errors (e.g. Windows/WSL file locks)."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def download_audio(
    rss_episode: RSSEpisode,
    show_folder: Path,
    force: bool = False,
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

    if not force and dest.exists():
        logger.info(f"Audio already exists: {dest.name}")
        return dest

    import httpx
    import time as _time

    max_retries = 3
    # connect=15s, read=30s per chunk, no total cap (large files need time)
    timeout = httpx.Timeout(connect=15, read=30, write=30, pool=15)
    logger.info(f"Downloading {rss_episode.audio_url} → {dest.name}")
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.stream(
                "GET", rss_episode.audio_url, follow_redirects=True, timeout=timeout
            ) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
            break  # success
        except httpx.HTTPStatusError:
            logger.warning(
                f"Download failed for {dest.name} (HTTP error, attempt {attempt}/{max_retries})"
            )
            _cleanup_partial(dest)
            if attempt == max_retries:
                return None
            _time.sleep(2 * attempt)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            logger.warning(
                f"Download failed for {dest.name}: {exc} (attempt {attempt}/{max_retries})"
            )
            _cleanup_partial(dest)
            if attempt == max_retries:
                return None
            _time.sleep(2 * attempt)

    logger.success(
        f"Downloaded {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)"
    )
    return dest
