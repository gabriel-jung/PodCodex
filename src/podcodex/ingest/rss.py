"""
podcodex.ingest.rss — RSS feed parsing and episode discovery.

Fetches a podcast RSS feed, extracts episode metadata, and caches it
locally as ``.feed_cache.json`` in the show folder.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
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
    removed: bool = False  # true when episode no longer appears in the live feed
    # Position in the source feed (0 = newest). Used as a sort fallback when
    # pub_date is missing (YouTube flat extraction often omits dates).
    feed_order: int | None = None


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


def _guid_suffix(guid: str) -> str:
    """Short filesystem-safe suffix derived from a GUID, for stem disambiguation.

    YouTube video IDs (11 chars, ``[\\w-]{11}``) are used as-is. Anything else
    (URL GUIDs from RSS, etc.) is hashed to 8 hex chars for stability.
    """
    if not guid:
        return ""
    if re.fullmatch(r"[\w-]{11}", guid):
        return guid
    return hashlib.sha1(guid.encode("utf-8")).hexdigest()[:8]


def episode_stem(rss_episode: RSSEpisode, show_folder: Path | str | None = None) -> str:
    """Return the filesystem stem for an RSS episode.

    - With an ``episode_number`` (RSS ``itunes:episode``, or a legacy-numbered
      YouTube entry): ``"{N}_{slug}"``.
    - Without a number: ``"{slug}_{guid_suffix}"`` to prevent title collisions
      (two videos called "Episode Rerun" would otherwise share a stem).

    When *show_folder* is provided and a legacy slug-only directory already
    exists (episodes created before the guid suffix was introduced), the legacy
    stem is returned so existing on-disk outputs keep resolving.
    """
    slug = slug_from_title(rss_episode.title)
    if rss_episode.episode_number is not None:
        return f"{rss_episode.episode_number}_{slug}"
    if show_folder is not None:
        legacy_dir = Path(show_folder) / slug
        if legacy_dir.is_dir():
            return slug
    suffix = _guid_suffix(rss_episode.guid)
    return f"{slug}_{suffix}" if suffix else slug


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


_feed_cache_memo: dict[str, tuple[float, list[RSSEpisode] | None]] = {}


def load_feed_cache(show_folder: Path) -> list[RSSEpisode] | None:
    """Load cached feed data from *show_folder*. Returns None if no cache exists.

    Result is memoised per ``(path, mtime)`` so hot paths that call this
    once per episode directory (``load_episode_meta`` self-heal,
    ``list_shows`` endpoint) don't re-parse the JSON on every call. Reads
    at most once per write.
    """
    path = Path(show_folder) / _FEED_CACHE
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _feed_cache_memo.pop(str(path), None)
        return None
    except OSError:
        return None
    key = str(path)
    cached = _feed_cache_memo.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _feed_cache_memo[key] = (mtime, None)
        return None
    episodes = [RSSEpisode(**ep) for ep in raw]
    _feed_cache_memo[key] = (mtime, episodes)
    return episodes


def merge_with_cache(
    fresh: list[RSSEpisode],
    existing: list[RSSEpisode] | None,
    on_match: Callable[[RSSEpisode, RSSEpisode], RSSEpisode] | None = None,
) -> list[RSSEpisode]:
    """Merge a fresh feed fetch with an existing cache.

    Episodes still in *fresh* keep ``removed=False``. Entries in *existing*
    that disappear from *fresh* are appended with ``removed=True`` so their
    local outputs stay visible in the UI. When *on_match* is supplied, it is
    called with ``(fresh_ep, cached_ep)`` for every GUID match and its return
    value replaces ``fresh_ep`` — used e.g. to keep a legacy YouTube
    ``episode_number``.
    """
    if not existing:
        for ep in fresh:
            ep.removed = False
        return list(fresh)
    old_by_guid = {ep.guid: ep for ep in existing}
    merged: list[RSSEpisode] = []
    for ep in fresh:
        old = old_by_guid.pop(ep.guid, None)
        if old is not None and on_match is not None:
            ep = on_match(ep, old)
        ep.removed = False
        merged.append(ep)
    for leftover in old_by_guid.values():
        leftover.removed = True
        merged.append(leftover)
    return merged


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
    """Load episode metadata from ``EPISODE_META_FILE``, or None if absent.

    When the file is empty or corrupt, fall back to the show-level
    ``.feed_cache.json`` and match by stem. On successful recovery, rewrite
    ``.episode_meta.json`` so subsequent loads are fast and the transcript-meta
    path (used by the chunker to stamp ``episode_title``) picks up the title.
    """
    episode_dir = Path(episode_dir)
    path = episode_dir / EPISODE_META_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        text = ""
    except OSError:
        return None
    if text:
        try:
            return RSSEpisode(**json.loads(text))
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.warning(
                f"Corrupt episode meta: {path}; attempting recovery from feed cache"
            )

    show_folder = episode_dir.parent
    cached = load_feed_cache(show_folder)
    if not cached:
        return None
    stem = episode_dir.name
    for ep in cached:
        if episode_stem(ep, show_folder) == stem:
            try:
                save_episode_meta(episode_dir, ep)
                logger.info(f"Recovered .episode_meta.json for {stem} from feed cache")
            except OSError:
                logger.opt(exception=True).warning(
                    f"Could not rewrite recovered meta for {stem}"
                )
            return ep
    return None


# ── Context builder ───────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_description(raw: str, max_chars: int = 500) -> str:
    """Strip HTML and soft-truncate at the nearest word boundary."""
    desc = _HTML_TAG_RE.sub("", raw).strip()
    if len(desc) > max_chars:
        desc = desc[:max_chars].rsplit(" ", 1)[0] + "…"
    return desc


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
            if meta.description:
                parts.append(
                    f"Description: {clean_description(meta.description, max_desc)}"
                )

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
) -> tuple[Path | None, str | None]:
    """Download audio from *rss_episode.audio_url* into *show_folder*.

    Also persists episode metadata to the episode output directory.

    Returns ``(path, error)``:
      - ``(path, None)`` on success (or if file already existed).
      - ``(None, reason)`` on failure — ``reason`` is a short
        user-facing string like ``"HTTP 429 (rate limited)"`` or
        ``"timeout"``, suitable for surfacing in a progress bar.
      - ``(None, None)`` only when *rss_episode* has no audio URL.
    """
    if not rss_episode.audio_url:
        return None, None

    ext = _audio_ext_from_url(rss_episode.audio_url)
    stem = episode_stem(rss_episode, show_folder)
    dest = Path(show_folder) / f"{stem}{ext}"

    # Always persist episode metadata (even if audio already downloaded)
    save_episode_meta(Path(show_folder) / stem, rss_episode)

    if not force and dest.exists():
        logger.info(f"Audio already exists: {dest.name}")
        return dest, None

    import httpx
    import time as _time

    max_retries = 2
    # connect=15s, read=30s per chunk, no total cap (large files need time)
    timeout = httpx.Timeout(connect=15, read=30, write=30, pool=15)
    # Default httpx User-Agent trips bot filters on some podcast CDNs (ausha, etc.).
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) PodCodex/0.1"
        ),
        "Accept": "*/*",
    }
    last_error: str = "unknown error"
    logger.info(f"Downloading {rss_episode.audio_url} → {dest.name}")
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.stream(
                "GET",
                rss_episode.audio_url,
                follow_redirects=True,
                timeout=timeout,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
            break  # success
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                last_error = "HTTP 429 (rate limited)"
            elif status == 503:
                last_error = "HTTP 503 (server overloaded)"
            elif status == 404:
                last_error = "HTTP 404 (audio not found)"
            elif status in (401, 403):
                last_error = f"HTTP {status} (access denied)"
            else:
                last_error = f"HTTP {status}"
            logger.warning(
                f"Download failed for {dest.name} ({last_error}, attempt {attempt}/{max_retries})"
            )
            _cleanup_partial(dest)
            if attempt == max_retries:
                return None, last_error
            if status in (429, 503):
                wait = 15
            else:
                wait = 2 * attempt
            logger.info(f"Retrying {dest.name} in {wait}s")
            _time.sleep(wait)
        except httpx.TimeoutException as exc:
            last_error = "timeout"
            logger.warning(
                f"Download failed for {dest.name}: {exc} (attempt {attempt}/{max_retries})"
            )
            _cleanup_partial(dest)
            if attempt == max_retries:
                return None, last_error
            _time.sleep(2 * attempt)
        except httpx.TransportError as exc:
            last_error = f"network error: {type(exc).__name__}"
            logger.warning(
                f"Download failed for {dest.name}: {exc} (attempt {attempt}/{max_retries})"
            )
            _cleanup_partial(dest)
            if attempt == max_retries:
                return None, last_error
            _time.sleep(2 * attempt)

    logger.success(
        f"Downloaded {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)"
    )
    return dest, None
