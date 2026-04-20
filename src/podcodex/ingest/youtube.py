"""
podcodex.ingest.youtube — YouTube channel/playlist/video discovery and download.

Mirrors the RSS ingest flow: fetch metadata → cache episodes → download audio.
Reuses :class:`~podcodex.ingest.rss.RSSEpisode` so that the rest of the pipeline
(unified episodes, folder scanning, transcription) works without modification.

Requires the optional ``[youtube]`` dependency group (``yt-dlp``).
"""

from __future__ import annotations

import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from podcodex.ingest.rss import (
    RSSEpisode,
    load_feed_cache,
    save_feed_cache,
)

_YT_VIDEO_URL = "https://www.youtube.com/watch?v={video_id}"

# ── Rate limiting ────────────────────────────────────────────
# YouTube throttles unauthenticated requests after ~30-50 calls.
# We pace requests with increasing delays to stay under the limit.
_BASE_DELAY = 2.0  # seconds between requests (first batch)
_BACKOFF_AFTER = 20  # start increasing delay after this many requests
_MAX_DELAY = 8.0  # maximum delay between requests
_CONSECUTIVE_FAIL_LIMIT = 3  # abort batch after this many consecutive failures
_request_count = 0
_pace_lock = threading.Lock()

# yt-dlp error strings that indicate rate limiting / bot detection
_THROTTLE_PATTERNS = (
    "HTTP Error 429",
    "Sign in to confirm",
    "too many requests",
    "This helps protect our community",
)


class YouTubeThrottledError(RuntimeError):
    """Raised when YouTube appears to be rate-limiting requests."""


def _is_throttle_error(exc: Exception) -> bool:
    """Check if an exception looks like YouTube rate limiting."""
    msg = str(exc).lower()
    return any(p.lower() in msg for p in _THROTTLE_PATTERNS)


def _pace_request() -> None:
    """Sleep between YouTube API calls to avoid rate limiting."""
    import time

    global _request_count
    with _pace_lock:
        _request_count += 1
        count = _request_count

    if count <= 1:
        return

    if count <= _BACKOFF_AFTER:
        delay = _BASE_DELAY
    else:
        extra = min((count - _BACKOFF_AFTER) / 30.0, 1.0)
        delay = _BASE_DELAY + extra * (_MAX_DELAY - _BASE_DELAY)

    time.sleep(delay)


def reset_pace() -> None:
    """Reset the request counter (call at the start of a new batch)."""
    global _request_count
    with _pace_lock:
        _request_count = 0


def _base_ydl_opts(**extra: Any) -> dict[str, Any]:
    """Build common yt-dlp options."""
    opts: dict[str, Any] = {"quiet": True, "no_warnings": True}
    opts.update(extra)
    return opts


# Patterns that indicate a channel URL (not a playlist or single video).
_CHANNEL_PATTERNS = (
    "youtube.com/@",
    "youtube.com/channel/",
    "youtube.com/c/",
    "youtube.com/user/",
)


def _normalize_channel_url(url: str) -> str:
    """Append ``/videos`` to channel URLs so yt-dlp returns actual videos
    instead of channel tabs (Videos, Live, Shorts)."""
    lower = url.lower().rstrip("/")
    if any(p in lower for p in _CHANNEL_PATTERNS):
        # Don't double-append if already targeting a tab
        if not any(
            lower.endswith(f"/{tab}")
            for tab in ("videos", "live", "shorts", "streams", "playlists")
        ):
            return url.rstrip("/") + "/videos"
    return url


def _require_yt_dlp():
    """Import and return yt_dlp, raising a clear error if missing."""
    try:
        import yt_dlp

        return yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp is required for YouTube support. "
            "Install with: pip install podcodex[youtube]"
        )


def _extract_video_id(url_or_id: str) -> str:
    """Extract a YouTube video ID from a URL or return as-is if already an ID."""
    # Already a bare ID (11 chars, alphanumeric + dash/underscore)
    if re.match(r"^[\w-]{11}$", url_or_id):
        return url_or_id
    # Standard watch URL
    m = re.search(r"[?&]v=([\w-]{11})", url_or_id)
    if m:
        return m.group(1)
    # Short URL (youtu.be/ID)
    m = re.search(r"youtu\.be/([\w-]{11})", url_or_id)
    if m:
        return m.group(1)
    return url_or_id


def _entry_to_episode(entry: dict[str, Any], idx: int) -> RSSEpisode:
    """Map a yt-dlp info_dict entry to an RSSEpisode."""
    video_id = entry.get("id", "")
    title = entry.get("title") or entry.get("fulltitle") or "Untitled"

    # Parse upload date — try multiple yt-dlp fields
    pub_date = ""
    upload_date = entry.get("upload_date", "")
    if upload_date and len(upload_date) == 8:
        try:
            dt = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
            pub_date = dt.isoformat()
        except ValueError:
            pub_date = upload_date
    if not pub_date:
        # Flat extraction often provides epoch timestamps instead
        ts = entry.get("timestamp") or entry.get("release_timestamp")
        if ts:
            try:
                pub_date = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            except (ValueError, OSError):
                pass

    description = entry.get("description", "") or ""
    duration = float(entry.get("duration") or 0)
    thumbnail = entry.get("thumbnail") or entry.get("thumbnails", [{}])[-1].get(
        "url", ""
    )

    return RSSEpisode(
        guid=video_id,
        title=title,
        pub_date=pub_date,
        description=description,
        audio_url="",  # yt-dlp handles download separately
        duration=duration,
        episode_number=None,
        season_number=None,
        artwork_url=thumbnail,
    )


def _best_thumbnail(info: dict[str, Any], prefer_square: bool = False) -> str:
    """Pick the best thumbnail URL from a yt-dlp info dict.

    Args:
        prefer_square: When True, prefer square-ish images (channel avatar)
                       over wide banners. Used for show-level artwork.
    """
    thumbs = info.get("thumbnails") or []
    if thumbs and prefer_square:
        # Separate square-ish (aspect ratio ≤ 1.5) from wide banners
        def _score(t: dict) -> tuple[int, int]:
            w, h = t.get("width", 0), t.get("height", 0)
            is_square = 0.6 <= (w / h if h else 0) <= 1.5
            return (1 if is_square else 0, w * h)

        best = max(thumbs, key=_score)
        url = best.get("url", "")
        if url:
            return url

    # Direct thumbnail field (fallback)
    thumb = info.get("thumbnail") or ""
    if thumb:
        return thumb
    # Pick the largest from the list
    if thumbs:
        best = max(thumbs, key=lambda t: t.get("width", 0) * t.get("height", 0))
        return best.get("url", "")
    return ""


def youtube_show_info(url: str) -> dict[str, Any]:
    """Extract channel/playlist metadata for show creation.

    Args:
        url: YouTube channel, playlist, or single video URL.

    Returns:
        Dict with ``name``, ``artwork_url``, and ``video_count`` keys.
    """
    yt_dlp = _require_yt_dlp()
    url = _normalize_channel_url(url)

    ydl_opts = _base_ydl_opts(extract_flat=True, skip_download=True)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        return {"name": "", "artwork_url": "", "video_count": 0}

    name = info.get("channel") or info.get("uploader") or info.get("title") or ""
    thumbnail = _best_thumbnail(info, prefer_square=True)
    entries = info.get("entries")
    if entries is not None:
        video_count = info.get("playlist_count") or 0
        if not video_count:
            try:
                entries = list(entries)
                video_count = len(entries)
            except Exception:
                video_count = 0
    else:
        video_count = 1

    return {
        "name": name,
        "artwork_url": thumbnail,
        "video_count": video_count,
    }


def fetch_youtube(url: str) -> list[RSSEpisode]:
    """Extract video metadata from a YouTube channel, playlist, or video URL.

    Uses flat extraction (no download) to quickly enumerate videos.
    Each video is mapped to an :class:`RSSEpisode` for compatibility with the
    existing feed cache and unified episode system.

    Args:
        url: YouTube channel, playlist, or single video URL.

    Returns:
        List of :class:`RSSEpisode` instances, newest first.
    """
    yt_dlp = _require_yt_dlp()
    url = _normalize_channel_url(url)

    ydl_opts = _base_ydl_opts(extract_flat="in_playlist", skip_download=True)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        return []

    entries: list[dict[str, Any]]
    if info.get("entries") is not None:
        entries = list(info["entries"])
    else:
        # Single video
        entries = [info]

    episodes: list[RSSEpisode] = []
    for idx, entry in enumerate(entries):
        if not entry:
            continue
        ep = _entry_to_episode(entry, idx)
        if ep.guid:
            episodes.append(ep)

    logger.info("Fetched {} videos from {}", len(episodes), url)
    return episodes


def download_youtube_audio(
    video_id: str,
    show_folder: Path,
    stem: str,
    progress_cb: Callable[[float, str], None] | None = None,
) -> Path:
    """Download audio from a YouTube video via yt-dlp.

    Extracts audio as MP3 at 192K quality.

    Args:
        video_id: YouTube video ID.
        show_folder: Show folder to save the MP3 into.
        stem: Filename stem for the output file.
        progress_cb: Optional ``(fraction, message)`` callback for progress.

    Returns:
        Path to the downloaded MP3 file.

    Raises:
        RuntimeError: If the download fails.
    """
    yt_dlp = _require_yt_dlp()

    show_folder = Path(show_folder)
    show_folder.mkdir(parents=True, exist_ok=True)
    output_path = show_folder / f"{stem}.mp3"

    if output_path.exists():
        logger.debug("Audio already exists: {}", output_path)
        return output_path

    url = _YT_VIDEO_URL.format(video_id=video_id)

    def _progress_hook(d: dict) -> None:
        if progress_cb and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                frac = downloaded / total
                progress_cb(frac, f"Downloading {frac:.0%}")

    ydl_opts = _base_ydl_opts(
        format="bestaudio/best",
        outtmpl=str(show_folder / f"{stem}.%(ext)s"),
        postprocessors=[
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        progress_hooks=[_progress_hook],
    )

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        rc = ydl.download([url])
        if rc != 0:
            raise RuntimeError(f"yt-dlp download failed for {video_id} (rc={rc})")

    if not output_path.exists():
        # yt-dlp may have used a different extension before post-processing
        # Look for the file with any extension and the same stem
        candidates = list(show_folder.glob(f"{stem}.*"))
        mp3s = [c for c in candidates if c.suffix == ".mp3"]
        if mp3s:
            output_path = mp3s[0]
        elif candidates:
            output_path = candidates[0]
            logger.warning(
                "Expected .mp3 but got {}: {}", output_path.suffix, output_path
            )
        else:
            raise RuntimeError(f"No output file found after download for {video_id}")

    logger.info("Downloaded audio: {}", output_path)
    return output_path


_AUTO_SUB_SLEEP = 60  # yt-dlp native sleep between auto-generated subtitle downloads


def download_youtube_subtitles(
    video_id: str,
    lang: str = "en",
) -> tuple[str | None, dict[str, Any]]:
    """Download subtitles for a YouTube video.

    Prefers manually uploaded subtitles (fast, lightweight).
    Falls back to auto-generated subs which require deno + remote components
    and a native yt-dlp sleep to avoid YouTube 429 rate limiting.

    Args:
        video_id: YouTube video ID.
        lang: Desired subtitle language code (default ``"en"``).

    Returns:
        Tuple of (VTT subtitle text or None, full yt-dlp info dict).
    """
    yt_dlp = _require_yt_dlp()

    url = _YT_VIDEO_URL.format(video_id=video_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: extract metadata to check which subtitle tracks exist
        meta_opts = _base_ydl_opts(
            skip_download=True,
            writesubtitles=True,
            writeautomaticsub=True,
            subtitleslangs=[lang],
            subtitlesformat="vtt",
            outtmpl=str(Path(tmpdir) / "subs.%(ext)s"),
        )

        with yt_dlp.YoutubeDL(meta_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            return None, {}

        manual_subs = info.get("subtitles", {})
        auto_subs = info.get("automatic_captions", {})
        has_manual = lang in manual_subs
        has_auto = lang in auto_subs

        if not has_manual and not has_auto:
            available = sorted(set(list(manual_subs.keys()) + list(auto_subs.keys())))
            logger.info(
                "No {} subtitles for {}. Available: {}",
                lang,
                video_id,
                ", ".join(available) if available else "none",
            )
            return None, info

        # Step 2: download the subtitle track
        if has_manual:
            # Manual subs: fast, no special options needed
            dl_opts = _base_ydl_opts(
                skip_download=True,
                writesubtitles=True,
                subtitleslangs=[lang],
                subtitlesformat="vtt",
                outtmpl=str(Path(tmpdir) / "subs"),
            )
        else:
            # Auto-generated subs: remote components + native sleep
            dl_opts = _base_ydl_opts(
                skip_download=True,
                writeautomaticsub=True,
                subtitleslangs=[lang],
                subtitlesformat="vtt",
                outtmpl=str(Path(tmpdir) / "subs"),
                remote_components=["ejs:github"],
                sleep_interval_subtitles=_AUTO_SUB_SLEEP,
            )
            logger.info(
                "Downloading auto-generated {} subs for {} ({}s delay)",
                lang,
                video_id,
                _AUTO_SUB_SLEEP,
            )

        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])

        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if not vtt_files:
            logger.debug("No VTT file produced for {}", video_id)
            return None, info

        vtt_text = vtt_files[0].read_text(encoding="utf-8")
        source = "manual" if has_manual else "auto-generated"
        logger.info("Downloaded {} subtitles for {} ({})", lang, video_id, source)
        return vtt_text, info


def cache_youtube_subtitles(
    video_id: str,
    episode_dir: Path,
    stem: str,
    lang: str = "en",
) -> bool:
    """Download YouTube subtitles and cache the VTT file on disk.

    Does NOT create a transcript version — that happens in the transcribe step
    when the user explicitly chooses "From subtitles".

    Args:
        video_id: YouTube video ID.
        episode_dir: Episode output directory.
        stem: Episode stem.
        lang: Subtitle language code.

    Returns:
        ``True`` if subtitles were found and cached, ``False`` otherwise.
    """
    vtt_text, video_info = download_youtube_subtitles(video_id, lang=lang)

    # Backfill feed cache with full metadata from the per-video extraction
    # (flat playlist extraction often misses upload_date, duration, description)
    if video_info:
        show_folder = episode_dir.parent
        cached = load_feed_cache(show_folder)
        if cached:
            full_description = video_info.get("description", "") or ""
            full_ep = _entry_to_episode(video_info, 0)
            dirty = False
            for ep in cached:
                if ep.guid != video_id:
                    continue
                if full_description and len(full_description) > len(ep.description):
                    ep.description = full_description
                    dirty = True
                if full_ep.pub_date and not ep.pub_date:
                    ep.pub_date = full_ep.pub_date
                    dirty = True
                if full_ep.duration and not ep.duration:
                    ep.duration = full_ep.duration
                    dirty = True
                break
            if dirty:
                save_feed_cache(show_folder, cached)

    no_subs_marker = episode_dir / ".no_subtitles"
    if not vtt_text:
        # YouTube can transiently hide captions (rate-limits, regional blocks,
        # temporary removals). Don't delete any existing VTT — it's still valid
        # transcript input. Only plant the "no subs" marker when nothing is
        # cached yet, so the UI can distinguish "never had subs" from
        # "previously cached".
        has_existing_vtt = any(episode_dir.glob(f"{stem}.subtitles.*.vtt"))
        if not has_existing_vtt:
            no_subs_marker.write_text(lang, encoding="utf-8")
        return False

    # Clear any previous no-subtitles marker
    if no_subs_marker.exists():
        no_subs_marker.unlink()

    # Save VTT file
    vtt_path = episode_dir / f"{stem}.subtitles.{lang}.vtt"
    vtt_path.write_text(vtt_text, encoding="utf-8")
    logger.info("Cached subtitles ({}) for {} → {}", lang, stem, vtt_path.name)
    return True
