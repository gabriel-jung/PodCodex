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

    # Parse upload date (YYYYMMDD format from yt-dlp)
    upload_date = entry.get("upload_date", "")
    pub_date = ""
    if upload_date and len(upload_date) == 8:
        try:
            dt = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
            pub_date = dt.isoformat()
        except ValueError:
            pub_date = upload_date

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
        episode_number=idx + 1,
        season_number=None,
        artwork_url=thumbnail,
    )


def _best_thumbnail(info: dict[str, Any]) -> str:
    """Pick the best thumbnail URL from a yt-dlp info dict."""
    # Direct thumbnail field
    thumb = info.get("thumbnail") or ""
    if thumb:
        return thumb
    # thumbnails list — pick the largest
    thumbs = info.get("thumbnails") or []
    if thumbs:
        # Prefer by resolution, fallback to last entry
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

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        return {"name": "", "artwork_url": "", "video_count": 0}

    name = info.get("channel") or info.get("uploader") or info.get("title") or ""
    thumbnail = _best_thumbnail(info)
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

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
    }
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

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(show_folder / f"{stem}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_progress_hook],
    }

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


def download_youtube_subtitles(
    video_id: str,
    lang: str = "en",
) -> tuple[str | None, str]:
    """Download subtitles for a YouTube video.

    Prefers manually uploaded subtitles, falls back to auto-generated.
    Also returns the full video description (available from the same request).

    Args:
        video_id: YouTube video ID.
        lang: Desired subtitle language code (default ``"en"``).

    Returns:
        Tuple of (VTT subtitle text or None, full video description).
    """
    yt_dlp = _require_yt_dlp()

    url = _YT_VIDEO_URL.format(video_id=video_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [lang],
            "subtitlesformat": "vtt",
            "outtmpl": str(Path(tmpdir) / "subs.%(ext)s"),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            return None, ""

        full_description = info.get("description", "") or ""

        # Check for available subtitles
        manual_subs = info.get("subtitles", {})
        auto_subs = info.get("automatic_captions", {})

        # Prefer manual subs
        has_manual = lang in manual_subs
        has_auto = lang in auto_subs

        if not has_manual and not has_auto:
            logger.debug("No subtitles available for {} in {}", video_id, lang)
            return None, full_description

        # Download the subtitles
        dl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": has_manual,
            "writeautomaticsub": not has_manual and has_auto,
            "subtitleslangs": [lang],
            "subtitlesformat": "vtt",
            "outtmpl": str(Path(tmpdir) / "subs"),
        }

        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])

        # Find the VTT file
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if not vtt_files:
            logger.debug("No VTT file produced for {}", video_id)
            return None, full_description

        vtt_text = vtt_files[0].read_text(encoding="utf-8")
        source = "manual" if has_manual else "auto-generated"
        logger.info("Downloaded {} subtitles for {} ({})", lang, video_id, source)
        return vtt_text, full_description


def import_youtube_transcript(
    video_id: str,
    episode_dir: Path,
    show_name: str,
    stem: str,
    lang: str = "en",
) -> bool:
    """Download YouTube subtitles and save as a versioned transcript.

    Creates a version entry with provenance ``source="youtube-subtitles"``
    so the user can distinguish YouTube-sourced transcripts from Whisper runs.
    Also writes the legacy transcript file and marks the step in pipeline DB.

    Args:
        video_id: YouTube video ID.
        episode_dir: Episode output directory.
        show_name: Show name for metadata.
        stem: Episode stem.
        lang: Subtitle language code.

    Returns:
        ``True`` if subtitles were found and imported, ``False`` otherwise.
    """
    from podcodex.core._utils import save_segments_json, vtt_to_segments
    from podcodex.core.pipeline_db import mark_step
    from podcodex.core.versions import save_version

    vtt_text, full_description = download_youtube_subtitles(video_id, lang=lang)

    # Update feed cache with the full description (flat extraction truncates it)
    if full_description:
        show_folder = episode_dir.parent
        cached = load_feed_cache(show_folder)
        if cached:
            for ep in cached:
                if ep.guid == video_id and len(full_description) > len(ep.description):
                    ep.description = full_description
                    break
            save_feed_cache(show_folder, cached)

    if not vtt_text:
        return False

    # Save original VTT for reference / debugging
    vtt_path = episode_dir / f"{stem}.subtitles.{lang}.vtt"
    vtt_path.write_text(vtt_text, encoding="utf-8")
    logger.debug("Saved original VTT to {}", vtt_path)

    segments = vtt_to_segments(vtt_text)
    if not segments:
        logger.warning("VTT parsed to zero segments for {}", video_id)
        return False

    provenance = {
        "step": "transcript",
        "type": "raw",
        "model": None,
        "params": {
            "source": "youtube-subtitles",
            "video_id": video_id,
            "lang": lang,
        },
    }

    # Legacy file so folder scan detects transcribed=True
    legacy_path = episode_dir / f"{stem}.transcript.raw.json"
    save_segments_json(legacy_path, segments, "YouTube subtitles")

    # Version + pipeline DB
    save_version(
        base=episode_dir,
        step="transcript",
        segments=segments,
        provenance=provenance,
    )
    show_folder = episode_dir.parent
    mark_step(
        show_folder,
        episode_dir.name,
        transcribed=True,
        provenance={"transcript": provenance},
    )

    logger.info(
        "Imported {} YouTube subtitle segments for {}",
        len(segments),
        stem,
    )
    return True
