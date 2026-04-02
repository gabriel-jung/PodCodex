"""YouTube routes — fetch video list, download audio, import subtitles."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from podcodex.api.routes._helpers import is_downloaded, require_show_folder, submit_task
from podcodex.api.schemas import RSSEpisodeOut, TaskResponse
from podcodex.ingest.rss import (
    RSSEpisode,
    episode_stem,
    load_feed_cache,
    save_episode_meta,
    save_feed_cache,
)
from podcodex.ingest.show import load_show_meta

router = APIRouter()


# ── Request models ─────────────────────────────


class YouTubeDownloadRequest(BaseModel):
    video_ids: list[str] | None = None  # None = download all
    import_subs: bool = False  # also import YouTube subtitles as transcript
    sub_lang: str = "en"


class YouTubeSubsRequest(BaseModel):
    video_ids: list[str]
    lang: str = "en"


# ── Helpers ────────────────────────────────────


def _yt_to_out(ep: RSSEpisode, show_folder: Path) -> dict:
    """Convert an RSSEpisode to an RSSEpisodeOut dict."""
    stem = episode_stem(ep)
    return {
        **asdict(ep),
        "local_stem": stem,
        "downloaded": is_downloaded(show_folder, stem),
    }


# ── Routes ─────────────────────────────────────


@router.post("/{show_folder:path}/youtube/fetch", response_model=list[RSSEpisodeOut])
async def youtube_fetch(show_folder: str) -> list[dict]:
    """Refresh the video list for a YouTube show."""
    from podcodex.ingest.youtube import fetch_youtube

    path = require_show_folder(show_folder)
    meta = load_show_meta(path)
    if not meta or not meta.youtube_url:
        raise HTTPException(400, "No YouTube URL in show.toml")

    try:
        episodes = fetch_youtube(meta.youtube_url)
    except ImportError as exc:
        raise HTTPException(501, str(exc)) from None
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch videos: {exc}") from None

    if not episodes:
        raise HTTPException(502, "No videos found")

    save_feed_cache(path, episodes)
    return [_yt_to_out(ep, path) for ep in episodes]


@router.post(
    "/{show_folder:path}/youtube/download",
    response_model=TaskResponse,
)
async def youtube_download(
    show_folder: str,
    req: YouTubeDownloadRequest,
) -> TaskResponse:
    """Download YouTube episodes as a background task."""
    from podcodex.ingest.youtube import (
        download_youtube_audio,
        import_youtube_transcript,
    )

    path = require_show_folder(show_folder)
    cached = load_feed_cache(path)
    if cached is None:
        raise HTTPException(400, "No cached video list — fetch YouTube first")

    targets = cached
    if req.video_ids:
        id_set = set(req.video_ids)
        targets = [ep for ep in cached if ep.guid in id_set]
        if not targets:
            raise HTTPException(404, "No matching videos found for the given IDs")

    meta = load_show_meta(path)
    show_name = (meta.name if meta else "") or path.name

    def run_downloads(progress_cb, episodes=targets, show_path=path):
        """Download each episode, optionally importing subtitles."""
        from podcodex.ingest.folder import invalidate_scan_cache

        cancel = getattr(progress_cb, "cancel_event", None)
        results = []
        total = len(episodes)
        for i, ep in enumerate(episodes):
            if cancel and cancel.is_set():
                progress_cb(i / total, "Cancelled")
                break

            stem = episode_stem(ep)
            progress_cb(i / total, f"Downloading {i + 1}/{total}: {ep.title[:40]}")

            if is_downloaded(show_path, stem):
                results.append({"stem": stem, "status": "exists"})
            else:
                try:
                    audio_path = download_youtube_audio(
                        ep.guid,
                        show_path,
                        stem,
                    )
                    # Save episode metadata
                    episode_dir = show_path / stem
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    save_episode_meta(episode_dir, ep)

                    results.append(
                        {
                            "stem": stem,
                            "status": "downloaded",
                            "audio_path": str(audio_path),
                        }
                    )
                    invalidate_scan_cache(show_path)
                except Exception as exc:
                    logger.exception("Failed to download {}", ep.guid)
                    results.append(
                        {"stem": stem, "status": "failed", "error": str(exc)}
                    )
                    continue

            # Import subtitles if requested
            if req.import_subs:
                try:
                    episode_dir = show_path / stem
                    imported = import_youtube_transcript(
                        ep.guid,
                        episode_dir,
                        show_name,
                        stem,
                        lang=req.sub_lang,
                    )
                    if imported:
                        results[-1]["subs_imported"] = True
                except Exception as exc:
                    logger.warning("Subtitle import failed for {}: {}", stem, exc)

            # Small delay between downloads
            if total > 1:
                time.sleep(1)

        return results

    return submit_task("yt-download", str(path), run_downloads)


@router.post(
    "/{show_folder:path}/youtube/import-subs",
    response_model=TaskResponse,
)
async def youtube_import_subs(
    show_folder: str,
    req: YouTubeSubsRequest,
) -> TaskResponse:
    """Import YouTube subtitles as transcript versions. Background task."""
    from podcodex.ingest.youtube import import_youtube_transcript

    path = require_show_folder(show_folder)
    cached = load_feed_cache(path)
    if cached is None:
        raise HTTPException(400, "No cached video list")

    meta = load_show_meta(path)
    show_name = (meta.name if meta else "") or path.name

    id_set = set(req.video_ids)
    targets = [ep for ep in cached if ep.guid in id_set]
    if not targets:
        raise HTTPException(404, "No matching videos found")

    def run_import(progress_cb, episodes=targets, show_path=path):
        """Import subtitles for each episode, reporting progress."""
        from podcodex.ingest.folder import invalidate_scan_cache

        cancel = getattr(progress_cb, "cancel_event", None)
        imported = 0
        failed = 0
        total = len(episodes)
        for i, ep in enumerate(episodes):
            if cancel and cancel.is_set():
                progress_cb(i / total, "Cancelled")
                break

            stem = episode_stem(ep)
            progress_cb(i / total, f"Importing subs {i + 1}/{total}: {ep.title[:40]}")
            episode_dir = show_path / stem
            episode_dir.mkdir(parents=True, exist_ok=True)
            save_episode_meta(episode_dir, ep)
            try:
                if import_youtube_transcript(
                    ep.guid, episode_dir, show_name, stem, lang=req.lang
                ):
                    imported += 1
                else:
                    failed += 1
            except Exception as exc:
                logger.warning("Subtitle import failed for {}: {}", stem, exc)
                failed += 1

        invalidate_scan_cache(show_path)
        return {"imported": imported, "failed": failed, "total": total}

    return submit_task("yt-subs", str(path), run_import)
