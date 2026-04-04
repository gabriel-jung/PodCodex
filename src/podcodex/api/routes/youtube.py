"""YouTube routes — fetch video list, download audio, import subtitles."""

from __future__ import annotations


from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from podcodex.api.routes._helpers import (
    is_downloaded,
    require_show_folder,
    rss_episode_to_out,
    submit_task,
)
from podcodex.api.schemas import RSSEpisodeOut, TaskResponse
from podcodex.ingest.rss import (
    episode_stem,
    load_feed_cache,
    save_episode_meta,
    save_feed_cache,
)
from podcodex.ingest.show import load_show_meta, save_show_meta

router = APIRouter()


# ── Request models ─────────────────────────────


class YouTubeDownloadRequest(BaseModel):
    video_ids: list[str] | None = None  # None = download all
    import_subs: bool = False  # also import YouTube subtitles as transcript
    sub_lang: str = "en"


class YouTubeSubsRequest(BaseModel):
    video_ids: list[str]
    lang: str = "en"


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

    # One-time artwork upgrade: fetch channel avatar if artwork is missing
    # or doesn't look like a square avatar (yt3.googleusercontent.com)
    if meta:
        current = meta.artwork_url or ""
        needs_upgrade = not current or "yt3.googleusercontent.com" not in current
        if needs_upgrade:
            try:
                from podcodex.ingest.youtube import youtube_show_info

                info = youtube_show_info(meta.youtube_url)
                fresh = info.get("artwork_url", "")
                if fresh and fresh != current:
                    meta.artwork_url = fresh
                    save_show_meta(path, meta)
            except Exception:
                pass  # non-critical

    return [rss_episode_to_out(ep, path) for ep in episodes]


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
        from podcodex.ingest.youtube import (
            _CONSECUTIVE_FAIL_LIMIT,
            _pace_request,
            reset_pace,
        )

        cancel = getattr(progress_cb, "cancel_event", None)
        results = []
        consecutive_fails = 0
        total = len(episodes)
        reset_pace()
        for i, ep in enumerate(episodes):
            if cancel and cancel.is_set():
                progress_cb(i / total, "Cancelled")
                break

            stem = episode_stem(ep)
            progress_cb(i / total, f"Downloading {i + 1}/{total}: {ep.title[:40]}")

            if is_downloaded(show_path, stem):
                results.append({"stem": stem, "status": "exists"})
                consecutive_fails = 0
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
                    consecutive_fails = 0
                    invalidate_scan_cache(show_path)
                except Exception as exc:
                    logger.exception("Failed to download {}", ep.guid)
                    results.append(
                        {"stem": stem, "status": "failed", "error": str(exc)}
                    )
                    consecutive_fails += 1
                    if consecutive_fails >= _CONSECUTIVE_FAIL_LIMIT:
                        remaining = total - i - 1
                        progress_cb(
                            (i + 1) / total,
                            f"Stopped — YouTube is rate-limiting requests. "
                            f"{len(results)}/{total} processed, {remaining} skipped. "
                            f"Try again later with fewer episodes.",
                        )
                        break
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

            _pace_request()

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
        from podcodex.ingest.youtube import (
            _CONSECUTIVE_FAIL_LIMIT,
            _pace_request,
            reset_pace,
        )

        cancel = getattr(progress_cb, "cancel_event", None)
        imported = 0
        failed = 0
        consecutive_fails = 0
        total = len(episodes)
        reset_pace()
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
                _pace_request()
                if import_youtube_transcript(
                    ep.guid, episode_dir, show_name, stem, lang=req.lang
                ):
                    imported += 1
                    consecutive_fails = 0
                else:
                    failed += 1
                    consecutive_fails += 1
            except Exception as exc:
                logger.warning("Subtitle import failed for {}: {}", stem, exc)
                failed += 1
                consecutive_fails += 1

            if consecutive_fails >= _CONSECUTIVE_FAIL_LIMIT:
                remaining = total - i - 1
                progress_cb(
                    (i + 1) / total,
                    f"Stopped — YouTube is rate-limiting requests. "
                    f"Imported {imported}/{total}, {remaining} skipped. "
                    f"Try again later with fewer episodes.",
                )
                break

        invalidate_scan_cache(show_path)
        return {
            "imported": imported,
            "failed": failed,
            "total": total,
            "throttled": consecutive_fails >= _CONSECUTIVE_FAIL_LIMIT,
        }

    return submit_task("yt-subs", str(path), run_import)
