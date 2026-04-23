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
    RSSEpisode,
    episode_stem,
    load_feed_cache,
    merge_with_cache,
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

    # yt-dlp returns videos newest-first; persist that position so the UI can
    # sort sensibly even when pub_date is missing from flat extraction.
    for idx, ep in enumerate(episodes):
        ep.feed_order = idx

    # YouTube flat extraction often omits upload_date/duration/description, and
    # legacy numbered entries (pre-fix) need their episode_number kept so their
    # on-disk ``{n}_slug`` stems still resolve. Per-video subtitle import
    # (``cache_youtube_subtitles``) backfills those fields into the cache;
    # fall back to the cached value whenever the fresh record has nothing.
    def _preserve_enriched(fresh: RSSEpisode, old: RSSEpisode) -> RSSEpisode:
        merged = RSSEpisode(**fresh.__dict__)
        if old.episode_number is not None:
            merged.episode_number = old.episode_number
        if not merged.pub_date and old.pub_date:
            merged.pub_date = old.pub_date
        if not merged.duration and old.duration:
            merged.duration = old.duration
        if not merged.description and old.description:
            merged.description = old.description
        if not merged.artwork_url and old.artwork_url:
            merged.artwork_url = old.artwork_url
        return merged

    episodes = merge_with_cache(
        episodes, load_feed_cache(path), on_match=_preserve_enriched
    )

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
        cache_youtube_subtitles,
        download_youtube_audio,
    )

    path = require_show_folder(show_folder)
    cached = load_feed_cache(path)
    if cached is None:
        raise HTTPException(400, "No cached video list — fetch YouTube first")

    if req.video_ids:
        id_set = set(req.video_ids)
        targets = [ep for ep in cached if ep.guid in id_set]
        if not targets:
            raise HTTPException(404, "No matching videos found for the given IDs")
    else:
        targets = [ep for ep in cached if not ep.removed]

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

            stem = episode_stem(ep, show_path)
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

            # Cache subtitles if requested
            if req.import_subs:
                try:
                    episode_dir = show_path / stem
                    cached_subs = cache_youtube_subtitles(
                        ep.guid,
                        episode_dir,
                        stem,
                        lang=req.sub_lang,
                    )
                    if cached_subs:
                        results[-1]["subs_cached"] = True
                        invalidate_scan_cache(show_path)
                except Exception as exc:
                    logger.warning("Subtitle download failed for {}: {}", stem, exc)

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
    """Download and cache YouTube subtitles (VTT files). Background task."""
    from podcodex.ingest.youtube import cache_youtube_subtitles

    path = require_show_folder(show_folder)
    cached = load_feed_cache(path)
    if cached is None:
        raise HTTPException(400, "No cached video list")

    id_set = set(req.video_ids)
    targets = [ep for ep in cached if ep.guid in id_set]
    if not targets:
        raise HTTPException(404, "No matching videos found")

    def run_import(progress_cb, episodes=targets, show_path=path):
        """Download subtitles for each episode, reporting progress."""
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
        results: list[dict] = []
        reset_pace()
        for i, ep in enumerate(episodes):
            if cancel and cancel.is_set():
                progress_cb(i / total, "Cancelled")
                break

            stem = episode_stem(ep, show_path)
            progress_cb(i / total, f"Downloading subs {i + 1}/{total}: {ep.title[:40]}")
            episode_dir = show_path / stem
            episode_dir.mkdir(parents=True, exist_ok=True)
            save_episode_meta(episode_dir, ep)
            try:
                _pace_request()
                if cache_youtube_subtitles(ep.guid, episode_dir, stem, lang=req.lang):
                    imported += 1
                    consecutive_fails = 0
                    results.append(
                        {"stem": stem, "title": ep.title, "status": "cached"}
                    )
                else:
                    # Not available in this language — not a failure
                    results.append(
                        {"stem": stem, "title": ep.title, "status": "no_subtitles"}
                    )
                    consecutive_fails = 0
            except Exception as exc:
                logger.warning("Subtitle download failed for {}: {}", stem, exc)
                failed += 1
                consecutive_fails += 1
                results.append(
                    {
                        "stem": stem,
                        "title": ep.title,
                        "status": "error",
                        "error": str(exc),
                    }
                )

            # Invalidate per-iteration so the episodes poll picks up new VTTs
            # (and .no_subtitles markers) while the batch is still running.
            invalidate_scan_cache(show_path)

            if consecutive_fails >= _CONSECUTIVE_FAIL_LIMIT:
                remaining = total - i - 1
                progress_cb(
                    (i + 1) / total,
                    f"Stopped — YouTube is rate-limiting requests. "
                    f"Imported {imported}/{total}, {remaining} skipped. "
                    f"Try again later with fewer episodes.",
                )
                break
        return {
            "imported": imported,
            "failed": failed,
            "total": total,
            "throttled": consecutive_fails >= _CONSECUTIVE_FAIL_LIMIT,
            "results": results,
        }

    return submit_task("yt-subs", str(path), run_import)
