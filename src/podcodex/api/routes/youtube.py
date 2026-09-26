"""YouTube routes — fetch video list, download audio, import subtitles."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from podcodex.api.routes._helpers import (
    counted_progress,
    download_item,
    keyed_lock,
    is_downloaded,
    list_show_stems,
    scan_show_stems,
    require_registered_show,
    rss_episode_to_out,
    submit_task,
)
from podcodex.api.schemas import (
    DownloadItemStatus,
    RSSEpisodeOut,
    SubtitleImportResult,
    TaskResponse,
)
from podcodex.ingest.rss import (
    RSSEpisode,
    episode_stem,
    fill_empty_fields,
    load_feed_cache,
    merge_with_cache,
    save_episode_meta,
    save_feed_cache,
)
from podcodex.core.constants import LOCAL_ARTWORK_MARKER
from podcodex.ingest.show import load_show_meta, save_show_meta

router = APIRouter()


def _feed_cache_lock(path: Path):
    """Serializes the feed-cache read-merge-write in youtube_fetch, per show."""
    return keyed_lock("feed-cache", path)


# ── Request models ─────────────────────────────


class YouTubeDownloadRequest(BaseModel):
    video_ids: list[str] | None = None  # None = download all
    import_subs: bool = False  # also import YouTube subtitles as transcript
    sub_lang: str = "en"


class YouTubeSubsRequest(BaseModel):
    video_ids: list[str]
    lang: str = "en"


# ── Routes ─────────────────────────────────────


def _abort_reason(error: str | None) -> str:
    """Why a run stopped after repeated failures, in the user's words.

    The `except Exception` above catches everything `download_youtube_audio`
    and `cache_youtube_subtitles` can raise: a missing ffmpeg, a full disk, a
    geo-block, a private video, a yt-dlp extractor break. Naming all of them
    "YouTube is rate-limiting requests" gave wrong advice for every cause but
    one, so only say that when the error actually looks like a throttle.
    """
    text = (error or "").lower()
    throttled = any(
        marker in text
        for marker in ("429", "too many requests", "rate limit", "throttl")
    )
    if throttled or not error:
        return "YouTube is rate-limiting requests. "
    return f"{error.strip()[:200]} "


@router.post("/{show_folder:path}/youtube/fetch", response_model=list[RSSEpisodeOut])
def youtube_fetch(show_folder: str) -> list[dict]:
    """Refresh the video list for a YouTube show.

    Sync def on purpose: yt-dlp extraction blocks for minutes on big
    channels (network I/O plus rate-limit pacing sleeps). FastAPI runs
    sync handlers on its threadpool, so the event loop stays free and
    other requests (e.g. the show page's episode list) stay responsive.
    """
    from podcodex.ingest.youtube import fetch_youtube

    path = require_registered_show(show_folder)
    meta = load_show_meta(path)
    if not meta or not meta.youtube_url:
        raise HTTPException(400, "No YouTube URL in show.toml")

    try:
        episodes, channel_info = fetch_youtube(meta.youtube_url)
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
        fill_empty_fields(merged, old)
        return merged

    # Serialize the read-merge-write per show: threadpool handlers from two
    # clients would otherwise interleave and drop each other's enrichment.
    with _feed_cache_lock(path):
        episodes = merge_with_cache(
            episodes, load_feed_cache(path), on_match=_preserve_enriched
        )
        save_feed_cache(path, episodes)

    # Artwork upgrade: use the channel info distilled from the extraction we
    # just did (no second crawl). Re-load show.toml before writing: minutes
    # passed since the pre-fetch load, and writing that stale snapshot back
    # would clobber any edits the user made while the fetch ran.
    fresh = channel_info.get("artwork_url", "")
    if fresh:
        current_meta = load_show_meta(path)
        if current_meta and fresh != (current_meta.artwork_url or ""):
            # A locally uploaded cover (LOCAL_ARTWORK_MARKER) is the user's
            # explicit choice; never "upgrade" over it.
            needs_upgrade = current_meta.artwork_url != LOCAL_ARTWORK_MARKER and (
                not current_meta.artwork_url
                or "yt3.googleusercontent.com" not in current_meta.artwork_url
            )
            if needs_upgrade:
                current_meta.artwork_url = fresh
                save_show_meta(path, current_meta)

    existing_stems, audio_stems = scan_show_stems(path)
    return [
        rss_episode_to_out(
            ep, path, existing_stems=existing_stems, audio_stems=audio_stems
        )
        for ep in episodes
    ]


@router.post(
    "/{show_folder:path}/youtube/download",
    response_model=TaskResponse,
)
def youtube_download(
    show_folder: str,
    req: YouTubeDownloadRequest,
    force: bool = False,
) -> TaskResponse:
    """Download YouTube episodes as a background task.

    `force=true` re-downloads even when the audio already exists locally
    (matches the RSS download semantics so the "Re-download audio" button
    works the same on YouTube shows).
    """
    from podcodex.ingest.youtube import (
        cache_youtube_subtitles,
        download_youtube_audio,
    )

    path = require_registered_show(show_folder)
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
        from podcodex.ingest.youtube import _CONSECUTIVE_FAIL_LIMIT, Pacer

        cancel = getattr(progress_cb, "cancel_event", None)
        results = []
        consecutive_fails = 0
        last_error: str | None = None
        total = len(episodes)
        report = counted_progress(progress_cb, total)
        existing_stems = list_show_stems(show_path)
        pacer = Pacer()
        for i, ep in enumerate(episodes):
            if cancel and cancel.is_set():
                progress_cb(i / total, "Cancelled")
                break
            stem = episode_stem(ep, show_path, existing_stems=existing_stems)
            report(i, f"Downloading: {ep.title[:40]}")
            paced = False

            if not force and is_downloaded(show_path, stem):
                results.append(download_item(stem, DownloadItemStatus.EXISTS))
                consecutive_fails = 0
            else:
                # Before the request, after the cancel check: an episode
                # already on disk makes no request and so waits for nothing
                # (a mostly-downloaded show used to sleep for half an hour),
                # and the failure branch below still paces the next attempt.
                pacer.wait()
                paced = True
                try:
                    audio_path = download_youtube_audio(
                        ep.guid,
                        show_path,
                        stem,
                        force=force,
                    )
                    # Save episode metadata
                    episode_dir = show_path / stem
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    save_episode_meta(episode_dir, ep)

                    results.append(
                        download_item(
                            stem,
                            DownloadItemStatus.DOWNLOADED,
                            audio_path=str(audio_path),
                        )
                    )
                    consecutive_fails = 0
                    invalidate_scan_cache(show_path)
                except Exception as exc:
                    logger.exception("Failed to download {}", ep.guid)
                    results.append(
                        download_item(stem, DownloadItemStatus.FAILED, error=str(exc))
                    )
                    consecutive_fails += 1
                    last_error = str(exc)
                    if consecutive_fails >= _CONSECUTIVE_FAIL_LIMIT:
                        remaining = total - i - 1
                        progress_cb(
                            (i + 1) / total,
                            f"Stopped — {_abort_reason(last_error)}"
                            f"{len(results)}/{total} processed, {remaining} skipped. "
                            f"Try again later with fewer episodes.",
                        )
                        break
                    continue

            # Cache subtitles if requested
            if req.import_subs:
                # One pacer tick per episode that talks to YouTube: the
                # backoff is count-based, so a second tick for the subtitle
                # call reached the maximum delay at half the episodes.
                if not paced:
                    pacer.wait()
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

        return results

    return submit_task("yt-download", str(path), run_downloads)


@router.post(
    "/{show_folder:path}/youtube/import-subs",
    response_model=TaskResponse,
)
def youtube_import_subs(
    show_folder: str,
    req: YouTubeSubsRequest,
) -> TaskResponse:
    """Download and cache YouTube subtitles (VTT files). Background task."""
    from podcodex.ingest.youtube import cache_youtube_subtitles

    path = require_registered_show(show_folder)
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
        from podcodex.ingest.youtube import _CONSECUTIVE_FAIL_LIMIT, Pacer

        cancel = getattr(progress_cb, "cancel_event", None)
        imported = 0
        failed = 0
        consecutive_fails = 0
        last_error: str | None = None
        total = len(episodes)
        report = counted_progress(progress_cb, total)
        existing_stems = list_show_stems(show_path)
        results: list[dict] = []
        pacer = Pacer()
        for i, ep in enumerate(episodes):
            if cancel and cancel.is_set():
                progress_cb(i / total, "Cancelled")
                break

            stem = episode_stem(ep, show_path, existing_stems=existing_stems)
            report(i, f"Downloading subs: {ep.title[:40]}")
            episode_dir = show_path / stem
            episode_dir.mkdir(parents=True, exist_ok=True)
            save_episode_meta(episode_dir, ep)
            try:
                pacer.wait()
                if cache_youtube_subtitles(ep.guid, episode_dir, stem, lang=req.lang):
                    imported += 1
                    consecutive_fails = 0
                    results.append(
                        download_item(stem, DownloadItemStatus.CACHED, title=ep.title)
                    )
                else:
                    # Not available in this language — not a failure
                    results.append(
                        download_item(
                            stem, DownloadItemStatus.NO_SUBTITLES, title=ep.title
                        )
                    )
                    consecutive_fails = 0
            except Exception as exc:
                logger.warning("Subtitle download failed for {}: {}", stem, exc)
                failed += 1
                consecutive_fails += 1
                last_error = str(exc)
                results.append(
                    download_item(
                        stem, DownloadItemStatus.FAILED, title=ep.title, error=str(exc)
                    )
                )

            # Invalidate per-iteration so the episodes poll picks up new VTTs
            # (and .no_subtitles markers) while the batch is still running.
            invalidate_scan_cache(show_path)

            if consecutive_fails >= _CONSECUTIVE_FAIL_LIMIT:
                remaining = total - i - 1
                progress_cb(
                    (i + 1) / total,
                    f"Stopped — {_abort_reason(last_error)}"
                    f"Imported {imported}/{total}, {remaining} skipped. "
                    f"Try again later with fewer episodes.",
                )
                break
        return SubtitleImportResult(
            imported=imported,
            failed=failed,
            total=total,
            throttled=consecutive_fails >= _CONSECUTIVE_FAIL_LIMIT,
            results=results,
        ).model_dump(mode="json", exclude_none=True)

    return submit_task("yt-subs", str(path), run_import)
