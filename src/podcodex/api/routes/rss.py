"""RSS feed routes — fetch, cache, and download episodes."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException
from loguru import logger

from podcodex.api.routes._helpers import (
    counted_progress,
    is_downloaded,
    list_show_stems,
    require_registered_show,
    rss_episode_to_out,
    scan_show_stems,
    submit_task,
)
from podcodex.api.schemas import RSSEpisodeOut, TaskResponse
from podcodex.ingest.rss import (
    download_audio,
    episode_stem,
    fetch_feed_with_artwork,
    load_feed_cache,
    merge_with_cache,
    save_feed_cache,
)
from podcodex.core.constants import LOCAL_ARTWORK_MARKER
from podcodex.ingest.show import load_show_meta, save_show_meta

router = APIRouter()


@router.post("/{show_folder:path}/rss/fetch", response_model=list[RSSEpisodeOut])
async def rss_fetch(show_folder: str, rss_url: str | None = None) -> list[dict]:
    """Fetch (or refresh) the RSS feed for a show. Uses show.toml rss_url if not provided."""
    path = require_registered_show(show_folder)

    meta = load_show_meta(path)
    if not rss_url:
        if meta and meta.rss_url:
            rss_url = meta.rss_url
    if not rss_url:
        raise HTTPException(400, "No RSS URL provided and none in show.toml")

    import httpx

    feed_art = ""
    try:
        # feedparser blocks on network — keep it off the event loop.
        # `fetch_feed_with_artwork`, not `fetch_feed`: the artwork upgrade
        # below used to make its own `feed_artwork` call, downloading and
        # re-parsing the same multi-megabyte XML a second time on a request
        # that already had the parsed document in hand.
        episodes, feed_art = await asyncio.to_thread(fetch_feed_with_artwork, rss_url)
    except ValueError as exc:
        # Scheme rejection and other bad input: the caller's URL is wrong,
        # and no cache can stand in for it.
        raise HTTPException(400, str(exc)) from exc
    except httpx.HTTPError as exc:
        # DNS, a captive portal, a CDN 503. The cached-feed fallback below
        # exists for exactly this, but these are not ValueError, so they
        # used to escape as an opaque 500 with the cache sitting right
        # there. Fall through with no episodes to reach it.
        logger.warning("fetch_feed failed for {}: {}", rss_url, exc)
        episodes = []
    if not episodes:
        # Transient failures (DNS, captive portal, feedparser bozo) shouldn't
        # block the show page when we already have a cache to serve.
        cached = load_feed_cache(path)
        if cached:
            logger.warning(
                "fetch_feed returned no episodes for {}; serving cache ({} episodes)",
                rss_url,
                len(cached),
            )
            stems, audio = scan_show_stems(path)
            return [
                rss_episode_to_out(ep, path, existing_stems=stems, audio_stems=audio)
                for ep in cached
            ]
        raise HTTPException(502, "Feed returned no episodes (parse error or empty)")

    # Keep episodes pulled from the feed flagged ``removed=True`` rather than
    # silently dropping them — their local outputs stay visible in the UI.
    episodes = merge_with_cache(episodes, load_feed_cache(path))
    save_feed_cache(path, episodes)

    # Upgrade artwork if missing or low-res (e.g. old 60px iTunes thumbnails).
    # A locally uploaded cover (LOCAL_ARTWORK_MARKER) is never upgraded over.
    if meta:
        current = meta.artwork_url or ""
        if current != LOCAL_ARTWORK_MARKER and (
            not current or "60x60" in current or "artworkUrl60" in current
        ):
            fresh = feed_art
            if fresh and fresh != current:
                meta.artwork_url = fresh
                save_show_meta(path, meta)

    stems, audio = scan_show_stems(path)
    return [
        rss_episode_to_out(ep, path, existing_stems=stems, audio_stems=audio)
        for ep in episodes
    ]


@router.post(
    "/{show_folder:path}/rss/download",
    response_model=TaskResponse,
)
def rss_download(
    show_folder: str,
    guids: list[str] | None = None,
    force: bool = False,
) -> TaskResponse:
    """Download episodes as a background task. Progress is streamed via WebSocket."""
    path = require_registered_show(show_folder)
    cached = load_feed_cache(path)
    if cached is None:
        raise HTTPException(400, "No cached feed — fetch RSS first")

    if guids:
        guid_set = set(guids)
        targets = [ep for ep in cached if ep.guid in guid_set]
        if not targets:
            raise HTTPException(404, "No matching episodes found for the given GUIDs")
    else:
        targets = [ep for ep in cached if not ep.removed]

    def run_downloads(progress_cb, episodes=targets, show_path=path, force_dl=force):
        """Download each episode in sequence, reporting progress via callback."""
        from podcodex.ingest.folder import invalidate_scan_cache

        cancel = getattr(progress_cb, "cancel_event", None)
        results = []
        total = len(episodes)
        downloaded = 0
        skipped = 0
        failed = 0
        consecutive_failures = 0
        last_error: str | None = None
        abort_reason: str | None = None

        def _summary() -> str:
            parts = []
            if downloaded:
                parts.append(f"{downloaded} downloaded")
            if skipped:
                parts.append(f"{skipped} skipped")
            if failed:
                parts.append(f"{failed} failed")
            return " · ".join(parts) if parts else ""

        report = counted_progress(progress_cb, total)
        # Hoisted: episode_stem scandirs the show folder on every call that
        # is not handed a stem set, and download_audio resolves the stem the
        # same way, so the loop paid two listings per episode. `is_downloaded`
        # below stays a live check, since this loop is what changes the answer.
        existing_stems = list_show_stems(show_path)
        for i, ep in enumerate(episodes):
            if cancel and cancel.is_set():
                progress_cb(i / total, f"Cancelled — {_summary()}")
                break

            stem = episode_stem(ep, show_path, existing_stems=existing_stems)
            report(i, "Downloading…")

            if not force_dl and is_downloaded(show_path, stem):
                skipped += 1
                consecutive_failures = 0
                results.append({"stem": stem, "status": "exists"})
                continue
            if not ep.audio_url:
                skipped += 1
                results.append({"stem": stem, "status": "no_audio"})
                continue

            audio_path, error = download_audio(ep, show_path, force=force_dl)
            if audio_path:
                downloaded += 1
                consecutive_failures = 0
                results.append(
                    {
                        "stem": stem,
                        "status": "downloaded",
                        "audio_path": str(audio_path),
                    }
                )
                invalidate_scan_cache(show_path)
            else:
                failed += 1
                consecutive_failures += 1
                last_error = error
                results.append({"stem": stem, "status": "failed", "error": error})
                # Stop the batch on repeated rate-limits / outages — retrying
                # every episode past the 3rd consecutive failure just wastes
                # time and extends the rate-limit window.
                if consecutive_failures >= 3 and total > 1:
                    abort_reason = error or "repeated failures"
                    progress_cb(
                        (i + 1) / total,
                        f"Stopped after 3 failures ({abort_reason}) — {_summary()}",
                    )
                    break

            report(i, _summary(), frac=(i + 1) / total)

            # Delay between actual downloads to be respectful to servers
            if total > 1:
                time.sleep(5)

        if abort_reason:
            final = f"Stopped: {abort_reason} — {_summary()}"
        elif failed and not downloaded:
            final = f"Failed: {last_error or 'all downloads failed'}"
        else:
            final = _summary() or "Done"
        progress_cb(1.0, final)
        return results

    return submit_task("download", f"download:{path}", run_downloads)
