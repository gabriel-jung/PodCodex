"""RSS feed routes — fetch, cache, and download episodes."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger

from podcodex.api.routes._helpers import (
    is_downloaded,
    require_show_folder,
    rss_episode_to_out,
    submit_task,
)
from podcodex.api.schemas import RSSEpisodeOut, TaskResponse
from podcodex.ingest.rss import (
    download_audio,
    episode_stem,
    feed_artwork,
    fetch_feed,
    load_feed_cache,
    merge_with_cache,
    save_feed_cache,
)
from podcodex.ingest.show import load_show_meta, save_show_meta

router = APIRouter()


@router.post("/{show_folder:path}/rss/fetch", response_model=list[RSSEpisodeOut])
async def rss_fetch(show_folder: str, rss_url: str | None = None) -> list[dict]:
    """Fetch (or refresh) the RSS feed for a show. Uses show.toml rss_url if not provided."""
    path = require_show_folder(show_folder)

    meta = load_show_meta(path)
    if not rss_url:
        if meta and meta.rss_url:
            rss_url = meta.rss_url
    if not rss_url:
        raise HTTPException(400, "No RSS URL provided and none in show.toml")

    try:
        # feedparser blocks on network — keep it off the event loop
        episodes = await asyncio.to_thread(fetch_feed, rss_url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
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
            return [rss_episode_to_out(ep, path) for ep in cached]
        raise HTTPException(502, "Feed returned no episodes (parse error or empty)")

    # Keep episodes pulled from the feed flagged ``removed=True`` rather than
    # silently dropping them — their local outputs stay visible in the UI.
    episodes = merge_with_cache(episodes, load_feed_cache(path))
    save_feed_cache(path, episodes)

    # Upgrade artwork if missing or low-res (e.g. old 60px iTunes thumbnails)
    if meta:
        current = meta.artwork_url or ""
        if not current or "60x60" in current or "artworkUrl60" in current:
            fresh = await asyncio.to_thread(feed_artwork, rss_url)
            if fresh and fresh != current:
                meta.artwork_url = fresh
                save_show_meta(path, meta)

    return [rss_episode_to_out(ep, path) for ep in episodes]


@router.get("/{show_folder:path}/rss/cache", response_model=list[RSSEpisodeOut])
async def rss_cache(show_folder: str) -> list[dict]:
    """Return cached RSS feed data (no network call)."""
    path = Path(show_folder)
    cached = load_feed_cache(path)
    if cached is None:
        return []
    return [rss_episode_to_out(ep, path) for ep in cached]


@router.post(
    "/{show_folder:path}/rss/download",
    response_model=TaskResponse,
)
async def rss_download(
    show_folder: str,
    guids: list[str] | None = None,
    force: bool = False,
) -> TaskResponse:
    """Download episodes as a background task. Progress is streamed via WebSocket."""
    path = Path(show_folder)
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

        for i, ep in enumerate(episodes):
            if cancel and cancel.is_set():
                progress_cb(i / total, f"Cancelled — {_summary()}")
                break

            stem = episode_stem(ep, show_path)
            progress_cb(i / total, f"[{i + 1}/{total}] Downloading…")

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

            progress_cb((i + 1) / total, f"[{i + 1}/{total}] {_summary()}")

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
