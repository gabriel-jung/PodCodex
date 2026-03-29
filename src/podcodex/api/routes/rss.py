"""RSS feed routes — fetch, cache, and download episodes."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException

from podcodex.api.routes._helpers import is_downloaded, require_show_folder, submit_task
from podcodex.api.schemas import RSSEpisodeOut, TaskResponse
from podcodex.ingest.rss import (
    RSSEpisode,
    download_audio,
    episode_stem,
    fetch_feed,
    load_feed_cache,
    save_feed_cache,
)
from podcodex.ingest.show import load_show_meta

router = APIRouter()


def _rss_to_out(ep: RSSEpisode, show_folder: Path) -> dict:
    """Convert an RSSEpisode to an RSSEpisodeOut dict."""
    stem = episode_stem(ep)
    return {
        **asdict(ep),
        "local_stem": stem,
        "downloaded": is_downloaded(show_folder, stem),
    }


@router.post("/{show_folder:path}/rss/fetch", response_model=list[RSSEpisodeOut])
async def rss_fetch(show_folder: str, rss_url: str | None = None) -> list[dict]:
    """Fetch (or refresh) the RSS feed for a show. Uses show.toml rss_url if not provided."""
    path = require_show_folder(show_folder)

    if not rss_url:
        meta = load_show_meta(path)
        if meta and meta.rss_url:
            rss_url = meta.rss_url
    if not rss_url:
        raise HTTPException(400, "No RSS URL provided and none in show.toml")

    episodes = fetch_feed(rss_url)
    if not episodes:
        raise HTTPException(502, "Feed returned no episodes (parse error or empty)")

    save_feed_cache(path, episodes)
    return [_rss_to_out(ep, path) for ep in episodes]


@router.get("/{show_folder:path}/rss/cache", response_model=list[RSSEpisodeOut])
async def rss_cache(show_folder: str) -> list[dict]:
    """Return cached RSS feed data (no network call)."""
    path = Path(show_folder)
    cached = load_feed_cache(path)
    if cached is None:
        return []
    return [_rss_to_out(ep, path) for ep in cached]


@router.post(
    "/{show_folder:path}/rss/download",
    response_model=TaskResponse,
)
async def rss_download(
    show_folder: str,
    guids: list[str] | None = None,
) -> TaskResponse:
    """Download episodes as a background task. Progress is streamed via WebSocket."""
    path = Path(show_folder)
    cached = load_feed_cache(path)
    if cached is None:
        raise HTTPException(400, "No cached feed — fetch RSS first")

    targets = cached
    if guids:
        guid_set = set(guids)
        targets = [ep for ep in cached if ep.guid in guid_set]
        if not targets:
            raise HTTPException(404, "No matching episodes found for the given GUIDs")

    def run_downloads(progress_cb, episodes=targets, show_path=path):
        from podcodex.ingest.folder import invalidate_scan_cache

        cancel = getattr(progress_cb, "cancel_event", None)
        results = []
        total = len(episodes)
        for i, ep in enumerate(episodes):
            if cancel and cancel.is_set():
                progress_cb(i / total, "Cancelled")
                break

            stem = episode_stem(ep)
            progress_cb(i / total, f"Downloading {i + 1}/{total}")

            if is_downloaded(show_path, stem):
                results.append({"stem": stem, "status": "exists"})
                continue
            if not ep.audio_url:
                results.append({"stem": stem, "status": "no_audio"})
                continue

            audio_path = download_audio(ep, show_path)
            if audio_path:
                results.append(
                    {
                        "stem": stem,
                        "status": "downloaded",
                        "audio_path": str(audio_path),
                    }
                )
                invalidate_scan_cache(show_path)
            else:
                results.append({"stem": stem, "status": "failed"})

            # Small delay between downloads to be respectful to servers
            if total > 1:
                time.sleep(1)

        return results

    return submit_task("download", str(path), run_downloads)
