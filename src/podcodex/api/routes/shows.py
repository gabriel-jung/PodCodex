"""Show and episode management routes."""

from __future__ import annotations

import re
import shutil
from dataclasses import fields
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from podcodex.api.routes._helpers import is_downloaded, require_show_folder
from podcodex.api.routes.config import _load, _register_folder, _save
from podcodex.api.schemas import (
    CreateFromRSSRequest,
    CreateFromRSSResponse,
    EpisodeOut,
    RegisterShowRequest,
    ShowMeta,
    UnifiedEpisodeOut,
)
from podcodex.ingest.folder import EpisodeInfo, invalidate_scan_cache, scan_folder
from podcodex.ingest.rss import (
    episode_stem,
    feed_artwork,
    fetch_feed,
    load_episode_meta,
    load_feed_cache,
    save_feed_cache,
)
from podcodex.ingest.show import ShowMeta as _ShowMeta
from podcodex.ingest.show import load_show_meta, save_show_meta

router = APIRouter()


# ── Show listing & creation ─────────────────


class ShowSummary(BaseModel):
    name: str
    path: str
    episode_count: int = 0
    has_rss: bool = False
    artwork_url: str = ""
    last_rss_update: str | None = None  # ISO timestamp of last feed cache write


@router.get("/", response_model=list[ShowSummary])
async def list_shows() -> list[ShowSummary]:
    """List all known show folders."""
    cfg = _load()
    shows: list[ShowSummary] = []

    for folder_path in cfg.show_folders:
        child = Path(folder_path)
        if not child.is_dir():
            continue

        meta = load_show_meta(child)
        name = (meta.name if meta else None) or child.name
        artwork = (meta.artwork_url if meta else "") or ""

        audio_count = sum(
            1
            for f in child.iterdir()
            if f.is_file() and f.suffix in (".mp3", ".m4a", ".wav", ".ogg", ".flac")
        )

        feed_cache = child / ".feed_cache.json"
        has_rss = feed_cache.exists() or bool(meta and meta.rss_url)
        last_rss: str | None = None
        if feed_cache.exists():
            from datetime import datetime, timezone

            last_rss = datetime.fromtimestamp(
                feed_cache.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        shows.append(
            ShowSummary(
                name=name,
                path=str(child),
                episode_count=audio_count,
                has_rss=has_rss,
                artwork_url=artwork,
                last_rss_update=last_rss,
            )
        )
    return shows


@router.post("/from-rss", response_model=CreateFromRSSResponse)
async def create_from_rss(req: CreateFromRSSRequest) -> CreateFromRSSResponse:
    """Fetch an RSS feed and create a show folder for it."""
    save_base = Path(req.save_path).expanduser()
    if not save_base.is_dir():
        raise HTTPException(400, f"Save path does not exist: {req.save_path}")

    # Fetch the feed
    episodes = fetch_feed(req.rss_url)
    if not episodes:
        raise HTTPException(502, "Feed returned no episodes")

    # Determine folder name
    folder_name = req.folder_name.strip()
    if not folder_name:
        folder_name = re.sub(r"https?://", "", req.rss_url)
        folder_name = re.sub(r"[^a-zA-Z0-9]+", "_", folder_name).strip("_")[:40]

    show_path = save_base / folder_name
    show_path.mkdir(parents=True, exist_ok=True)

    # Get artwork: prefer what was passed (from search), fall back to feed
    artwork = req.artwork_url or feed_artwork(req.rss_url)

    # Save show metadata — use the display name from search, fall back to folder name
    show_name = req.name.strip() or folder_name
    save_show_meta(
        show_path,
        _ShowMeta(
            name=show_name,
            rss_url=req.rss_url,
            artwork_url=artwork,
        ),
    )

    # Cache the feed
    save_feed_cache(show_path, episodes)

    # Register in config
    cfg = _load()
    _register_folder(cfg, str(show_path))

    return CreateFromRSSResponse(
        folder=str(show_path),
        name=folder_name,
        episode_count=len(episodes),
    )


@router.post("/register")
async def register_show(req: RegisterShowRequest) -> dict:
    """Register an existing folder as a known show."""
    p = Path(req.path).expanduser().resolve()
    if not p.is_dir():
        raise HTTPException(400, f"Not a directory: {req.path}")
    cfg = _load()
    _register_folder(cfg, str(p))
    return {"status": "ok", "path": str(p)}


# ── Episode serialization ────────────────────

_EPISODE_FIELDS = {f.name for f in fields(EpisodeInfo)}


def _episode_to_dict(ep: EpisodeInfo) -> dict:
    """Serialize an EpisodeInfo to a JSON-safe dict."""
    d: dict = {}
    for f in fields(ep):
        val = getattr(ep, f.name)
        if isinstance(val, Path):
            val = str(val)
        d[f.name] = val
    return d


# ── Show metadata ────────────────────────────


@router.get("/{show_folder:path}/meta", response_model=ShowMeta)
async def get_show_meta(show_folder: str) -> ShowMeta:
    path = require_show_folder(show_folder)
    meta = load_show_meta(path)
    if meta is None:
        return ShowMeta(name=path.name)
    return ShowMeta(
        name=meta.name,
        rss_url=meta.rss_url,
        language=meta.language,
        speakers=meta.speakers,
        artwork_url=meta.artwork_url,
    )


@router.put("/{show_folder:path}/meta")
async def update_show_meta(show_folder: str, meta: ShowMeta) -> dict:
    path = Path(show_folder)
    path.mkdir(parents=True, exist_ok=True)
    save_show_meta(
        path,
        _ShowMeta(
            name=meta.name,
            rss_url=meta.rss_url,
            language=meta.language,
            speakers=meta.speakers,
            artwork_url=meta.artwork_url,
        ),
    )
    return {"status": "saved"}


# ── Episode listing ──────────────────────────


@router.get("/{show_folder:path}/episodes", response_model=list[EpisodeOut])
async def list_episodes(show_folder: str) -> list[dict]:
    path = require_show_folder(show_folder)
    episodes = scan_folder(path)
    return [_episode_to_dict(ep) for ep in episodes]


# ── Unified episodes (local + RSS merged) ───


@router.get(
    "/{show_folder:path}/unified",
    response_model=list[UnifiedEpisodeOut],
)
async def unified_episodes(show_folder: str) -> list[dict]:
    """Return a merged list of RSS + local episodes."""
    path = require_show_folder(show_folder)

    local = {ep.stem: ep for ep in scan_folder(path)}
    rss = load_feed_cache(path) or []

    result: list[dict] = []
    seen_stems: set[str] = set()

    # RSS episodes first (preserves feed order)
    for r in rss:
        stem = episode_stem(r)
        ep = local.get(stem)
        if stem:
            seen_stems.add(stem)
        result.append(
            {
                "id": r.guid,
                "title": r.title,
                "stem": stem,
                "pub_date": r.pub_date,
                "description": r.description or "",
                "audio_url": r.audio_url or None,
                "duration": r.duration,
                "episode_number": r.episode_number,
                "audio_path": str(ep.audio_path) if ep and ep.audio_path else None,
                "downloaded": is_downloaded(path, stem) if stem else False,
                "transcribed": ep.transcribed if ep else False,
                "polished": ep.polished if ep else False,
                "indexed": ep.indexed if ep else False,
                "synthesized": ep.synthesized if ep else False,
                "translations": ep.translations if ep else [],
                "artwork_url": r.artwork_url or "",
                "raw_transcript": ep.raw_transcript if ep else False,
                "validated_transcript": ep.validated_transcript if ep else False,
            }
        )

    # Local-only episodes (no RSS match) — restore cached RSS metadata if available
    for ep in local.values():
        if ep.stem in seen_stems:
            continue
        meta = load_episode_meta(ep.output_dir) if ep.output_dir else None
        result.append(
            {
                "id": meta.guid if meta else ep.stem,
                "title": (meta.title if meta else None) or ep.title or ep.stem,
                "stem": ep.stem,
                "pub_date": meta.pub_date if meta else None,
                "description": (meta.description or "") if meta else "",
                "audio_url": (meta.audio_url or None) if meta else None,
                "duration": meta.duration if meta else 0,
                "episode_number": meta.episode_number if meta else None,
                "audio_path": str(ep.audio_path) if ep.audio_path else None,
                "downloaded": bool(ep.audio_path),
                "transcribed": ep.transcribed,
                "polished": ep.polished,
                "indexed": ep.indexed,
                "synthesized": ep.synthesized,
                "translations": ep.translations,
                "artwork_url": (meta.artwork_url or "") if meta else "",
                "raw_transcript": ep.raw_transcript,
                "validated_transcript": ep.validated_transcript,
            }
        )

    return result


# ── Move / rename show folder ──────────────


class MoveShowRequest(BaseModel):
    new_path: str
    move_files: bool = True


@router.post("/{show_folder:path}/move")
async def move_show(show_folder: str, req: MoveShowRequest) -> dict:
    """Move or rename a show folder, optionally relocating all files."""
    old_path = require_show_folder(show_folder)
    new_path = Path(req.new_path).expanduser().resolve()

    if new_path == old_path.resolve():
        raise HTTPException(400, "Source and destination are the same")

    if new_path.exists() and any(new_path.iterdir()):
        raise HTTPException(
            409, f"Destination already exists and is not empty: {new_path}"
        )

    # Check no tasks are running on this show
    from podcodex.api.tasks import task_manager

    active = task_manager.get_active(show_folder)
    if active:
        raise HTTPException(
            409,
            f"Task {active.task_id} is running on this show — wait for it to finish",
        )

    if req.move_files:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))
        logger.info("Moved show folder {} → {}", old_path, new_path)
    else:
        # Just create the new folder with show metadata, leave files behind
        new_path.mkdir(parents=True, exist_ok=True)
        meta = load_show_meta(old_path)
        if meta:
            save_show_meta(new_path, meta)
        logger.info(
            "Created new show folder {} (files remain at {})", new_path, old_path
        )

    # Update config.json: replace old path with new
    cfg = _load()
    old_resolved = str(old_path.resolve())
    cfg.show_folders = [
        str(new_path) if str(Path(p).resolve()) == old_resolved else p
        for p in cfg.show_folders
    ]
    _save(cfg)

    # Invalidate caches
    invalidate_scan_cache(old_path)
    invalidate_scan_cache(new_path)

    return {"status": "moved", "new_path": str(new_path)}
