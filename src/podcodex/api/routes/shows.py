"""Show and episode management routes."""

from __future__ import annotations

import logging
import shutil
from dataclasses import fields
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from podcodex.api.routes._helpers import is_downloaded, require_show_folder
from podcodex.api.schemas import EpisodeOut, ShowMeta, UnifiedEpisodeOut
from podcodex.ingest.folder import EpisodeInfo, invalidate_scan_cache, scan_folder
from podcodex.ingest.rss import episode_stem, load_episode_meta, load_feed_cache
from podcodex.ingest.show import ShowMeta as _ShowMeta
from podcodex.ingest.show import load_show_meta, save_show_meta

router = APIRouter()
logger = logging.getLogger(__name__)

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
        logger.info("Moved show folder %s → %s", old_path, new_path)
    else:
        # Just create the new folder with show metadata, leave files behind
        new_path.mkdir(parents=True, exist_ok=True)
        meta = load_show_meta(old_path)
        if meta:
            save_show_meta(new_path, meta)
        logger.info(
            "Created new show folder %s (files remain at %s)", new_path, old_path
        )

    # Update config.json: replace old path with new
    from podcodex.api.routes.config import _load, _save

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
