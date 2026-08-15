"""Show and episode management routes."""

from __future__ import annotations

import asyncio
import re
import shutil
from collections.abc import Container
from dataclasses import fields
from pathlib import Path
from typing import NamedTuple

import hashlib
import urllib.request

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from podcodex.api.routes._helpers import (
    apply_broadcast_pattern,
    bad_path_component,
    list_show_stems,
    require_show_folder,
)
from podcodex.bundle.conflicts import rename_suffix
from podcodex.core._utils import MTIME_SETTLE_SECONDS, atomic_write
from podcodex.core.app_config import AppConfig, mutate_config
from podcodex.api.routes.config import _load, _register_folder
from podcodex.api.schemas import (
    BroadcastPreviewOut,
    CreateFromRSSRequest,
    CreateFromRSSResponse,
    CreateFromYouTubeRequest,
    CreateFromYouTubeResponse,
    EpisodeOut,
    EpisodeSpeakerEntry,
    EpisodeSpeakersResponse,
    EpisodeStatusOut,
    PipelineDefaultsSchema,
    RegisterShowRequest,
    ShowMeta,
    SpeakerEpisodeEntry,
    SpeakerRosterEntry,
    SpeakerRosterResponse,
    UnifiedEpisodeOut,
)
from podcodex.core.constants import AUDIO_EXTENSIONS
from podcodex.core.llm_failures import rejected_steps
from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db
from podcodex.core.versions import STEP_FLAG, is_edited, step_ext
from podcodex.ingest.folder import (
    EpisodeInfo,
    invalidate_scan_cache,
    lance_indexed_stems,
    scan_folder,
)
from podcodex.ingest.rss import (
    episode_stem,
    feed_cache_episode_count,
    fetch_feed_with_artwork,
    load_episode_meta,
    load_feed_cache,
    save_feed_cache,
)
from podcodex.core.translate import clean_translations
from podcodex.ingest.show import PipelineDefaults as _PipelineDefaults
from podcodex.ingest.show import ShowMeta as _ShowMeta
from podcodex.ingest.show import load_show_meta, save_show_meta

router = APIRouter()


# ── Show listing & creation ─────────────────


class ShowSummary(BaseModel):
    name: str
    path: str
    episode_count: int = 0  # downloaded audio files on disk
    feed_episode_count: int | None = (
        None  # total episodes in feed cache (RSS/YouTube), None if no feed
    )
    has_rss: bool = False
    has_youtube: bool = False
    artwork_url: str = ""
    last_rss_update: str | None = None  # ISO timestamp of last feed cache write
    # Per-stage progress aggregates from pipeline_db. All None when no pipeline.db file.
    pipeline_total_count: int | None = (
        None  # rows in pipeline_db (denominator for percentages)
    )
    transcribed_count: int | None = None
    transcribed_edited_count: int | None = None
    corrected_count: int | None = None
    corrected_edited_count: int | None = None
    translated_count: int | None = None
    translated_edited_count: int | None = None
    synthesized_count: int | None = None
    indexed_count: int | None = None
    verified_count: int | None = None  # episodes with a verified pointer


@router.get("/", response_model=list[ShowSummary])
def list_shows() -> list[ShowSummary]:
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
        has_youtube = bool(meta and meta.youtube_url)
        last_rss: str | None = None
        feed_count: int | None = None
        if feed_cache.exists():
            from datetime import datetime, timezone

            last_rss = datetime.fromtimestamp(
                feed_cache.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            feed_count = feed_cache_episode_count(child)

        # Per-stage progress aggregates: only computed when pipeline.db file
        # already exists (skip otherwise to avoid creating an empty DB file
        # for every feed-only show on home-page load).
        pipeline_total = transcribed = corrected = translated = synthesized = (
            indexed
        ) = None
        transcribed_edited = corrected_edited = translated_edited = None
        verified = None
        if (child / "pipeline.db").is_file():
            try:
                db = get_pipeline_db(child)
                agg = db.aggregate_status()
                pipeline_total = agg["total"]
                transcribed = agg["transcribed"]
                transcribed_edited = agg["transcribed_edited"]
                corrected = agg["corrected"]
                corrected_edited = agg["corrected_edited"]
                translated = agg["translated"]
                translated_edited = agg["translated_edited"]
                synthesized = agg["synthesized"]
                indexed = agg["indexed"]
                verified = len(db.stems_with_verified())
            except Exception as exc:
                logger.warning("aggregate_status failed for {}: {}", child, exc)

        shows.append(
            ShowSummary(
                name=name,
                path=str(child),
                episode_count=audio_count,
                feed_episode_count=feed_count,
                has_rss=has_rss,
                has_youtube=has_youtube,
                artwork_url=artwork,
                last_rss_update=last_rss,
                pipeline_total_count=pipeline_total,
                transcribed_count=transcribed,
                transcribed_edited_count=transcribed_edited,
                corrected_count=corrected,
                corrected_edited_count=corrected_edited,
                translated_count=translated,
                translated_edited_count=translated_edited,
                synthesized_count=synthesized,
                indexed_count=indexed,
                verified_count=verified,
            )
        )
    return shows


# ── Files bucket (standalone audio imports) ──

FILES_BUCKET_NAME = "Files"


class FilesImportRequest(BaseModel):
    file_path: str
    name: str | None = None


class FilesImportResponse(BaseModel):
    folder: str
    stem: str


def _files_bucket_path(cfg) -> Path:
    root = Path(cfg.default_save_path or "~").expanduser()
    return root / FILES_BUCKET_NAME


def _ensure_files_bucket() -> Path:
    """Return the registered Files bucket, creating + registering on first use.

    A candidate path is usable only when nothing exists there yet or it is a
    plain local folder (no feed cache). Feed-backed shows, even registered
    ones, and stray files at the path are never touched; the bucket shifts to
    ``Files-2``, ``Files-3``, ... instead.
    """
    cfg = _load()
    base = _files_bucket_path(cfg)
    registered = {str(Path(p).resolve()) for p in cfg.show_folders}
    for n in range(1, 100):
        bucket = base if n == 1 else base.parent / f"{FILES_BUCKET_NAME}-{n}"
        if bucket.exists() and not bucket.is_dir():
            continue
        if bucket.is_dir() and (bucket / ".feed_cache.json").exists():
            continue
        if str(bucket.resolve()) in registered:
            return bucket
        bucket.mkdir(parents=True, exist_ok=True)
        if load_show_meta(bucket) is None:
            save_show_meta(bucket, _ShowMeta(name=bucket.name))
        _register_folder(cfg, str(bucket))
        return bucket
    raise HTTPException(500, "Could not allocate a Files bucket folder")


@router.post("/files/import", response_model=FilesImportResponse)
async def import_local_file(req: FilesImportRequest) -> FilesImportResponse:
    """Copy a standalone audio file into the Files bucket show."""
    src = Path(req.file_path).expanduser()
    if not src.is_file():
        raise HTTPException(404, f"File not found: {req.file_path}")
    ext = src.suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(400, f"Not an audio file: {src.name}")

    stem = (req.name or src.stem).strip()
    if bad_path_component(stem):
        raise HTTPException(400, f"Invalid name: {stem!r}")

    bucket = _ensure_files_bucket()
    # The folder scanner keys episodes by stem alone, so any same-stem audio
    # file (regardless of extension) or output dir counts as a collision.
    # list_show_stems is the scanner-aligned set of both.
    taken = list_show_stems(bucket)
    if stem in taken:
        raise HTTPException(
            409, detail={"suggested": rename_suffix(stem, taken, suffix="")}
        )

    dest = bucket / f"{stem}{ext}"
    try:
        # Off-thread: a multi-GB copy must not block the event loop.
        # atomic_write's temp naming also keeps a crash-abandoned copy
        # visible to the recovery reaper.
        await asyncio.to_thread(atomic_write, dest, lambda p: shutil.copyfile(src, p))
    except OSError as exc:
        raise HTTPException(500, f"Copy failed: {exc}")

    invalidate_scan_cache(bucket)
    logger.info("Imported standalone file {} -> {}", src, dest)
    return FilesImportResponse(folder=str(bucket), stem=stem)


# ── Artwork caching ────────────────────────────


_ARTWORK_STEM = "artwork"
_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _url_hash(url: str) -> str:
    """Short hash of a URL — used to detect when the source URL changes."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _find_cached_artwork(show_path: Path) -> Path | None:
    """Return the cached artwork file if it exists."""
    for ext in _IMG_EXTENSIONS:
        p = show_path / f"{_ARTWORK_STEM}{ext}"
        if p.exists():
            return p
    return None


def _download_artwork(url: str, show_path: Path) -> Path | None:
    """Download artwork from *url* into *show_path*, return the local path."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PodCodex/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read(5 * 1024 * 1024)  # cap at 5 MB
    except Exception as exc:
        logger.warning("Artwork download failed for {}: {}", url, exc)
        return None

    # Determine extension from Content-Type or URL
    ext = ".jpg"  # default
    for e, mime in _MIME.items():
        if mime in content_type:
            ext = e
            break
    else:
        # Try URL extension
        url_lower = url.lower().split("?")[0]
        for e in _IMG_EXTENSIONS:
            if url_lower.endswith(e):
                ext = e
                break

    # Remove any old cached artwork
    for old_ext in _IMG_EXTENSIONS:
        (show_path / f"{_ARTWORK_STEM}{old_ext}").unlink(missing_ok=True)

    dest = show_path / f"{_ARTWORK_STEM}{ext}"
    dest.write_bytes(data)

    # Write URL hash so we know when to re-download
    (show_path / ".artwork_url_hash").write_text(_url_hash(url), encoding="utf-8")

    return dest


@router.get("/artwork")
async def get_artwork(show_folder: str = Query(...)):
    """Serve cached artwork for a show, downloading it if needed."""
    path = require_show_folder(show_folder)
    meta = load_show_meta(path)
    artwork_url = (meta.artwork_url if meta else "") or ""

    if not artwork_url:
        raise HTTPException(404, "No artwork URL configured")

    cached = _find_cached_artwork(path)
    url_hash_file = path / ".artwork_url_hash"

    # Re-download if URL changed or no cache
    need_download = cached is None
    if cached and url_hash_file.exists():
        stored_hash = url_hash_file.read_text(encoding="utf-8").strip()
        if stored_hash != _url_hash(artwork_url):
            need_download = True

    if need_download:
        import asyncio

        cached = await asyncio.get_running_loop().run_in_executor(
            None, _download_artwork, artwork_url, path
        )

    if not cached:
        raise HTTPException(502, "Failed to download artwork")

    media_type = _MIME.get(cached.suffix.lower(), "image/jpeg")
    return FileResponse(
        cached,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.post("/from-rss", response_model=CreateFromRSSResponse)
async def create_from_rss(req: CreateFromRSSRequest) -> CreateFromRSSResponse:
    """Fetch an RSS feed and create a show folder for it."""
    save_base = Path(req.save_path).expanduser()
    if not save_base.is_dir():
        raise HTTPException(400, f"Save path does not exist: {req.save_path}")

    try:
        episodes, feed_art = await asyncio.to_thread(
            fetch_feed_with_artwork, req.rss_url
        )
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch feed: {exc}") from exc
    if not episodes:
        raise HTTPException(502, "Feed returned no episodes")

    # Determine folder name
    folder_name = req.folder_name.strip()
    if not folder_name:
        folder_name = re.sub(r"https?://", "", req.rss_url)
        folder_name = re.sub(r"[^a-zA-Z0-9]+", "_", folder_name).strip("_")[:40]

    show_path = save_base / folder_name
    show_path.mkdir(parents=True, exist_ok=True)

    artwork = req.artwork_url or feed_art

    # Save show metadata — use the display name from search, fall back to folder name
    show_name = req.name.strip() or folder_name
    save_show_meta(
        show_path,
        _ShowMeta(
            name=show_name,
            rss_url=req.rss_url,
            artwork_url=artwork,
            language=req.language,
        ),
    )

    # Cache the feed
    save_feed_cache(show_path, episodes)

    # Register in config
    cfg = _load()
    _register_folder(cfg, str(show_path))

    return CreateFromRSSResponse(
        folder=str(show_path),
        name=show_name,
        episode_count=len(episodes),
    )


@router.post("/from-youtube", response_model=CreateFromYouTubeResponse)
def create_from_youtube(
    req: CreateFromYouTubeRequest,
) -> CreateFromYouTubeResponse:
    """Fetch YouTube metadata and create a show folder.

    Sync def on purpose: the yt-dlp crawl can take minutes; FastAPI's
    threadpool keeps it off the event loop.
    """
    from podcodex.ingest.youtube import fetch_youtube

    save_base = Path(req.save_path).expanduser()
    if not save_base.is_dir():
        raise HTTPException(400, f"Save path does not exist: {req.save_path}")

    # One extraction yields both the episode list and the channel info
    # (name, artwork); a separate youtube_show_info call would re-crawl
    # the whole channel.
    try:
        episodes, info = fetch_youtube(req.youtube_url)
    except ImportError as exc:
        raise HTTPException(501, str(exc)) from None
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch videos: {exc}") from None

    if not episodes:
        raise HTTPException(502, "No videos found at this URL")

    # Determine folder name
    folder_name = req.folder_name.strip()
    if not folder_name:
        folder_name = re.sub(r"[^a-zA-Z0-9]+", "_", info.get("name", "youtube")).strip(
            "_"
        )[:40]

    show_path = save_base / folder_name
    show_path.mkdir(parents=True, exist_ok=True)

    # Save show metadata
    show_name = req.name.strip() or info.get("name", "") or folder_name
    artwork = req.artwork_url or info.get("artwork_url", "")
    save_show_meta(
        show_path,
        _ShowMeta(
            name=show_name,
            youtube_url=req.youtube_url,
            artwork_url=artwork,
            language=req.language,
        ),
    )

    # Cache the episode list (same format as RSS)
    save_feed_cache(show_path, episodes)

    # Register in config
    cfg = _load()
    _register_folder(cfg, str(show_path))

    return CreateFromYouTubeResponse(
        folder=str(show_path),
        name=show_name,
        episode_count=len(episodes),
    )


@router.post("/register")
def register_show(req: RegisterShowRequest) -> dict:
    """Register an existing folder as a known show."""
    p = Path(req.path).expanduser().resolve()
    if not p.is_dir():
        raise HTTPException(400, f"Not a directory: {req.path}")

    # Create show.toml if it doesn't exist yet
    if not load_show_meta(p):
        save_show_meta(p, _ShowMeta(name=p.name))

    cfg = _load()
    _register_folder(cfg, str(p))
    return {"status": "ok", "path": str(p)}


# ── Episode serialization ────────────────────


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
def get_show_meta(show_folder: str) -> ShowMeta:
    """Return metadata for a show folder."""
    path = require_show_folder(show_folder)
    meta = load_show_meta(path)
    last_feed_update: str | None = None
    try:
        from datetime import datetime, timezone

        mtime = (path / ".feed_cache.json").stat().st_mtime
        last_feed_update = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except FileNotFoundError:
        pass
    if meta is None:
        return ShowMeta(name=path.name, last_feed_update=last_feed_update)
    return ShowMeta(
        name=meta.name,
        rss_url=meta.rss_url,
        youtube_url=meta.youtube_url,
        language=meta.language,
        speakers=meta.speakers,
        artwork_url=meta.artwork_url,
        broadcast_number_pattern=meta.broadcast_number_pattern,
        pipeline=PipelineDefaultsSchema(
            model_size=meta.pipeline.model_size,
            diarize=meta.pipeline.diarize,
            num_speakers=meta.pipeline.num_speakers,
            llm_mode=meta.pipeline.llm_mode,
            llm_provider_profile=meta.pipeline.llm_provider_profile,
            llm_key_name=meta.pipeline.llm_key_name,
            llm_models_by_mode=dict(meta.pipeline.llm_models_by_mode or {}),
            llm_batch_minutes=meta.pipeline.llm_batch_minutes,
            context=meta.pipeline.context,
            target_lang=meta.pipeline.target_lang,
            rag_model=meta.pipeline.rag_model,
            rag_chunker=meta.pipeline.rag_chunker,
        ),
        last_feed_update=last_feed_update,
    )


@router.put("/{show_folder:path}/meta")
def update_show_meta(show_folder: str, meta: ShowMeta) -> dict:
    """Persist updated show metadata to show.toml."""
    path = require_show_folder(show_folder)
    p = meta.pipeline
    save_show_meta(
        path,
        _ShowMeta(
            name=meta.name,
            rss_url=meta.rss_url,
            youtube_url=meta.youtube_url,
            language=meta.language,
            speakers=meta.speakers,
            artwork_url=meta.artwork_url,
            broadcast_number_pattern=meta.broadcast_number_pattern,
            pipeline=_PipelineDefaults(
                model_size=p.model_size,
                diarize=p.diarize,
                num_speakers=p.num_speakers,
                llm_mode=p.llm_mode,
                llm_provider_profile=p.llm_provider_profile,
                llm_key_name=p.llm_key_name,
                llm_models_by_mode=dict(p.llm_models_by_mode or {}),
                llm_batch_minutes=p.llm_batch_minutes,
                context=p.context,
                target_lang=p.target_lang,
                rag_model=p.rag_model,
                rag_chunker=p.rag_chunker,
            ),
        ),
    )
    return {"status": "saved"}


def _latest_episode_title(path: Path) -> str | None:
    """Best-effort title of the show's newest episode, for pattern previews.

    Reads the feed cache (``feed_order`` 0 = newest, matching the episode
    list's fallback ordering). Returns None when there is no feed cache or no
    titled entry; the preview then shows nothing rather than testing the
    pattern against an arbitrary episode.
    """
    feed = load_feed_cache(path)
    if not feed:
        return None
    titled = [e for e in feed if e.title]
    if not titled:
        return None
    with_order = [e for e in titled if e.feed_order is not None]
    if with_order:
        return min(with_order, key=lambda e: e.feed_order).title
    return titled[0].title  # cache is written in feed order (newest first)


def _compute_broadcast_preview(path: Path, pattern: str) -> dict:
    """Sync body of ``broadcast_preview``; runs off the event loop."""
    title = _latest_episode_title(path)
    if not pattern.strip():
        return {"title": title, "number": None, "error": None}
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return {"title": title, "number": None, "error": f"Invalid pattern: {exc}"}
    if compiled.groups == 0:
        return {
            "title": title,
            "number": None,
            "error": "Pattern has no capture group: wrap the number in parentheses",
        }
    try:
        number = apply_broadcast_pattern(pattern, title or "")
    except re.error as exc:
        return {"title": title, "number": None, "error": f"Invalid pattern: {exc}"}
    return {"title": title, "number": number, "error": None}


@router.get(
    "/{show_folder:path}/broadcast-preview",
    response_model=BroadcastPreviewOut,
)
async def broadcast_preview(show_folder: str, pattern: str = Query("")) -> dict:
    """Test a broadcast-number pattern against the latest episode title.

    Uses the same extraction logic as indexing, so the previewed number is
    exactly what a reindex would store. Returns the tested title, the extracted
    number (or null), and an error when the regex is invalid or has no capture
    group. Runs in a worker thread: the pattern is user input typed live, and a
    backtracking-heavy regex must not stall the event loop.
    """
    import asyncio

    path = require_show_folder(show_folder)
    return await asyncio.get_running_loop().run_in_executor(
        None, _compute_broadcast_preview, path, pattern
    )


# ── Episode listing ──────────────────────────


@router.get("/{show_folder:path}/episodes", response_model=list[EpisodeOut])
def list_episodes(show_folder: str) -> list[dict]:
    """List locally scanned episodes for a show folder."""
    path = require_show_folder(show_folder)
    episodes = scan_folder(path)
    return [_episode_to_dict(ep) for ep in episodes]


# ── Unified episodes (local + RSS merged) ───


@router.get(
    "/{show_folder:path}/unified",
    response_model=list[UnifiedEpisodeOut],
)
def unified_episodes(
    show_folder: str,
    defaults: str | None = None,
) -> list[dict]:
    """Return a merged list of RSS + local episodes.

    Pipeline status comes from the per-show SQLite DB (pipeline.db).
    On first access the DB is populated from a filesystem scan.

    Args:
        defaults: Optional JSON string with app-level pipeline defaults
                  (model_size, diarize, llm_mode, llm_provider,
                  llm_models_by_mode, target_lang). Show-level overrides
                  take precedence.
    """
    path = require_show_folder(show_folder)
    ctx = _load_status_context(path, defaults)

    rss = load_feed_cache(path) or []

    result: list[dict] = []
    seen_stems: set[str] = set()
    seen_ids: set[str] = set()

    def _build_episode_out(
        *,
        ep_id: str,
        title: str,
        stem: str | None,
        pub_date: str | None,
        description: str,
        audio_url: str | None,
        duration: float,
        episode_number: int | None,
        audio_path: Path | None,
        output_dir: Path | None,
        artwork_url: str,
        st: dict,
        ep_files: list[str],
        removed: bool = False,
        feed_order: int | None = None,
    ) -> dict:
        return {
            "id": ep_id,
            "title": title,
            "pub_date": pub_date,
            "description": description,
            "audio_url": audio_url,
            "duration": duration,
            "episode_number": episode_number,
            "artwork_url": artwork_url,
            "removed": removed,
            "feed_order": feed_order,
            **_build_status_out(
                stem=stem,
                audio_path=audio_path,
                output_dir=output_dir,
                st=st,
                ep_files=ep_files,
                ctx=ctx,
            ),
        }

    # Pass the set of stems already on disk so episode_stem can match a
    # changed-title episode to its existing file without re-scandir-ing per
    # call. Covers root-audio stems and per-episode subdir stems.
    existing_stems = set(ctx.local_audio) | set(ctx.episode_files)

    # RSS episodes first (preserves feed order)
    for r in rss:
        stem = episode_stem(r, path, existing_stems=existing_stems)
        if r.guid in seen_ids:
            continue
        seen_ids.add(r.guid)
        st = ctx.status_map.get(stem, {}) if stem else {}
        audio_path = ctx.local_audio.get(stem)
        if stem:
            seen_stems.add(stem)
        result.append(
            _build_episode_out(
                ep_id=r.guid,
                title=r.title,
                stem=stem,
                pub_date=r.pub_date,
                description=r.description or "",
                audio_url=r.audio_url or None,
                duration=r.duration,
                episode_number=r.episode_number,
                audio_path=audio_path,
                output_dir=path / stem if stem else None,
                artwork_url=r.artwork_url or "",
                st=st,
                ep_files=ctx.episode_files.get(stem, []) if stem else [],
                removed=r.removed,
                feed_order=r.feed_order,
            )
        )

    # Local-only episodes (no RSS match)
    for stem, st in ctx.status_map.items():
        if stem in seen_stems:
            continue
        output_dir = path / stem
        meta = load_episode_meta(output_dir) if output_dir.is_dir() else None
        ep_id = meta.guid if meta else stem
        if ep_id in seen_ids:
            continue
        seen_ids.add(ep_id)
        audio_path = ctx.local_audio.get(stem)
        result.append(
            _build_episode_out(
                ep_id=ep_id,
                title=(meta.title if meta else None) or stem,
                stem=stem,
                pub_date=meta.pub_date if meta else None,
                description=(meta.description or "") if meta else "",
                audio_url=(meta.audio_url or None) if meta else None,
                duration=meta.duration if meta else 0,
                episode_number=meta.episode_number if meta else None,
                audio_path=audio_path,
                output_dir=output_dir,
                artwork_url=(meta.artwork_url or "") if meta else "",
                st=st,
                ep_files=ctx.episode_files.get(stem, []),
            )
        )

    return result


@router.get(
    "/{show_folder:path}/status",
    response_model=list[EpisodeStatusOut],
)
def episode_statuses(
    show_folder: str,
    defaults: str | None = None,
) -> list[dict]:
    """Return live pipeline status for every known episode, keyed by stem.

    The cheap counterpart to ``/unified``, meant for the 5s poll the UI runs
    while a download or batch is in flight. It reuses the exact same status
    builder, but skips everything that only feeds the *static* half of an
    episode: the feed cache parse (10-500KB of JSON per request), the
    per-feed-entry stem resolution, and the per-episode ``.episode_meta.json``
    reads. Feed-only episodes with no local footprint are omitted — they have
    no status to report, and the client already holds their static fields.

    Args:
        defaults: Same JSON string as ``/unified``; step statuses are relative
                  to the effective defaults, so it must match or the poll
                  would flip ``outdated`` markers back and forth.
    """
    path = require_show_folder(show_folder)
    ctx = _load_status_context(path, defaults)
    return [
        _build_status_out(
            stem=stem,
            audio_path=ctx.local_audio.get(stem),
            output_dir=path / stem,
            st=st,
            ep_files=ctx.episode_files.get(stem, []),
            ctx=ctx,
        )
        for stem, st in ctx.status_map.items()
    ]


class _StatusContext(NamedTuple):
    """Per-request state shared by every episode's status build."""

    status_map: dict[str, dict]
    seg_counts: dict[str, int]
    stems_with_speaker_map: Container[str]
    local_audio: dict[str, Path]
    episode_files: dict[str, list[str]]
    effective: dict


def _load_status_context(path: Path, defaults: str | None) -> _StatusContext:
    """Gather everything the status half of an episode payload needs.

    Shared by ``/unified`` and ``/status`` so the two can never disagree about
    a flag. Also runs the DB reconciliation passes (indexed / synthesized /
    verified pointers), which must happen on the polled endpoint too or a
    step finishing mid-batch would not surface until the next heavy fetch.
    """
    import json as _json

    # ── Resolve effective defaults (app → show override) ──
    try:
        app_defaults = _json.loads(defaults) if defaults else {}
    except _json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid JSON in 'defaults' parameter: {exc}")
    show_meta = load_show_meta(path)
    effective = _resolve_defaults(app_defaults, show_meta)

    # ── Pipeline status from DB (or one-time migration) ──
    # LanceDB is the source of truth for indexed status; query once and
    # share between the (possible) initial scan and the reconciliation
    # pass below.
    lance_indexed = lance_indexed_stems(path)

    db = get_pipeline_db(path)
    if db.episode_count() == 0:
        episodes = scan_folder(path, indexed_stems=lance_indexed)
        if episodes:
            db.populate_from_scan(episodes)

    status_map: dict[str, dict] = {row["stem"]: row for row in db.all_episodes()}

    indexed_updates: dict[str, bool] = {}
    for stem, row in status_map.items():
        truth = stem in lance_indexed
        if bool(row.get("indexed", False)) != truth:
            indexed_updates[stem] = truth
            row["indexed"] = truth
    if indexed_updates:
        db.mark_indexed_bulk(indexed_updates)

    local_audio = _scan_audio_files(path)
    episode_files = _scan_episode_files(path, local_audio)

    # Reconcile the per-step flags: an episode is transcribed / corrected /
    # synthesized when it has a registered version for that step *or* a file
    # in the step directory. Both directions matter. Without the promote, the
    # overview StageCard stays "not started" for any episode whose first sync
    # predates our first assemble; without the demote, a flag survives content
    # deleted out of band.
    #
    # The on-disk half is not optional: `populate_from_scan` above derives
    # these very flags from the step directories, and a DB bootstrapped that
    # way has no `versions` rows at all. Reconciling against rows alone would
    # undo the bootstrap in the same call, and `POST /resync` (which deletes
    # the DB file, versions table included) would report a whole library as
    # not started. Read from the already-cached file list, so this costs no
    # extra syscalls.
    for step, flag in STEP_FLAG.items():
        stems_with_versions = set(db.stems_with_step(step))
        ext = step_ext(step)
        for stem, row in status_map.items():
            desired = stem in stems_with_versions or _has_step_files(
                episode_files.get(stem, []), stem, step, ext
            )
            if row.get(flag, False) != desired:
                row[flag] = desired
                db.mark(stem, **{flag: desired})

    # Reconcile verified pointers: a pointer whose target version no longer
    # exists (out-of-band file deletion, manual DB edit) is stale and must
    # be cleared so the UI never highlights a missing version.
    verified_pointers = db.verified_pointers()
    if verified_pointers:
        ids_by_step: dict[str, dict[str, set[str]]] = {}
        for step_name in {p["step"] for p in verified_pointers.values()}:
            ids_by_step[step_name] = db.version_ids_by_stem(step_name)
        for stem, ptr in list(verified_pointers.items()):
            step_ids = ids_by_step.get(ptr["step"], {}).get(stem, set())
            if ptr["version_id"] not in step_ids:
                db.clear_verified(stem)
                verified_pointers.pop(stem, None)
                row = status_map.get(stem)
                if row:
                    row["verified"] = None

    return _StatusContext(
        status_map=status_map,
        seg_counts=db.latest_segment_counts("transcript"),
        stems_with_speaker_map=db.stems_with_step("speaker_map"),
        local_audio=local_audio,
        episode_files=episode_files,
        effective=effective,
    )


def _has_step_files(ep_files: list[str], stem: str, step: str, ext: str) -> bool:
    """True when the episode's file list holds a version file for *step*.

    `ep_files` entries are paths relative to the show folder, so a version
    file reads as ``<stem>/<step>/<id><ext>``. Matching only that one level
    keeps this in step with `ingest/folder._step_has_versions`, which globs
    ``<step>/*<ext>`` and feeds the very bootstrap this defends; a nested
    sub-step that ever emits the same extension would otherwise make the two
    disagree about the same episode.
    """
    prefix = f"{stem}/{step}/"
    return any(
        f.startswith(prefix) and f.endswith(ext) and "/" not in f[len(prefix) :]
        for f in ep_files
    )


def _build_status_out(
    *,
    stem: str | None,
    audio_path: Path | None,
    output_dir: Path | None,
    st: dict,
    ep_files: list[str],
    ctx: _StatusContext,
) -> dict:
    """Build the `EpisodeStatusOut` half of an episode payload."""
    prov = _normalize_provenance(st.get("provenance", {}))
    # Speaker labels resolved by user counts as editing the displayed transcript,
    # even though raw segment text is unchanged.
    if stem and stem in ctx.stems_with_speaker_map:
        tprov = prov.get("transcript")
        prov["transcript"] = {
            **(tprov if isinstance(tprov, dict) else {}),
            "manual_edit": True,
        }
    cleaned_translations = clean_translations(st.get("translations", []))
    out_dir_exists = bool(output_dir and output_dir.is_dir())
    return {
        "stem": stem,
        "audio_path": str(audio_path) if audio_path else None,
        "output_dir": str(output_dir) if out_dir_exists else None,
        "downloaded": audio_path is not None,
        "transcribed": st.get("transcribed", False),
        "corrected": st.get("corrected", False),
        "indexed": st.get("indexed", False),
        "synthesized": st.get("synthesized", False),
        "has_subtitles": any(f.endswith(".vtt") for f in ep_files),
        "translations": cleaned_translations,
        "segment_count": ctx.seg_counts.get(stem) if stem else None,
        "files": ep_files,
        "provenance": prov,
        "verified": st.get("verified"),
        "llm_failed_steps": rejected_steps(output_dir) if out_dir_exists else [],
        **_step_statuses(st, prov, ctx.effective, cleaned_translations),
    }


_PARAM_RENAMES = {"mode": "llm_mode"}


def _normalize_provenance(prov: dict) -> dict:
    """Rename legacy param keys (mode→llm_mode)."""
    out = {}
    for step_key, meta in prov.items():
        if not isinstance(meta, dict):
            out[step_key] = meta
            continue
        params = meta.get("params")
        if isinstance(params, dict):
            params = {_PARAM_RENAMES.get(k, k): v for k, v in params.items()}
            meta = {**meta, "params": params}
        out[step_key] = meta
    return out


def _resolve_defaults(app_defaults: dict, show_meta: _ShowMeta | None) -> dict:
    """Merge app-level defaults with show-level overrides.

    Show-level values override app defaults when explicitly set. Strings
    use `""` as the unset sentinel; `diarize` uses `None`.
    """
    effective = dict(app_defaults)
    # Merge per-mode model dicts: app first, show overrides per-mode entries.
    app_models = dict(effective.get("llm_models_by_mode") or {})
    effective.pop("llm_models_by_mode", None)
    show_models: dict[str, str] = {}
    if show_meta and show_meta.pipeline:
        p = show_meta.pipeline
        if p.model_size:
            effective["model_size"] = p.model_size
        if p.llm_mode:
            effective["llm_mode"] = p.llm_mode
        if p.llm_provider_profile:
            effective["llm_provider_profile"] = p.llm_provider_profile
        if p.llm_key_name:
            effective["llm_key_name"] = p.llm_key_name
        if p.target_lang:
            effective["target_lang"] = p.target_lang
        if p.diarize is not None:
            effective["diarize"] = p.diarize
        if p.llm_batch_minutes is not None and p.llm_batch_minutes > 0:
            effective["llm_batch_minutes"] = p.llm_batch_minutes
        show_models = {k: v for k, v in (p.llm_models_by_mode or {}).items() if v}
    merged_models = {**app_models, **show_models}
    mode = effective.get("llm_mode", "")
    resolved_model = merged_models.get(mode, "") if mode else ""
    if resolved_model:
        effective["llm_model"] = resolved_model
    return effective


def _transcribe_outdated(prov: dict, effective: dict) -> bool:
    """Check if a transcribe step's provenance is outdated relative to effective defaults."""
    params = prov.get("params", {})
    source = params.get("source", "whisper")
    # Imported/uploaded transcripts are not outdated — they weren't auto-generated
    if source not in ("whisper",):
        return False
    if not effective:
        return False
    if effective.get("model_size") and prov.get("model") != effective["model_size"]:
        return True
    if "diarize" in effective and params.get("diarize") != effective["diarize"]:
        return True
    return False


def _llm_outdated(prov: dict, effective: dict) -> bool:
    """Check if an LLM step's provenance is outdated relative to effective defaults."""
    params = prov.get("params", {})
    if effective.get("llm_mode") and params.get("llm_mode") != effective["llm_mode"]:
        return True
    if (
        effective.get("llm_provider_profile")
        and params.get("llm_provider_profile") != effective["llm_provider_profile"]
    ):
        return True
    if effective.get("llm_model") and prov.get("model") != effective["llm_model"]:
        return True
    if (
        effective.get("source_lang")
        and params.get("source_lang") != effective["source_lang"]
    ):
        return True
    return False


def _step_statuses(
    st: dict, provenance: dict, effective: dict, translations: list[str]
) -> dict:
    """Compute per-step status: 'none' | 'outdated' | 'done'.

    Compares the episode's provenance against the effective defaults.
    User-validated versions short-circuit to 'done': re-running would
    discard the edits, so 'outdated' is misleading.

    `translations` is the pre-cleaned languages list (see clean_translations);
    callers pass it through so the scrub runs once per episode, not twice.
    """

    verified = st.get("verified") or {}
    verified_step = verified.get("step") if isinstance(verified, dict) else None

    def _check_transcribe() -> str:
        if not st.get("transcribed", False):
            return "none"
        # Verified pointer is the user's explicit "I'm done with this step"
        # signal; it outranks model drift just like edited content does.
        if verified_step == "transcript":
            return "done"
        prov = provenance.get("transcript")
        if not prov:
            return "done"  # no provenance → legacy, assume done
        if is_edited(prov):
            return "done"
        return "outdated" if _transcribe_outdated(prov, effective) else "done"

    def _check_correct() -> str:
        if not st.get("corrected", False):
            return "none"
        if verified_step == "corrected":
            return "done"
        prov = provenance.get("corrected")
        if not prov or not effective:
            return "done"
        if is_edited(prov):
            return "done"
        return "outdated" if _llm_outdated(prov, effective) else "done"

    def _check_translate() -> str:
        if not translations:
            return "none"
        target = effective.get("target_lang", "").strip().lower()
        if target and target not in translations:
            return "none"
        lang_key = target or (translations[0] if translations else "")
        prov = provenance.get(lang_key)
        if not prov or not effective:
            return "done"
        if is_edited(prov):
            return "done"
        return "outdated" if _llm_outdated(prov, effective) else "done"

    return {
        "transcribe_status": _check_transcribe(),
        "correct_status": _check_correct(),
        "translate_status": _check_translate(),
    }


# Roster is expensive: it reads every episode's canonical transcript. Cache it
# keyed on the resolved canonical refs, the known-speaker set, and the episode
# meta mtimes (titles are baked into the response). All are recomputed from
# two bulk DB queries plus per-stem stats, so a cache hit skips the N seglist
# reads. Any version save/delete, verified-pointer change, show.toml speaker
# edit, or episode-meta (title) refresh shifts the signature.
_ROSTER_CACHE: dict[str, tuple[object, SpeakerRosterResponse]] = {}


def _compute_speaker_roster(path: Path) -> SpeakerRosterResponse:
    from concurrent.futures import ThreadPoolExecutor

    from podcodex.core._utils import speaker_airtime
    from podcodex.core.versions import load_version, resolve_canonical_refs
    from podcodex.ingest.rss import EPISODE_META_FILE

    db = get_pipeline_db(path)
    if db.episode_count() == 0:
        eps = scan_folder(path)
        if eps:
            db.populate_from_scan(eps)

    meta = load_show_meta(path)
    known = set(meta.speakers) if meta else set()

    totals: dict[str, dict] = {}
    per_episode: dict[str, list[SpeakerEpisodeEntry]] = {}
    episodes_with_transcripts = 0

    # Resolve every episode's canonical version ref (verified pointer wins,
    # same ladder as the per-episode speaker endpoint) via the bulk resolver:
    # two DB queries total, single-threaded (pipeline_db isn't safe to fan
    # out). Only the seglist file loads below are parallelized.
    stems = [ep["stem"] for ep in db.all_episodes()]
    episodes_scanned = len(stems)
    refs = resolve_canonical_refs(path, stems)

    def _meta_mtime(stem: str) -> float:
        try:
            return (path / stem / EPISODE_META_FILE).stat().st_mtime
        except OSError:
            return 0.0

    signature = (
        frozenset(refs.items()),
        frozenset(known),
        frozenset((s, _meta_mtime(s)) for s in stems),
    )
    cached = _ROSTER_CACHE.get(str(path))
    if cached is not None and cached[0] == signature:
        return cached[1]

    def _load_segments(stem: str) -> tuple[str, list[dict] | None]:
        ref = refs.get(stem)
        if not ref:
            return stem, None
        step, vid = ref
        try:
            return stem, load_version(path / stem / stem, step, vid)
        except FileNotFoundError:
            return stem, None

    # JSON reads parallelize well: they're disk-bound, not CPU-bound.
    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(_load_segments, stems))

    # A ref that resolved but failed to load (deleted out-of-band, or not yet
    # visible on a shared mount) must not be frozen into the cache: skip the
    # cache write below so the miss self-heals on the next request.
    load_failed = any(
        segs is None and refs.get(stem) is not None for stem, segs in loaded
    )

    for stem, segments in loaded:
        if not segments:
            continue
        episodes_with_transcripts += 1

        ep_meta = load_episode_meta(path / stem)
        ep_title = ep_meta.title if ep_meta and ep_meta.title else stem

        for spk, air in speaker_airtime(segments).items():
            secs = air["total_seconds"]
            n = air["segment_count"]
            row = totals.setdefault(
                spk,
                {"episode_count": 0, "segment_count": 0, "total_seconds": 0.0},
            )
            row["episode_count"] += 1
            row["segment_count"] += n
            row["total_seconds"] += secs
            per_episode.setdefault(spk, []).append(
                SpeakerEpisodeEntry(
                    stem=stem,
                    title=ep_title,
                    segment_count=n,
                    total_seconds=secs,
                )
            )

    for spk in known:
        totals.setdefault(
            spk,
            {"episode_count": 0, "segment_count": 0, "total_seconds": 0.0},
        )

    entries = [
        SpeakerRosterEntry(
            name=name,
            is_known=name in known,
            episode_count=row["episode_count"],
            segment_count=row["segment_count"],
            total_seconds=row["total_seconds"],
            episodes=sorted(
                per_episode.get(name, []),
                key=lambda e: e.total_seconds,
                reverse=True,
            ),
        )
        for name, row in totals.items()
    ]
    entries.sort(key=lambda s: (s.total_seconds, s.segment_count), reverse=True)

    response = SpeakerRosterResponse(
        speakers=entries,
        episodes_scanned=episodes_scanned,
        episodes_with_transcripts=episodes_with_transcripts,
    )
    if not load_failed:
        _ROSTER_CACHE[str(path)] = (signature, response)
    return response


@router.get(
    "/{show_folder:path}/speakers/roster",
    response_model=SpeakerRosterResponse,
)
async def speakers_roster(show_folder: str) -> SpeakerRosterResponse:
    """Aggregate speaker stats across every transcribed episode in the show.

    Each episode contributes its canonical transcript: the verified version
    when set, else the best ``corrected`` (hand-edited outranks newer model
    output), else the newest ``transcript``. There is no fallback when the
    canonical file is missing; the episode is skipped so the gap is visible.
    Placeholder labels from ``UNKNOWN_SPEAKERS`` and the ``[BREAK]`` sentinel
    are filtered out. Speakers listed in ``show.toml`` that never appear are
    still returned with zero counts so the UI can surface
    configured-but-unseen names.
    """
    import asyncio

    path = require_show_folder(show_folder)
    return await asyncio.get_running_loop().run_in_executor(
        None, _compute_speaker_roster, path
    )


def _compute_episode_speakers(path: Path, stem: str) -> EpisodeSpeakersResponse:
    """Speakers + per-speaker airtime for one episode's canonical transcript."""
    from podcodex.core._utils import speaker_airtime
    from podcodex.core.versions import load_canonical_segments

    base = path / stem / stem
    segments = load_canonical_segments(base)
    if not segments:
        return EpisodeSpeakersResponse(
            speakers=[], episode_seconds=0.0, has_transcript=False
        )

    ep_meta = load_episode_meta(path / stem)
    audio_seconds = float(ep_meta.duration) if ep_meta else 0.0
    last_end = max((float(s.get("end", 0.0)) for s in segments), default=0.0)
    # Denominator is the full episode length so unattributed time (music, gaps,
    # silence) is simply not counted, so the percentages can sum to under 100%.
    denom = max(audio_seconds, last_end)

    air = speaker_airtime(segments)
    entries = [
        EpisodeSpeakerEntry(
            name=spk,
            total_seconds=v["total_seconds"],
            pct=(v["total_seconds"] / denom * 100.0) if denom > 0 else 0.0,
        )
        for spk, v in air.items()
    ]
    entries.sort(key=lambda e: e.total_seconds, reverse=True)
    return EpisodeSpeakersResponse(
        speakers=entries, episode_seconds=denom, has_transcript=True
    )


@router.get(
    "/{show_folder:path}/episode/{stem}/speakers",
    response_model=EpisodeSpeakersResponse,
)
async def episode_speakers(show_folder: str, stem: str) -> EpisodeSpeakersResponse:
    """Speakers of one episode's canonical transcript, with airtime shares.

    The canonical transcript is the verified version if set, else the newest
    ``corrected``, else the newest ``transcript``. Each speaker's ``pct`` is
    its share of the episode duration; music, gaps, and unlabeled time are not
    attributed, so the shares may sum to less than 100%.
    """
    import asyncio

    path = require_show_folder(show_folder)
    return await asyncio.get_running_loop().run_in_executor(
        None, _compute_episode_speakers, path, stem
    )


@router.post("/{show_folder:path}/resync")
def resync_pipeline_db(show_folder: str) -> dict:
    """Force-rebuild pipeline.db from filesystem scan."""
    path = require_show_folder(show_folder)
    from podcodex.core.pipeline_db import reset_pipeline_db

    reset_pipeline_db(path)
    db = get_pipeline_db(path)
    episodes = scan_folder(path)
    if episodes:
        db.populate_from_scan(episodes)
    return {"status": "resynced", "episode_count": len(episodes)}


@router.get("/best-source-segments")
def best_source_segments(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Return the verified-first, corrected-next, transcript-last source segments.

    Single facility consumed by both panels (translate reference pane) and
    the floating audio player so they cannot disagree on which version is
    the canonical playback source.
    """
    from podcodex.api.routes._helpers import load_best_source, require_audio_or_output

    require_audio_or_output(audio_path, output_dir)
    try:
        return load_best_source(audio_path=audio_path, output_dir=output_dir)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@router.get("/versions")
def list_all_versions(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """List versions across all pipeline steps for an episode, newest first.

    Backfills ``params.file_size_bytes`` for every version (persisted on
    first call) so the "All other files" UI can show real sizes without
    forcing a re-run of each step.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.core.versions import backfill_version_sizes, list_all_versions

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    versions = list_all_versions(p.base)
    backfill_version_sizes(p.base, versions)
    return versions


class VerifiedRequest(BaseModel):
    """Payload for setting / clearing the verified pointer on an episode."""

    step: str | None = None
    version_id: str | None = None


@router.put("/verified")
def set_verified_version(
    req: VerifiedRequest,
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> dict:
    """Set or clear the episode's verified-version pointer.

    Body ``{step, version_id}`` marks that version as verified (canonical
    source). Body ``{step: null, version_id: null}`` clears the pointer.
    Singleton: replaces any previous pointer for the episode.
    """
    from podcodex.api.routes._helpers import require_audio_or_output
    from podcodex.core._utils import AudioPaths
    from podcodex.core.versions import VERIFIABLE_STEPS, version_path

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    # PipelineDB lives at the show level. base = <show>/<stem>/<stem>, so
    # base.parent is the episode dir and base.parent.parent is the show dir.
    show_dir = p.base.parent.parent
    db = get_pipeline_db(show_dir)

    if req.step is None and req.version_id is None:
        db.clear_verified(p.base.name)
        return {"status": "cleared", "verified": None}
    if not req.step or not req.version_id:
        raise HTTPException(
            400, "step and version_id must both be provided, or both null to clear"
        )
    if req.step not in VERIFIABLE_STEPS:
        raise HTTPException(
            400,
            f"step must be one of {sorted(VERIFIABLE_STEPS)}, got {req.step!r}",
        )
    if not version_path(p.base, req.step, req.version_id).exists():
        raise HTTPException(
            404,
            f"Version {req.version_id} not found for step {req.step!r}",
        )
    # Don't materialize a phantom episode row: aggregate_status would then
    # count a stem with no real pipeline progress as "verified".
    if db.get_episode(p.base.name) is None:
        raise HTTPException(
            404,
            f"Episode {p.base.name!r} not registered in pipeline_db",
        )
    db.set_verified(p.base.name, req.step, req.version_id)
    return {
        "status": "set",
        "verified": {"step": req.step, "version_id": req.version_id},
    }


@router.delete("/versions/{version_id}")
def delete_any_version(
    version_id: str,
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> dict:
    """Delete a version regardless of step. Step is resolved from the DB.

    Lets the episode overview "All other files" section delete intermediates
    (segments, diarization, diarized_segments, speaker_map) without a
    per-step DELETE route.
    """
    from podcodex.api.routes._helpers import require_audio_or_output
    from podcodex.core._utils import AudioPaths
    from podcodex.core.versions import delete_version_by_id

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    if not delete_version_by_id(p.base, version_id):
        raise HTTPException(404, f"Version {version_id} not found")
    return {"status": "deleted", "version_id": version_id}


def _scan_audio_files(show_folder: Path) -> dict[str, Path]:
    """Quick scan of audio files at show root — single os.scandir call."""
    import os
    from podcodex.ingest.folder import AUDIO_EXTENSIONS

    audio: dict[str, Path] = {}
    try:
        with os.scandir(show_folder) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    name = entry.name
                    dot = name.rfind(".")
                    if dot > 0 and name[dot:].lower() in AUDIO_EXTENSIONS:
                        audio[name[:dot]] = show_folder / name
    except OSError:
        pass
    return audio


_INTERESTING_EXTS = AUDIO_EXTENSIONS | {
    ".vtt",
    ".srt",  # subtitles
    ".json",
    ".parquet",  # transcripts / pipeline outputs
}
_SKIP_PREFIXES = (".", "__")
_SKIP_NAMES = {"manifest.json"}


def _walk_episode_dir(
    root: Path, rel_prefix: str
) -> tuple[list[str], list[tuple[str, int]] | None]:
    """Recursively collect interesting files under an episode dir.

    ``rel_prefix`` is the path (relative to the show folder) to prepend to
    each file name, so we skip allocating a Path per entry just to call
    ``relative_to``.

    Returns the file list plus every directory visited paired with its mtime,
    which is what `_scan_episode_files` caches on. The directory list is
    ``None`` when any part of the walk hit an ``OSError``: the file list is
    then incomplete, and caching it would pin a truncated result against
    mtimes that will not change. Since this list also drives the status-flag
    reconcile, a transient EACCES could otherwise demote a step and keep it
    demoted.
    """
    import os

    collected: list[str] = []
    try:
        stamp = os.stat(root).st_mtime_ns
    except OSError:
        return [], None
    visited: list[tuple[str, int]] | None = [(str(root), stamp)]
    try:
        with os.scandir(root) as it:
            for f in it:
                name = f.name
                if name.startswith(_SKIP_PREFIXES):
                    continue
                if f.is_dir(follow_symlinks=False):
                    sub_files, sub_dirs = _walk_episode_dir(
                        Path(f.path), f"{rel_prefix}/{name}"
                    )
                    collected.extend(sub_files)
                    if sub_dirs is None:
                        visited = None
                    elif visited is not None:
                        visited.extend(sub_dirs)
                    continue
                if not f.is_file(follow_symlinks=False) or name in _SKIP_NAMES:
                    continue
                dot = name.rfind(".")
                if dot <= 0 or name[dot:].lower() not in _INTERESTING_EXTS:
                    continue
                collected.append(f"{rel_prefix}/{name}")
    except OSError:
        return collected, None
    return collected, visited


# show folder → stem → (visited dirs with their mtimes, file list). Walking a
# show's episode dirs is the most expensive part of building the episode list
# (~20ms for 269 episodes) and it runs on every request, including the 5s
# status poll. Re-stat'ing the recorded directories instead costs ~0.7ms.
_EPISODE_FILES_CACHE: dict[str, dict[str, tuple[list[tuple[str, int]], list[str]]]] = {}

# A directory whose mtime is younger than this is treated as a cache miss.
# Same reasoning (and same value) as the mtime caches in ingest/rss.py; see
# core/_utils.MTIME_SETTLE_SECONDS.
_MTIME_SETTLE_NS = int(MTIME_SETTLE_SECONDS * 1_000_000_000)


def _dirs_unchanged(visited: list[tuple[str, int]]) -> bool:
    """True when every recorded directory still has its recorded mtime."""
    import os

    for path, stamp in visited:
        try:
            if os.stat(path).st_mtime_ns != stamp:
                return False
        except OSError:
            return False
    return True


def _settled(visited: list[tuple[str, int]], now_ns: int) -> bool:
    """True when every recorded mtime was already old when we recorded it.

    The settle window has to gate *storing* an entry, not trusting one: a
    directory written twice inside one coarse timestamp bucket (FAT32 rounds
    to 2s) keeps the same mtime, so an entry recorded between the two writes
    matches forever and hides the second. Refusing to cache until the mtime
    has stopped moving means anything we do cache cannot have a same-bucket
    write after it.
    """
    return all(now_ns - stamp >= _MTIME_SETTLE_NS for _, stamp in visited)


def _scan_episode_files(
    show_folder: Path, local_audio: dict[str, Path]
) -> dict[str, list[str]]:
    """Scan episode subdirectories for user-facing files.

    Returns a mapping of stem → list of filenames relative to show folder.
    Walks version subdirectories (``transcript/``, ``corrected/``,
    ``speaker_map/``, language folders, etc.) so the Pipeline file list
    surfaces version artifacts alongside legacy flat files.

    Per-episode results are cached against the mtimes of every directory the
    walk touched, so an added or removed file anywhere in the tree is caught:
    adding a file bumps its directory's mtime, and adding a directory bumps
    its parent's. Content edits don't bump anything, which is fine because
    only names are reported.
    """
    import os
    import time

    cached = _EPISODE_FILES_CACHE.get(str(show_folder), {})
    fresh: dict[str, tuple[list[tuple[str, int]], list[str]]] = {}
    now_ns = time.time_ns()

    result: dict[str, list[str]] = {}
    try:
        with os.scandir(show_folder) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                stem = entry.name
                if stem.startswith("."):
                    continue
                hit = cached.get(stem)
                if hit is not None and _dirs_unchanged(hit[0]):
                    fresh[stem] = hit
                    files = hit[1]
                else:
                    files, visited = _walk_episode_dir(Path(entry.path), stem)
                    files.sort()
                    # Only cache a complete walk whose mtimes have settled;
                    # anything else is re-walked next call (see _settled and
                    # _walk_episode_dir's None contract).
                    if visited and _settled(visited, now_ns):
                        fresh[stem] = (visited, files)
                # Hand out a copy: the root-audio merge below prepends to these
                # lists, which would otherwise grow the cached entry per call.
                if files:
                    result[stem] = list(files)
    except OSError:
        pass
    # Replacing the show's map (rather than updating it) drops entries for
    # episode directories that no longer exist.
    _EPISODE_FILES_CACHE[str(show_folder)] = fresh

    # Prepend root audio (already discovered by _scan_audio_files).
    for stem, audio_path in local_audio.items():
        result.setdefault(stem, []).insert(0, audio_path.name)

    return result


# ── Move / rename show folder ──────────────


class MoveShowRequest(BaseModel):
    new_path: str
    move_files: bool = True


@router.post("/{show_folder:path}/move")
def move_show(show_folder: str, req: MoveShowRequest) -> dict:
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

    # Release any cached file handles BEFORE the move. On Windows, SQLite (WAL)
    # holds file locks that prevent rename/unlink while open.
    close_pipeline_db(old_path)
    invalidate_scan_cache(old_path)

    leftover_warning: str | None = None

    if req.move_files:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(old_path), str(new_path))
        except OSError as exc:
            # Fallback: copy then best-effort cleanup. Handles cross-volume
            # moves and Windows file locks that survive close().
            logger.warning(
                "shutil.move failed ({}); falling back to copytree + rmtree",
                exc,
            )
            shutil.copytree(str(old_path), str(new_path), dirs_exist_ok=True)
            try:
                shutil.rmtree(str(old_path))
            except OSError as rm_exc:
                leftover_warning = (
                    f"Copied to {new_path} but could not remove {old_path}: {rm_exc}. "
                    "Delete it manually once no process is using it."
                )
                logger.warning(leftover_warning)
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
    old_resolved = str(old_path.resolve())

    def _replace(cfg: AppConfig) -> None:
        cfg.show_folders = [
            str(new_path) if str(Path(p).resolve()) == old_resolved else p
            for p in cfg.show_folders
        ]

    mutate_config(_replace)

    invalidate_scan_cache(new_path)

    result: dict = {"status": "moved", "new_path": str(new_path)}
    if leftover_warning:
        result["warning"] = leftover_warning
    return result


class DeleteShowRequest(BaseModel):
    delete_files: bool = False


@router.post("/{show_folder:path}/delete")
def delete_show(show_folder: str, req: DeleteShowRequest) -> dict:
    """Remove a show from the app. Optionally delete the local folder."""
    path = require_show_folder(show_folder)

    # Check no tasks are running on this show
    from podcodex.api.tasks import task_manager

    active = task_manager.get_active(show_folder)
    if active:
        raise HTTPException(
            409,
            f"Task {active.task_id} is running on this show — wait for it to finish",
        )

    # Close DB handles and invalidate caches
    close_pipeline_db(path)
    invalidate_scan_cache(path)
    _ROSTER_CACHE.pop(str(path), None)

    # Remove from config.json
    resolved = str(path.resolve())

    def _remove(cfg: AppConfig) -> None:
        cfg.show_folders = [
            p for p in cfg.show_folders if str(Path(p).resolve()) != resolved
        ]

    mutate_config(_remove)

    # Optionally delete the folder on disk
    deleted_files = False
    if req.delete_files and path.exists():
        shutil.rmtree(path)
        deleted_files = True
        logger.info("Deleted show folder: {}", path)
    else:
        logger.info("Unregistered show (files kept): {}", path)

    return {"status": "deleted", "files_deleted": deleted_files}
