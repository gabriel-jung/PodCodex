"""Show and episode management routes."""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import hashlib
import urllib.request

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from podcodex.api.tasks import TaskInfo

from podcodex.api.routes._helpers import (
    apply_broadcast_pattern,
    bad_path_component,
    keyed_lock,
    get_index_store,
    list_show_stems,
    require_registered_show,
    require_show_folder,
)
from podcodex.bundle.conflicts import rename_suffix
from podcodex.core._utils import (
    atomic_write,
    episode_base,
    virtual_audio_path,
)
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
    RegisterShowRequest,
    ShowMeta,
    SpeakerEpisodeEntry,
    SpeakerRosterEntry,
    SpeakerRosterResponse,
    UnifiedEpisodeOut,
)
from podcodex.core.constants import AUDIO_EXTENSIONS, LOCAL_ARTWORK_MARKER
from podcodex.core.episode_status import (
    build_status_out,
    load_status_context,
    reconcile_show_status,
)
from podcodex.core.pipeline_db import aggregate_rows, close_pipeline_db, get_pipeline_db
from podcodex.core.source import show_audio_files
from podcodex.core.versions import (
    seed_show_db,
)
from podcodex.ingest.folder import (
    EpisodeInfo,
    invalidate_scan_cache,
    scan_folder,
)
from podcodex.ingest.rss import (
    EPISODE_META_FILE,
    episode_stem,
    feed_cache_episode_count,
    fetch_feed_with_artwork,
    load_episode_meta,
    load_feed_cache,
    save_feed_cache,
)
from podcodex.ingest.show import PipelineDefaults as _PipelineDefaults
from podcodex.ingest.show import ShowMeta as _ShowMeta
from podcodex.ingest.show import is_feed_backed, load_show_meta, save_show_meta

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
    # Read-only: whether standalone audio can be imported here. Mirrors the
    # server's own import gate (``ingest.show.is_feed_backed``) so the target
    # picker can't offer a destination the endpoint rejects.
    accepts_imports: bool = False
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

        # Episodes with audio, by the scanner's own rule (core.source): the
        # card and the show page count the same thing.
        audio_count = len(show_audio_files(child))

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
                # The same file reconcile the show page runs, so the card's
                # counts agree with it; its corrected rows are what gets
                # counted. The index is not opened per show (stored indexed
                # flags are kept).
                rows = list(
                    reconcile_show_status(child, check_index=False).status_map.values()
                )
            except Exception as exc:
                # A failed reconcile keeps the stored counts.
                logger.warning("status reconcile failed for {}: {}", child, exc)
                rows = None
            try:
                if rows is None:
                    rows = get_pipeline_db(child).all_episodes()
                agg = aggregate_rows(rows)
                pipeline_total = agg["total"]
                transcribed = agg["transcribed"]
                transcribed_edited = agg["transcribed_edited"]
                corrected = agg["corrected"]
                corrected_edited = agg["corrected_edited"]
                translated = agg["translated"]
                translated_edited = agg["translated_edited"]
                synthesized = agg["synthesized"]
                indexed = agg["indexed"]
                verified = sum(1 for r in rows if r.get("verified"))
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
                accepts_imports=not is_feed_backed(child, meta),
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
    # Destination show folder. The app always sends one (target picker);
    # None is kept for direct API callers and falls back to the managed
    # "Files" bucket, created on first use.
    folder: str | None = None


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


class CreateLocalShowRequest(BaseModel):
    name: str


class CreateLocalShowResponse(BaseModel):
    folder: str
    name: str


@router.post("/create-local", response_model=CreateLocalShowResponse)
def create_local_show(req: CreateLocalShowRequest) -> CreateLocalShowResponse:
    """Create + register an empty local show under the default save path.

    Backs the import-target picker's "New show" option. A directory already
    at the path is a 409 (the picker asks for a different name) rather than
    a silent adopt: the folder might belong to an unrelated app.
    """
    name = req.name.strip()
    cfg = _load()
    base = Path(cfg.default_save_path or "~").expanduser()
    folder = _child_of_save_path(base, name)
    _refuse_taken_label(name, folder)
    try:
        folder.mkdir(parents=True)
    except FileExistsError:
        # Same 409 shape as the import collision, so the picker can offer a
        # free name rather than making the user guess which ones are taken.
        taken = {p.name for p in base.iterdir()} if base.is_dir() else set()
        raise HTTPException(
            409, detail={"suggested": rename_suffix(name, taken, suffix="")}
        )
    save_show_meta(folder, _ShowMeta(name=name))
    _register_folder(cfg, str(folder))
    return CreateLocalShowResponse(folder=str(folder), name=name)


@router.post("/files/import", response_model=FilesImportResponse)
async def import_local_file(req: FilesImportRequest) -> FilesImportResponse:
    """Copy a standalone audio file into a local show (default: Files bucket)."""
    src = Path(req.file_path).expanduser()
    if not src.is_file():
        raise HTTPException(404, f"File not found: {req.file_path}")
    ext = src.suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(400, f"Not an audio file: {src.name}")

    stem = (req.name or src.stem).strip()
    if bad_path_component(stem):
        raise HTTPException(400, f"Invalid name: {stem!r}")

    if req.folder is not None:
        # Registered-show gate: the import writes into the folder, so confine
        # it to tracked shows. Feed-backed shows are owned by their feed; a
        # root audio file there would collide with the downloader's naming.
        bucket = require_registered_show(req.folder)
        if is_feed_backed(bucket):
            raise HTTPException(400, "Cannot import files into a feed-backed show")
    else:
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
_ARTWORK_HASH_FILE = ".artwork_url_hash"
_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


_ARTWORK_MAX_BYTES = 5 * 1024 * 1024  # shared cap: URL download and upload


# Magic numbers of the formats the cover cache stores. A body that is none
# of them (a captive portal's HTML, a CDN error page answered as 200) is not
# a cover, whatever its URL says.
_IMG_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
)


def _sniff_image_ext(data: bytes) -> str | None:
    """Extension for an image body, from its magic bytes, else None."""
    for magic, ext in _IMG_MAGIC:
        if data.startswith(magic):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


def _url_hash(url: str) -> str:
    """Short hash of a URL — used to detect when the source URL changes."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _clear_cached_artwork(show_path: Path, *, keep_ext: str | None = None) -> None:
    """Forget the cached cover: every ``artwork.*`` variant and its URL stamp.

    The variants go so a new cover can't coexist with a stale one under a
    different extension. The stamp goes with them because it is meaningless
    without the image it describes, and a stale one suppresses the next
    re-download. *keep_ext* keeps the cover just written (and the stamp,
    which the writer sets next).
    """
    for old_ext in _IMG_EXTENSIONS:
        if old_ext != keep_ext:
            (show_path / f"{_ARTWORK_STEM}{old_ext}").unlink(missing_ok=True)
    if keep_ext is None:
        (show_path / _ARTWORK_HASH_FILE).unlink(missing_ok=True)


def _fresh_cached_artwork(show_path: Path, url: str) -> Path | None:
    """The cached cover when it is the one *url* names, else None.

    A cached file with no stamp is not fresh: that state means a local upload
    was replaced by a URL (upload unlinks the stamp), and the uploaded image
    must not be served under the new URL.
    """
    cached = _find_cached_artwork(show_path)
    stamp = show_path / _ARTWORK_HASH_FILE
    if cached is None or not stamp.exists():
        return None
    if stamp.read_text(encoding="utf-8").strip() != _url_hash(url):
        return None
    return cached


def _find_cached_artwork(show_path: Path) -> Path | None:
    """Return the cached artwork file if it exists."""
    for ext in _IMG_EXTENSIONS:
        p = show_path / f"{_ARTWORK_STEM}{ext}"
        if p.exists():
            return p
    return None


def _artwork_lock(show_path: Path):
    """The lock every writer of a show's cover files holds.

    One download per show at a time: the grid, the sidebar and the page ask
    for the same cover at once, and each miss would otherwise rewrite the
    file another request is streaming.
    """
    return keyed_lock("artwork", show_path)


def _download_artwork(url: str, show_path: Path) -> Path | None:
    """Download artwork from *url* into *show_path*, return the local path."""
    with _artwork_lock(show_path):
        # The cover may have changed while this request waited: an upload
        # (artwork_url becomes "local") must not be overwritten, and its
        # file not unlinked, by a download of the old URL.
        meta = load_show_meta(show_path)
        if meta is not None and meta.artwork_url != url:
            return None
        # A concurrent request may have fetched it while this one waited.
        cached = _fresh_cached_artwork(show_path, url)
        if cached is not None:
            return cached
        return _download_artwork_locked(url, show_path)


def _download_artwork_locked(url: str, show_path: Path) -> Path | None:
    from podcodex.ingest.rss import _require_http_scheme

    try:
        # A feed controls this URL. Without the scheme guard the feed and
        # audio fetches already carry, ``file://`` would copy a local file
        # into artwork.jpg, which GET /api/shows/artwork then serves.
        _require_http_scheme(url, "Artwork URL")
        req = urllib.request.Request(url, headers={"User-Agent": "PodCodex/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            # Read one byte past the cap: an over-limit body is a failed
            # download, not a file to truncate. Writing the first 5 MB of a
            # large cover stored a corrupt image and stamped the URL hash, so
            # it was served forever without ever being re-fetched.
            data = resp.read(_ARTWORK_MAX_BYTES + 1)
    except Exception as exc:
        logger.warning("Artwork download failed for {}: {}", url, exc)
        return None

    if len(data) > _ARTWORK_MAX_BYTES:
        logger.warning(
            "Artwork too large for {} (over {} bytes), skipping",
            url,
            _ARTWORK_MAX_BYTES,
        )
        return None

    # The bytes decide, not the Content-Type or the URL: a 200 HTML page
    # stamped as the cover would be served until the URL changed. No hash
    # is written on rejection, so the next request tries again.
    ext = _sniff_image_ext(data)
    if ext is None:
        logger.warning("Artwork at {} is not an image, skipping", url)
        return None

    # New file in place first, then the other-extension leftovers, so a
    # concurrent GET never finds no cover at all.
    dest = show_path / f"{_ARTWORK_STEM}{ext}"
    atomic_write(dest, lambda p: p.write_bytes(data))
    _clear_cached_artwork(show_path, keep_ext=ext)

    # Write URL hash so we know when to re-download
    atomic_write(
        show_path / _ARTWORK_HASH_FILE,
        lambda p: p.write_text(_url_hash(url), encoding="utf-8"),
    )
    return dest


@router.post("/artwork")
async def upload_artwork(
    show_folder: str = Query(...), file: UploadFile = File(...)
) -> dict:
    """Store an uploaded image as the show's cover.

    Writes ``artwork.{ext}`` into the show folder (replacing any previous
    cover) and marks ``artwork_url = "local"`` in show.toml so readers know
    the cover is file-backed and must never be re-downloaded over.
    """
    # Registered-show gate: this writes into the folder.
    path = require_registered_show(show_folder)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in _IMG_EXTENSIONS:
        raise HTTPException(400, f"Not an image file: {file.filename}")

    data = await file.read(_ARTWORK_MAX_BYTES + 1)
    if len(data) > _ARTWORK_MAX_BYTES:
        raise HTTPException(413, "Image too large (max 5 MB)")

    def _store() -> None:
        # Under the cover lock, off the event loop (a download may hold it
        # for its whole fetch).
        with _artwork_lock(path):
            _clear_cached_artwork(path)
            atomic_write(path / f"{_ARTWORK_STEM}{ext}", lambda p: p.write_bytes(data))
            meta = load_show_meta(path) or _ShowMeta(name=path.name)
            meta.artwork_url = LOCAL_ARTWORK_MARKER
            save_show_meta(path, meta)

    await asyncio.to_thread(_store)
    return {"status": "ok"}


@router.delete("/artwork")
def delete_artwork(show_folder: str = Query(...)) -> dict:
    """Remove the show's cover.

    Deliberately reverts to the *feed's* artwork rather than pinning "no
    cover": clearing ``artwork_url`` is what the upgrade branches in
    ``rss.py`` / ``youtube.py`` read as "missing", so the next feed refresh
    downloads the feed's own art again. On a local show there is no feed, so
    ``GET /artwork`` 404s and the UI falls back to ``default-cover.png``.

    No re-fetch happens here: ``feed_artwork`` is a network call and would
    make removing a cover fail whenever the feed is unreachable.
    """
    # Registered-show gate: this unlinks files inside the folder.
    path = require_registered_show(show_folder)

    with _artwork_lock(path):
        _clear_cached_artwork(path)
        meta = load_show_meta(path)
        if meta and meta.artwork_url:
            meta.artwork_url = ""
            save_show_meta(path, meta)
    return {"status": "ok"}


@router.get("/artwork")
async def get_artwork(show_folder: str = Query(...)):
    """Serve cached artwork for a show, downloading it if needed."""
    path = require_show_folder(show_folder)
    meta = load_show_meta(path)
    artwork_url = (meta.artwork_url if meta else "") or ""

    if not artwork_url:
        raise HTTPException(404, "No artwork URL configured")

    if artwork_url == LOCAL_ARTWORK_MARKER:
        local = _find_cached_artwork(path)
        if not local:
            raise HTTPException(404, "No local artwork file")
        # no-cache (revalidate, not "don't cache"): a replaced upload keeps
        # the same URL, so a day-long max-age would pin the old cover.
        return FileResponse(
            local,
            media_type=_MIME.get(local.suffix.lower(), "image/jpeg"),
            headers={"Cache-Control": "no-cache"},
        )

    cached = _fresh_cached_artwork(path, artwork_url)
    if cached is None:
        cached = await asyncio.to_thread(_download_artwork, artwork_url, path)

    if not cached:
        raise HTTPException(502, "Failed to download artwork")

    # no-cache (revalidate, not "don't cache"): the URL never changes, so a
    # day-long max-age would keep serving the old cover after a local upload
    # replaces it. FileResponse's ETag/Last-Modified make revalidation a 304.
    media_type = _MIME.get(cached.suffix.lower(), "image/jpeg")
    return FileResponse(
        cached,
        media_type=media_type,
        headers={"Cache-Control": "no-cache"},
    )


def _merge_into_existing_show(
    show_path: Path, fresh: _ShowMeta, explicit_name: str = ""
) -> _ShowMeta:
    """Reconcile a feed-create request with a ``show.toml`` already on disk.

    Unregistering a show keeps its folder, and re-adding the feed is meant
    to restore it intact: the id in ``show.toml`` is what its collections
    and bot password are keyed on. Writing the request's id-less metadata
    over that file would mint a new id and orphan both, and also drop the
    speakers and pipeline defaults the user set. So the existing metadata
    wins and only the request-supplied fields are overlaid.

    A folder that already belongs to a *different* feed is refused (409),
    the same way ``create_local_show`` refuses an existing directory: the
    picker asks for another folder name rather than repointing that show.
    A local show (no feed URL) is adopted, keeping its id.

    A show can carry an RSS *and* a YouTube URL, so the request's feed kind
    decides which existing URL it has to match, and only that URL is
    rewritten. Comparing against whichever URL happened to be set refused a
    legitimate re-add of the second feed, and overwriting both cleared the
    one the request says nothing about (leaving the show's other source
    unusable until the user retyped it).
    """
    existing = load_show_meta(show_path)
    if existing is None:
        return fresh
    if fresh.rss_url:
        same_kind, other_kind = existing.rss_url, existing.youtube_url
    else:
        same_kind, other_kind = existing.youtube_url, existing.rss_url
    blocking = same_kind or other_kind
    if blocking and same_kind != (fresh.rss_url or fresh.youtube_url):
        raise HTTPException(
            409,
            f"Folder already holds the show {existing.name!r} ({blocking}). "
            "Pick a different folder name.",
        )
    if fresh.rss_url:
        existing.rss_url = fresh.rss_url
    if fresh.youtube_url:
        existing.youtube_url = fresh.youtube_url
    # The user's name wins over the folder or feed fallback; only a name the
    # request actually carried replaces it.
    if explicit_name:
        existing.name = explicit_name
    # A cover the user uploaded is never replaced by the feed's artwork.
    if fresh.artwork_url and existing.artwork_url != LOCAL_ARTWORK_MARKER:
        existing.artwork_url = fresh.artwork_url
    if fresh.language:
        existing.language = fresh.language
    return existing


def _child_of_save_path(save_base: Path, name: str) -> Path:
    """``save_base / name``, refused (400) unless it is a direct child.

    bad_path_component plus a resolve check behind it: a show root is later
    an rmtree target for delete-with-files and move.
    """
    if bad_path_component(name):
        raise HTTPException(400, f"Invalid folder name: {name!r}")
    folder = save_base / name
    if folder.parent.resolve() != save_base.resolve():
        raise HTTPException(400, f"Invalid folder name: {name!r}")
    return folder


def _feed_show_path(save_base: Path, folder_name: str) -> Path:
    """Where a feed-created show lands, refusing anything but a direct child.

    The same checks ``create_local_show`` applies: a name that climbs out of
    the save path, or an existing unrelated directory, must not become a show
    root, since delete-with-files and move later run rmtree on it.
    """
    show_path = _child_of_save_path(save_base, folder_name)
    if (
        show_path.is_dir()
        and load_show_meta(show_path) is None
        and any(show_path.iterdir())
    ):
        raise HTTPException(
            409,
            f"Folder {show_path} already exists and is not a show. "
            "Pick a different folder name.",
        )
    return show_path


def _refuse_taken_label(label: str, folder: Path) -> None:
    """409 when another show already uses this display name.

    The bot, bot access and search pick shows by label, so every route that
    names a show applies the rule rename does, and like rename only to a
    *change*: a folder whose show.toml already carries the name keeps it even
    if it collides (the register_show policy), so a kept show can always be
    restored.
    """
    from podcodex.ingest.show_registry import label_is_taken

    existing = load_show_meta(folder) if folder.is_dir() else None
    if existing is not None and (existing.name or "").strip() == label.strip():
        return
    if label and label_is_taken(label, excluding=folder):
        raise HTTPException(
            409,
            f"Another show is already called {label!r}. Show names are "
            "labels, but the bot picks shows by name, so two shows sharing one "
            "would be ambiguous.",
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

    show_path = _feed_show_path(save_base, folder_name)
    artwork = req.artwork_url or feed_art

    # Display name from search, falling back to the folder name. Merged
    # before anything touches the folder so a 409 leaves it untouched.
    show_name = req.name.strip() or folder_name
    meta = _merge_into_existing_show(
        show_path,
        _ShowMeta(
            name=show_name,
            rss_url=req.rss_url,
            artwork_url=artwork,
            language=req.language,
        ),
        explicit_name=req.name.strip(),
    )
    _refuse_taken_label(meta.name, show_path)
    show_path.mkdir(parents=True, exist_ok=True)
    save_show_meta(show_path, meta)
    _ROSTER_CACHE.pop(str(show_path), None)

    # Cache the feed
    save_feed_cache(show_path, episodes)

    # Register in config
    cfg = _load()
    _register_folder(cfg, str(show_path))

    return CreateFromRSSResponse(
        folder=str(show_path),
        name=meta.name,
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

    show_path = _feed_show_path(save_base, folder_name)

    # Merged before anything touches the folder so a 409 leaves it untouched.
    show_name = req.name.strip() or info.get("name", "") or folder_name
    artwork = req.artwork_url or info.get("artwork_url", "")
    meta = _merge_into_existing_show(
        show_path,
        _ShowMeta(
            name=show_name,
            youtube_url=req.youtube_url,
            artwork_url=artwork,
            language=req.language,
        ),
        explicit_name=req.name.strip(),
    )
    _refuse_taken_label(meta.name, show_path)
    show_path.mkdir(parents=True, exist_ok=True)
    save_show_meta(show_path, meta)
    _ROSTER_CACHE.pop(str(show_path), None)

    # Cache the episode list (same format as RSS)
    save_feed_cache(show_path, episodes)

    # Register in config
    cfg = _load()
    _register_folder(cfg, str(show_path))

    return CreateFromYouTubeResponse(
        folder=str(show_path),
        name=meta.name,
        episode_count=len(episodes),
    )


@router.post("/register")
def register_show(req: RegisterShowRequest) -> dict:
    """Register an existing folder as a known show."""
    p = Path(req.path).expanduser().resolve()
    if not p.is_dir():
        raise HTTPException(400, f"Not a directory: {req.path}")

    # Create show.toml if it doesn't exist yet. A folder that already has one
    # keeps its name even if it collides: refusing it would leave a show on
    # disk the user cannot add, and rename stays the way out.
    if not load_show_meta(p):
        _refuse_taken_label(p.name, p)
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
    accepts_imports = not is_feed_backed(path, meta)
    if meta is None:
        return ShowMeta(
            name=path.name,
            last_feed_update=last_feed_update,
            accepts_imports=accepts_imports,
        )
    # asdict, not a field list: a field added to the dataclass then reaches
    # the settings panel without a second edit here.
    return ShowMeta(
        **asdict(meta),
        last_feed_update=last_feed_update,
        accepts_imports=accepts_imports,
    )


@router.put("/{show_folder:path}/meta")
def update_show_meta(show_folder: str, meta: ShowMeta) -> dict:
    """Persist updated show metadata to show.toml."""
    # Registered-show gate: this writes show.toml unconditionally, so confine
    # it to a tracked show rather than any directory on disk.
    path = require_registered_show(show_folder)

    from podcodex.ingest.show import ensure_show_id

    new_label = (meta.name or "").strip()
    current = load_show_meta(path)
    current_label = (current.name if current else "").strip()

    # Touch the index before show.toml changes. The one-shot show-id
    # migration runs on the first store open of the process and bridges
    # collections by their stored name, so it has to see the old name.
    if new_label and new_label != current_label:
        try:
            get_index_store()
        except Exception:
            logger.opt(exception=True).debug("Index unavailable before rename")
    # Only a *change* is checked (the helper's rule). A show whose name
    # already collides (two installs merged, a hand-edited show.toml) stays
    # editable, so an unrelated edit like a language change is never blocked.
    _refuse_taken_label(new_label, path)

    # Minted after the 409, so a rejected rename never writes to disk.
    # Identity never comes from the request body: the client sends a label,
    # and the id on disk is what every other store keys on.
    show_id = ensure_show_id(path)

    # Built from the request's own field set so the PUT cannot silently drop
    # a field the settings panel sends; the id always comes from disk.
    fields_in = meta.model_dump(
        exclude={"id", "pipeline", "last_feed_update", "accepts_imports"}
    )
    save_show_meta(
        path,
        _ShowMeta(
            id=show_id,
            **fields_in,
            pipeline=_PipelineDefaults(**meta.pipeline.model_dump()),
        ),
    )

    # The label the bot displays is reconciled from show.toml on the next
    # index read (``IndexStore._heal_collection_meta``), so a rename writes
    # one file and nothing else. The password row is the exception: it can
    # exist for a show with no collection, so nothing would read it back.
    if new_label and new_label != current_label:
        _relabel_password(show_id, new_label, current_label)

    return {"status": "saved"}


def _relabel_password(show_id: str, label: str, previous_label: str = "") -> None:
    """Carry a renamed show's label onto its password row, if it has one.

    Collections heal themselves on read; a password row may belong to a show
    that was never indexed, so there is no read path to heal it from.
    """
    try:
        store = get_index_store()
        entries = store.get_show_password_entries()
        entry = entries.get(show_id) or entries.get(previous_label)
        if entry and entry.get("label") != label:
            # A legacy row keyed by the previous name must go, or the bot
            # keeps enforcing the password under that name.
            store.set_show_password(
                show_id,
                entry["password_hash"],
                show_label=label,
                legacy_label=previous_label,
            )
    except Exception:
        # A rename must not fail because the index is busy, absent, or a
        # replica. Identity, the part that used to break, is already safe.
        logger.opt(exception=True).warning(
            f"Could not update stored label for show {show_id!r}"
        )


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
def unified_episodes(show_folder: str) -> list[dict]:
    """Return a merged list of RSS + local episodes.

    Pipeline status comes from the per-show SQLite DB (pipeline.db).
    Step statuses are relative to the app-level pipeline defaults from
    server config, with show-level overrides merged on top. On first
    access the DB is populated from a filesystem scan.
    """
    path = require_show_folder(show_folder)
    ctx = load_status_context(path)

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
            **build_status_out(
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
    # Episode *dirs*, not just the ones holding files: the suffix match in
    # `episode_stem` has to see a suffixed-but-still-empty episode directory.
    # (Its *legacy slug* fallback deliberately stats instead of reading this
    # listing — see `ingest/rss.py` — because this set also carries root audio
    # stems, which would collapse two same-titled episodes onto one stem.)
    # frozenset: lets episode_stem's suffix lookup hit its memoized index.
    existing_stems = frozenset(ctx.local_audio) | frozenset(ctx.episode_dirs)

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
        meta = load_episode_meta(output_dir) if stem in ctx.episode_dirs else None
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
def episode_statuses(show_folder: str) -> list[dict]:
    """Return live pipeline status for every known episode, keyed by stem.

    The cheap counterpart to ``/unified``, meant for the 5s poll the UI runs
    while a download or batch is in flight. It reuses the exact same status
    builder, but skips everything that only feeds the *static* half of an
    episode: the feed cache parse (10-500KB of JSON per request), the
    per-feed-entry stem resolution, and the per-episode ``.episode_meta.json``
    reads. Feed-only episodes with no local footprint are omitted — they have
    no status to report, and the client already holds their static fields.
    """
    path = require_show_folder(show_folder)
    ctx = load_status_context(path)
    return [
        build_status_out(
            stem=stem,
            audio_path=ctx.local_audio.get(stem),
            output_dir=path / stem,
            st=st,
            ep_files=ctx.episode_files.get(stem, []),
            ctx=ctx,
        )
        for stem, st in ctx.status_map.items()
    ]


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

    db = get_pipeline_db(path)
    if db.episode_count() == 0:
        eps = scan_folder(path)
        if eps:
            seed_show_db(path, eps)

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
            return stem, load_version(episode_base(path, stem), step, vid)
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

        for spk, air in speaker_airtime(segments, known).items():
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

    base = episode_base(path, stem)
    segments = load_canonical_segments(base)
    if not segments:
        return EpisodeSpeakersResponse(
            speakers=[], episode_seconds=0.0, has_transcript=False
        )

    show_meta = load_show_meta(path)
    # A show that declares a speaker called "Narrator" means it.
    declared = set(show_meta.speakers) if show_meta else set()
    ep_meta = load_episode_meta(path / stem)
    audio_seconds = float(ep_meta.duration) if ep_meta else 0.0
    last_end = max((float(s.get("end", 0.0)) for s in segments), default=0.0)
    # Denominator is the full episode length so unattributed time (music, gaps,
    # silence) is simply not counted, so the percentages can sum to under 100%.
    denom = max(audio_seconds, last_end)

    air = speaker_airtime(segments, declared)
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


class DeleteEpisodeRequest(BaseModel):
    stem: str


class DeleteEpisodeResponse(BaseModel):
    # "partial": nothing was fully removed and the episode is still listed.
    status: Literal["deleted", "partial"]
    collections: int = 0
    output_dir_removed: bool = False
    audio_removed: bool = False
    db_row_removed: bool = False
    warnings: list[str] = []


def _active_task_on_episode(show_folder: str, stem: str) -> "TaskInfo | None":
    """The task blocking a delete, or None.

    ``task_manager`` locks are plain strings, and each submit path mints its
    own, so every one that can be running against this episode has to be
    enumerated. Missing one means ``rmtree`` runs underneath a live job.

    Show-level keys:

    - ``{show_folder}`` (feed refresh, move, delete)
    - ``batch:{show_folder}`` (``batch.py``'s own key for the whole run)
    - ``download:{show_folder}`` (``rss.py``'s bulk download; it writes
      ``{show}/{stem}.mp3`` on completion, which would resurrect the episode
      as a bare row right after the delete)

    The three show-level shapes come from ``tasks.show_lock_keys`` rather
    than being spelled again here, so this and ``get_active_in_show`` cannot
    drift apart.

    Episode-level keys, all three, because which one is held depends on who
    started the job rather than on the episode:

    - the real ``audio_path``, when there is audio
    - ``{show}/{stem}.mp3``, the synthetic path the *single-episode* runs use
      for an audio-less episode (``stores/episodeStore.ts:useAudioPath``)
    - ``virtual_audio_path(...)``, the ``.virtual`` form the *batch* runner
      uses for the same episode
    """
    from podcodex.api.tasks import show_lock_keys, task_manager
    from podcodex.core.source import episode_audio_files

    show_dir = Path(show_folder)
    ep_base = show_dir / stem
    refs = [
        *show_lock_keys(show_folder),
        f"{ep_base}.mp3",
        virtual_audio_path(ep_base),
        *(str(a) for a in episode_audio_files(show_dir, stem)),
    ]
    for ref in refs:
        active = task_manager.get_active(ref)
        if active:
            return active
    return None


@router.post(
    "/{show_folder:path}/episodes/delete", response_model=DeleteEpisodeResponse
)
def delete_episode_route(
    show_folder: str, req: DeleteEpisodeRequest
) -> DeleteEpisodeResponse:
    """Delete one episode outright: chunks, output dir, audio copy, DB row.

    Distinct from ``DELETE /api/audio/file``, which frees disk by removing
    only the audio and leaves the transcripts re-usable.
    """
    # Registered-show gate: this runs rmtree, so confine it to a tracked show
    # rather than any directory on disk.
    path = require_registered_show(show_folder)
    stem = req.stem.strip()
    if bad_path_component(stem):
        raise HTTPException(400, f"Invalid episode name: {stem!r}")
    # Containment is the service's own guard (it resolves the episode dir's
    # parent against the show), so it holds for direct callers too and there
    # is exactly one authority rather than three overlapping ones here.

    active = _active_task_on_episode(show_folder, stem)
    if active:
        raise HTTPException(
            409,
            f"Task {active.task_id} is running on this show, so wait for it to finish",
        )

    from podcodex.core.delete_episode import delete_episode

    try:
        report = delete_episode(path, stem)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))

    # asdict, not field-by-field: DeleteReport and the response model carry the
    # same fields, and a hand-copied constructor drops new ones silently.
    return DeleteEpisodeResponse(
        status="deleted" if report.files_clean else "partial", **asdict(report)
    )


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
    from podcodex.api.routes._helpers import require_audio_or_output
    from podcodex.core._utils import AudioPaths
    from podcodex.core.versions import backfill_version_sizes, list_all_versions

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    versions = list_all_versions(p.base)
    backfill_version_sizes(p.base, versions)
    return versions


@router.get("/{show_folder:path}/versions")
def list_show_versions(show_folder: str) -> dict[str, list[dict]]:
    """Every episode's versions in one pass, keyed by stem.

    The per-episode ``GET /versions`` above is right for one episode and
    wrong for a batch: opening the batch editor over a 300-episode
    selection fired 300 of them, each hitting the same SQLite file and
    running its own size backfill, before the source-group picker could
    render. Sizes are not backfilled here — that is a per-file stat the
    batch picker does not read.
    """
    from podcodex.core.versions import list_all_versions_by_stem

    path = require_registered_show(show_folder)
    return list_all_versions_by_stem(path)


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
    show_dir = p.show_dir
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


# ── Move / rename show folder ──────────────


class MoveShowRequest(BaseModel):
    new_path: str
    move_files: bool = True


@router.post("/{show_folder:path}/move")
def move_show(show_folder: str, req: MoveShowRequest) -> dict:
    """Move or rename a show folder, optionally relocating all files."""
    # Registered-show gate: move runs shutil.move/rmtree on the source, so
    # confine it to a tracked show rather than any directory on disk.
    old_path = require_registered_show(show_folder)
    new_path = Path(req.new_path).expanduser().resolve()

    old_resolved_path = old_path.resolve()
    if new_path == old_resolved_path:
        raise HTTPException(400, "Source and destination are the same")
    # Nested either way, shutil.move cannot do it: into its own subfolder it
    # raises and the copy fallback then deletes the copy with the source.
    if new_path.is_relative_to(old_resolved_path):
        raise HTTPException(400, "Cannot move a show into one of its own folders")
    if old_resolved_path.is_relative_to(new_path):
        raise HTTPException(400, "Cannot move a show into one of its parent folders")

    if new_path.exists() and (not new_path.is_dir() or any(new_path.iterdir())):
        raise HTTPException(
            409, f"Destination already exists and is not empty: {new_path}"
        )

    # Any lock touching the show blocks: a batch, a bulk download or a
    # single-episode run all keep writing into the folder being moved.
    from podcodex.api.tasks import task_manager

    active = task_manager.get_active_in_show(show_folder)
    if active:
        raise HTTPException(
            409,
            f"Task {active.task_id} is running on this show, wait for it to finish",
        )

    # Release any cached file handles BEFORE the move. On Windows, SQLite (WAL)
    # holds file locks that prevent rename/unlink while open.
    close_pipeline_db(old_path)
    invalidate_scan_cache(old_path)

    leftover_warning: str | None = None

    if req.move_files:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        # shutil.move into an existing directory moves the source *inside*
        # it, so the show would land one level down while config points at
        # the empty folder. It was checked empty above.
        if new_path.exists():
            new_path.rmdir()
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
    old_resolved = str(old_resolved_path)

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
    # Registered-show gate: delete_files runs shutil.rmtree, so confine it to
    # a folder the app actually tracks rather than any directory on disk.
    path = require_registered_show(show_folder)

    # Any lock touching the show blocks: a batch, a bulk download or a
    # single-episode run all keep writing into the folder being deleted.
    from podcodex.api.tasks import task_manager

    active = task_manager.get_active_in_show(show_folder)
    if active:
        raise HTTPException(
            409,
            f"Task {active.task_id} is running on this show, wait for it to finish",
        )

    # Identity resolved before anything is removed: once show.toml is gone
    # there is no way back to the collections this show owns. Minted rather
    # than merely read, because a never-minted show would otherwise have its
    # folder deleted and its collections stranded with nothing to find them by.
    from podcodex.ingest.show import ensure_show_id, show_display

    show_label = show_display(path)
    show_id = ensure_show_id(path)

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

    # The index follows the files. Unregistering keeps both, so re-adding the
    # folder restores the show intact: its id lives in show.toml, so its
    # collections and its password are still keyed to it.
    collections_deleted = 0
    password_removed = False
    purge_error: str | None = None
    if deleted_files:
        collections_deleted, password_removed, purge_error = _purge_show_from_index(
            show_id, show_label
        )

    result: dict = {
        "status": "deleted",
        "files_deleted": deleted_files,
        "collections_deleted": collections_deleted,
        "password_removed": password_removed,
    }
    if purge_error:
        result["warning"] = (
            "The show was deleted, but its search index and bot password "
            f"could not be removed: {purge_error}"
        )
    return result


def _purge_show_from_index(
    show_id: str, show_label: str = ""
) -> tuple[int, bool, str | None]:
    """Drop a deleted show's collections and password.

    Returns ``(collections_deleted, password_removed, error)``; ``error`` is
    set when the index could not be purged, so a failure is not mistaken for
    "nothing was indexed".

    Without this, deleting a show orphans its index exactly as renaming one
    used to: rows nothing can reach, and a password still protecting a name
    that no longer exists.
    """
    if not show_id:
        return 0, False, None
    try:
        store = get_index_store()
        # Label fallback included: on a partially migrated index this show's
        # collections may not carry an id yet, and they would be stranded.
        names = store.collections_for_show(show_id, show_label=show_label)
        for name in names:
            store.delete_collection(name)
        entries = store.get_show_password_entries()
        # id-else-label, matching collections_for_show above: a row that the
        # migration has not rekeyed yet is still keyed by the display name,
        # and would otherwise keep protecting a show that no longer exists.
        pw_key = show_id if show_id in entries else (show_label or "")
        removed = bool(pw_key) and pw_key in entries
        if removed:
            store.delete_show_password(pw_key)
        if names or removed:
            logger.info(
                f"Purged show {show_id!r} from the index: "
                f"{len(names)} collection(s), password removed: {removed}"
            )
        return len(names), removed, None
    except Exception as exc:
        # Deleting the show itself already succeeded; a busy or read-only
        # index must not turn that into a failed request.
        logger.opt(exception=True).warning(
            f"Could not purge show {show_id!r} from the index"
        )
        return 0, False, str(exc) or type(exc).__name__
