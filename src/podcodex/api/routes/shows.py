"""Show and episode management routes."""

from __future__ import annotations

import re
import shutil
from dataclasses import fields
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from podcodex.api.routes._helpers import require_show_folder
from podcodex.api.routes.config import _load, _register_folder, _save
from podcodex.api.schemas import (
    CreateFromRSSRequest,
    CreateFromRSSResponse,
    CreateFromYouTubeRequest,
    CreateFromYouTubeResponse,
    EpisodeOut,
    PipelineDefaultsSchema,
    RegisterShowRequest,
    ShowMeta,
    UnifiedEpisodeOut,
)
from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db
from podcodex.ingest.folder import EpisodeInfo, invalidate_scan_cache, scan_folder
from podcodex.ingest.rss import (
    episode_stem,
    feed_artwork,
    fetch_feed,
    load_episode_meta,
    load_feed_cache,
    save_feed_cache,
)
from podcodex.ingest.show import PipelineDefaults as _PipelineDefaults
from podcodex.ingest.show import ShowMeta as _ShowMeta
from podcodex.ingest.show import load_show_meta, save_show_meta

router = APIRouter()


# ── Show listing & creation ─────────────────


class ShowSummary(BaseModel):
    name: str
    path: str
    episode_count: int = 0
    has_rss: bool = False
    has_youtube: bool = False
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
        has_youtube = bool(meta and meta.youtube_url)
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
                has_youtube=has_youtube,
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


@router.post("/from-youtube", response_model=CreateFromYouTubeResponse)
async def create_from_youtube(
    req: CreateFromYouTubeRequest,
) -> CreateFromYouTubeResponse:
    """Fetch YouTube metadata and create a show folder."""
    from podcodex.ingest.youtube import fetch_youtube, youtube_show_info

    save_base = Path(req.save_path).expanduser()
    if not save_base.is_dir():
        raise HTTPException(400, f"Save path does not exist: {req.save_path}")

    # Get channel/playlist info for show name and artwork
    try:
        info = youtube_show_info(req.youtube_url)
    except ImportError as exc:
        raise HTTPException(501, str(exc)) from None
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch YouTube info: {exc}") from None

    # Fetch episode list
    try:
        episodes = fetch_youtube(req.youtube_url)
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
    """Return metadata for a show folder."""
    path = require_show_folder(show_folder)
    meta = load_show_meta(path)
    if meta is None:
        return ShowMeta(name=path.name)
    return ShowMeta(
        name=meta.name,
        rss_url=meta.rss_url,
        youtube_url=meta.youtube_url,
        language=meta.language,
        speakers=meta.speakers,
        artwork_url=meta.artwork_url,
        pipeline=PipelineDefaultsSchema(
            model_size=meta.pipeline.model_size,
            diarize=meta.pipeline.diarize,
            llm_mode=meta.pipeline.llm_mode,
            llm_provider=meta.pipeline.llm_provider,
            llm_model=meta.pipeline.llm_model,
            target_lang=meta.pipeline.target_lang,
        ),
    )


@router.put("/{show_folder:path}/meta")
async def update_show_meta(show_folder: str, meta: ShowMeta) -> dict:
    """Persist updated show metadata to show.toml."""
    path = Path(show_folder)
    path.mkdir(parents=True, exist_ok=True)
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
            pipeline=_PipelineDefaults(
                model_size=p.model_size,
                diarize=p.diarize,
                llm_mode=p.llm_mode,
                llm_provider=p.llm_provider,
                llm_model=p.llm_model,
                target_lang=p.target_lang,
            ),
        ),
    )
    return {"status": "saved"}


# ── Episode listing ──────────────────────────


@router.get("/{show_folder:path}/episodes", response_model=list[EpisodeOut])
async def list_episodes(show_folder: str) -> list[dict]:
    """List locally scanned episodes for a show folder."""
    path = require_show_folder(show_folder)
    episodes = scan_folder(path)
    return [_episode_to_dict(ep) for ep in episodes]


# ── Unified episodes (local + RSS merged) ───


@router.get(
    "/{show_folder:path}/unified",
    response_model=list[UnifiedEpisodeOut],
)
async def unified_episodes(
    show_folder: str,
    defaults: str | None = None,
) -> list[dict]:
    """Return a merged list of RSS + local episodes.

    Pipeline status comes from the per-show SQLite DB (pipeline.db).
    On first access the DB is populated from a filesystem scan.

    Args:
        defaults: Optional JSON string with app-level pipeline defaults
                  (model_size, diarize, llm_mode, llm_provider, llm_model,
                  target_lang). Show-level overrides take precedence.
    """
    import json as _json

    path = require_show_folder(show_folder)

    # ── Resolve effective defaults (app → show override) ──
    app_defaults = _json.loads(defaults) if defaults else {}
    show_meta = load_show_meta(path)
    effective = _resolve_defaults(app_defaults, show_meta)

    # ── Pipeline status from DB (or one-time migration) ──
    db = get_pipeline_db(path)
    if db.episode_count() == 0:
        episodes = scan_folder(path)
        if episodes:
            db.populate_from_scan(episodes)

    status_map: dict[str, dict] = {row["stem"]: row for row in db.all_episodes()}

    # ── Audio file discovery (single scandir at show root) ──
    local_audio = _scan_audio_files(path)
    episode_files = _scan_episode_files(path)

    rss = load_feed_cache(path) or []

    result: list[dict] = []
    seen_stems: set[str] = set()

    # RSS episodes first (preserves feed order)
    for r in rss:
        stem = episode_stem(r)
        st = status_map.get(stem, {}) if stem else {}
        audio_path = local_audio.get(stem)
        if stem:
            seen_stems.add(stem)
        prov = st.get("provenance", {})
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
                "audio_path": str(audio_path) if audio_path else None,
                "output_dir": str(path / stem)
                if stem and (path / stem).is_dir()
                else None,
                "downloaded": audio_path is not None,
                "transcribed": st.get("transcribed", False),
                "polished": st.get("polished", False),
                "indexed": st.get("indexed", False),
                "synthesized": st.get("synthesized", False),
                "translations": st.get("translations", []),
                "artwork_url": r.artwork_url or "",
                "files": episode_files.get(stem, []) if stem else [],
                "provenance": prov,
                **_step_statuses(st, prov, effective),
            }
        )

    # Local-only episodes (no RSS match)
    for stem, st in status_map.items():
        if stem in seen_stems:
            continue
        output_dir = path / stem
        meta = load_episode_meta(output_dir) if output_dir.is_dir() else None
        audio_path = local_audio.get(stem)
        prov = st.get("provenance", {})
        result.append(
            {
                "id": meta.guid if meta else stem,
                "title": (meta.title if meta else None) or stem,
                "stem": stem,
                "pub_date": meta.pub_date if meta else None,
                "description": (meta.description or "") if meta else "",
                "audio_url": (meta.audio_url or None) if meta else None,
                "duration": meta.duration if meta else 0,
                "episode_number": meta.episode_number if meta else None,
                "audio_path": str(audio_path) if audio_path else None,
                "output_dir": str(output_dir) if output_dir.is_dir() else None,
                "downloaded": audio_path is not None,
                "transcribed": st.get("transcribed", False),
                "polished": st.get("polished", False),
                "indexed": st.get("indexed", False),
                "synthesized": st.get("synthesized", False),
                "translations": st.get("translations", []),
                "artwork_url": (meta.artwork_url or "") if meta else "",
                "files": episode_files.get(stem, []),
                "provenance": prov,
                **_step_statuses(st, prov, effective),
            }
        )

    return result


def _resolve_defaults(app_defaults: dict, show_meta: _ShowMeta | None) -> dict:
    """Merge app-level defaults with show-level overrides.

    Show-level non-empty values win; otherwise fall back to app defaults.
    """
    effective = dict(app_defaults)
    if show_meta and show_meta.pipeline:
        p = show_meta.pipeline
        for key in (
            "model_size",
            "diarize",
            "llm_mode",
            "llm_provider",
            "llm_model",
            "target_lang",
        ):
            val = getattr(p, key, None)
            # Only override if show has a non-default value
            if val is not None and val != "" and val is not True:
                effective[key] = val
            elif key == "diarize" and not p.diarize:
                # Explicitly set to False overrides
                effective[key] = False
    return effective


def _step_statuses(st: dict, provenance: dict, effective: dict) -> dict:
    """Compute per-step status: 'none' | 'outdated' | 'done'.

    Compares the episode's provenance against the effective defaults.
    """

    def _check_transcribe() -> str:
        """Return transcribe status by comparing stored provenance with effective defaults."""
        if not st.get("transcribed", False):
            return "none"
        prov = provenance.get("transcript")
        if not prov:
            return "done"  # no provenance → legacy, assume done
        # Raw transcript (import / never validated) → outdated (yellow)
        if prov.get("type") != "validated" and not prov.get("manual_edit"):
            return "outdated"
        if not effective:
            return "done"
        params = prov.get("params", {})
        if effective.get("model_size") and prov.get("model") != effective["model_size"]:
            return "outdated"
        if "diarize" in effective and params.get("diarize") != effective["diarize"]:
            return "outdated"
        return "done"

    def _check_polish() -> str:
        """Return polish status by comparing stored provenance with effective defaults."""
        if not st.get("polished", False):
            return "none"
        prov = provenance.get("polished")
        if not prov or not effective:
            return "done"
        params = prov.get("params", {})
        if effective.get("llm_mode") and params.get("mode") != effective["llm_mode"]:
            return "outdated"
        if (
            effective.get("llm_provider")
            and params.get("provider") != effective["llm_provider"]
        ):
            return "outdated"
        if effective.get("llm_model") and prov.get("model") != effective["llm_model"]:
            return "outdated"
        return "done"

    def _check_translate() -> str:
        """Return translate status by comparing stored provenance with effective defaults."""
        translations = st.get("translations", [])
        if not translations:
            return "none"
        target = effective.get("target_lang", "").strip().lower()
        if target and target not in translations:
            return "none"  # target lang not translated at all
        # Check provenance for the target language
        lang_key = target or (translations[0] if translations else "")
        prov = provenance.get(lang_key)
        if not prov or not effective:
            return "done"
        params = prov.get("params", {})
        if effective.get("llm_mode") and params.get("mode") != effective["llm_mode"]:
            return "outdated"
        if (
            effective.get("llm_provider")
            and params.get("provider") != effective["llm_provider"]
        ):
            return "outdated"
        if effective.get("llm_model") and prov.get("model") != effective["llm_model"]:
            return "outdated"
        return "done"

    return {
        "transcribe_status": _check_transcribe(),
        "polish_status": _check_polish(),
        "translate_status": _check_translate(),
    }


@router.post("/{show_folder:path}/resync")
async def resync_pipeline_db(show_folder: str) -> dict:
    """Force-rebuild pipeline.db from filesystem scan."""
    path = require_show_folder(show_folder)
    close_pipeline_db(path)
    db_file = path / "pipeline.db"
    if db_file.exists():
        db_file.unlink()
    db = get_pipeline_db(path)
    episodes = scan_folder(path)
    if episodes:
        db.populate_from_scan(episodes)
    return {"status": "resynced", "episode_count": len(episodes)}


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


def _scan_episode_files(show_folder: Path) -> dict[str, list[str]]:
    """Scan episode subdirectories for user-facing files.

    Returns a mapping of stem → list of filenames relative to show folder
    (e.g. ``["stem/stem.subtitles.fr.vtt", "stem/stem.transcript.raw.json"]``).
    Includes audio at show root. Skips dotfiles, version dirs, and DB files.
    """
    import os

    _INTERESTING_EXTS = {
        ".mp3",
        ".m4a",
        ".wav",
        ".ogg",
        ".flac",  # audio
        ".vtt",
        ".srt",  # subtitles
        ".json",  # transcripts / pipeline outputs
    }
    _SKIP_PREFIXES = (".", "__")
    _SKIP_NAMES = {"manifest.json"}

    # Collect audio at show root keyed by stem
    root_audio: dict[str, str] = {}
    result: dict[str, list[str]] = {}
    try:
        with os.scandir(show_folder) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    name = entry.name
                    dot = name.rfind(".")
                    if dot > 0 and name[dot:].lower() in (
                        ".mp3",
                        ".m4a",
                        ".wav",
                        ".ogg",
                        ".flac",
                    ):
                        root_audio[name[:dot]] = name
                elif entry.is_dir(follow_symlinks=False) and not entry.name.startswith(
                    "."
                ):
                    stem = entry.name
                    files: list[str] = []
                    subpath = show_folder / stem
                    try:
                        with os.scandir(subpath) as sub_it:
                            for f in sub_it:
                                if not f.is_file(follow_symlinks=False):
                                    continue
                                fname = f.name
                                if any(fname.startswith(p) for p in _SKIP_PREFIXES):
                                    continue
                                if fname in _SKIP_NAMES:
                                    continue
                                fdot = fname.rfind(".")
                                if (
                                    fdot > 0
                                    and fname[fdot:].lower() in _INTERESTING_EXTS
                                ):
                                    files.append(f"{stem}/{fname}")
                    except OSError:
                        pass
                    if files:
                        files.sort()
                        result[stem] = files
    except OSError:
        pass

    # Prepend root audio
    for stem, audio_name in root_audio.items():
        result.setdefault(stem, []).insert(0, audio_name)

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
    close_pipeline_db(old_path)
    invalidate_scan_cache(old_path)
    invalidate_scan_cache(new_path)

    return {"status": "moved", "new_path": str(new_path)}


class DeleteShowRequest(BaseModel):
    delete_files: bool = False


@router.post("/{show_folder:path}/delete")
async def delete_show(show_folder: str, req: DeleteShowRequest) -> dict:
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

    # Remove from config.json
    cfg = _load()
    resolved = str(path.resolve())
    cfg.show_folders = [
        p for p in cfg.show_folders if str(Path(p).resolve()) != resolved
    ]
    _save(cfg)

    # Optionally delete the folder on disk
    deleted_files = False
    if req.delete_files and path.exists():
        shutil.rmtree(path)
        deleted_files = True
        logger.info("Deleted show folder: {}", path)
    else:
        logger.info("Unregistered show (files kept): {}", path)

    return {"status": "deleted", "files_deleted": deleted_files}
