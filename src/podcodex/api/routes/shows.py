"""Show and episode management routes."""

from __future__ import annotations

import re
import shutil
from dataclasses import fields
from pathlib import Path

import hashlib
import urllib.request

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
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
    SpeakerEpisodeEntry,
    SpeakerRosterEntry,
    SpeakerRosterResponse,
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
async def register_show(req: RegisterShowRequest) -> dict:
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
    try:
        app_defaults = _json.loads(defaults) if defaults else {}
    except _json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid JSON in 'defaults' parameter: {exc}")
    show_meta = load_show_meta(path)
    effective = _resolve_defaults(app_defaults, show_meta)

    # ── Pipeline status from DB (or one-time migration) ──
    db = get_pipeline_db(path)
    if db.episode_count() == 0:
        episodes = scan_folder(path)
        if episodes:
            db.populate_from_scan(episodes)

    status_map: dict[str, dict] = {row["stem"]: row for row in db.all_episodes()}
    seg_counts = db.latest_segment_counts("transcript")

    local_audio = _scan_audio_files(path)
    episode_files = _scan_episode_files(path, local_audio)

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
    ) -> dict:
        prov = _normalize_provenance(st.get("provenance", {}))
        seg_count = seg_counts.get(stem) if stem else None
        return {
            "id": ep_id,
            "title": title,
            "stem": stem,
            "pub_date": pub_date,
            "description": description,
            "audio_url": audio_url,
            "duration": duration,
            "episode_number": episode_number,
            "audio_path": str(audio_path) if audio_path else None,
            "output_dir": str(output_dir)
            if output_dir and output_dir.is_dir()
            else None,
            "downloaded": audio_path is not None,
            "transcribed": st.get("transcribed", False),
            "corrected": st.get("corrected", False),
            "indexed": st.get("indexed", False),
            "synthesized": st.get("synthesized", False),
            "has_subtitles": any(f.endswith(".vtt") for f in ep_files),
            "translations": st.get("translations", []),
            "artwork_url": artwork_url,
            "segment_count": seg_count,
            "files": ep_files,
            "provenance": prov,
            **_step_statuses(st, prov, effective),
        }

    # RSS episodes first (preserves feed order)
    for r in rss:
        stem = episode_stem(r)
        if r.guid in seen_ids:
            continue
        seen_ids.add(r.guid)
        st = status_map.get(stem, {}) if stem else {}
        audio_path = local_audio.get(stem)
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
                ep_files=episode_files.get(stem, []) if stem else [],
            )
        )

    # Local-only episodes (no RSS match)
    for stem, st in status_map.items():
        if stem in seen_stems:
            continue
        output_dir = path / stem
        meta = load_episode_meta(output_dir) if output_dir.is_dir() else None
        ep_id = meta.guid if meta else stem
        if ep_id in seen_ids:
            continue
        seen_ids.add(ep_id)
        audio_path = local_audio.get(stem)
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
                ep_files=episode_files.get(stem, []),
            )
        )

    return result


_PARAM_RENAMES = {"mode": "llm_mode", "provider": "llm_provider"}


def _normalize_provenance(prov: dict) -> dict:
    """Rename legacy param keys (mode→llm_mode, provider→llm_provider)."""
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
    if not (show_meta and show_meta.pipeline):
        return effective
    p = show_meta.pipeline
    if p.model_size:
        effective["model_size"] = p.model_size
    if p.llm_mode:
        effective["llm_mode"] = p.llm_mode
    if p.llm_provider:
        effective["llm_provider"] = p.llm_provider
    if p.llm_model:
        effective["llm_model"] = p.llm_model
    if p.target_lang:
        effective["target_lang"] = p.target_lang
    if p.diarize is not None:
        effective["diarize"] = p.diarize
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
        effective.get("llm_provider")
        and params.get("llm_provider") != effective["llm_provider"]
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


def _step_statuses(st: dict, provenance: dict, effective: dict) -> dict:
    """Compute per-step status: 'none' | 'outdated' | 'done'.

    Compares the episode's provenance against the effective defaults.
    """

    def _check_transcribe() -> str:
        if not st.get("transcribed", False):
            return "none"
        prov = provenance.get("transcript")
        if not prov:
            return "done"  # no provenance → legacy, assume done
        return "outdated" if _transcribe_outdated(prov, effective) else "done"

    def _check_correct() -> str:
        if not st.get("corrected", False):
            return "none"
        prov = provenance.get("corrected")
        if not prov or not effective:
            return "done"
        return "outdated" if _llm_outdated(prov, effective) else "done"

    def _check_translate() -> str:
        translations = st.get("translations", [])
        if not translations:
            return "none"
        target = effective.get("target_lang", "").strip().lower()
        if target and target not in translations:
            return "none"
        lang_key = target or (translations[0] if translations else "")
        prov = provenance.get(lang_key)
        if not prov or not effective:
            return "done"
        return "outdated" if _llm_outdated(prov, effective) else "done"

    return {
        "transcribe_status": _check_transcribe(),
        "correct_status": _check_correct(),
        "translate_status": _check_translate(),
    }


def _compute_speaker_roster(path: Path) -> SpeakerRosterResponse:
    from podcodex.core._utils import BREAK_SPEAKER, UNKNOWN_SPEAKERS, group_by_speaker
    from podcodex.core.versions import load_latest

    db = get_pipeline_db(path)
    if db.episode_count() == 0:
        eps = scan_folder(path)
        if eps:
            db.populate_from_scan(eps)

    meta = load_show_meta(path)
    known = set(meta.speakers) if meta else set()

    totals: dict[str, dict] = {}
    per_episode: dict[str, list[SpeakerEpisodeEntry]] = {}
    episodes_scanned = 0
    episodes_with_transcripts = 0

    for ep in db.all_episodes():
        stem = ep["stem"]
        episodes_scanned += 1
        base = path / stem / stem
        segments = load_latest(base, "corrected") or load_latest(base, "transcript")
        if not segments:
            continue
        episodes_with_transcripts += 1

        ep_meta = load_episode_meta(path / stem)
        ep_title = ep_meta.title if ep_meta and ep_meta.title else stem

        for spk, segs in group_by_speaker(segments).items():
            if not spk or spk == BREAK_SPEAKER or spk in UNKNOWN_SPEAKERS:
                continue
            secs = sum(
                max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
                for s in segs
            )
            row = totals.setdefault(
                spk,
                {"episode_count": 0, "segment_count": 0, "total_seconds": 0.0},
            )
            row["episode_count"] += 1
            row["segment_count"] += len(segs)
            row["total_seconds"] += secs
            per_episode.setdefault(spk, []).append(
                SpeakerEpisodeEntry(
                    stem=stem,
                    title=ep_title,
                    segment_count=len(segs),
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

    return SpeakerRosterResponse(
        speakers=entries,
        episodes_scanned=episodes_scanned,
        episodes_with_transcripts=episodes_with_transcripts,
    )


@router.get(
    "/{show_folder:path}/speakers/roster",
    response_model=SpeakerRosterResponse,
)
async def speakers_roster(show_folder: str) -> SpeakerRosterResponse:
    """Aggregate speaker stats across every transcribed episode in the show.

    For each episode the most recent ``corrected`` segments are preferred,
    falling back to the latest raw transcript. Placeholder labels from
    ``UNKNOWN_SPEAKERS`` and the ``[BREAK]`` sentinel are filtered out.
    Speakers listed in ``show.toml`` that never appear are still returned
    with zero counts so the UI can surface configured-but-unseen names.
    """
    import asyncio

    path = require_show_folder(show_folder)
    return await asyncio.get_running_loop().run_in_executor(
        None, _compute_speaker_roster, path
    )


@router.post("/{show_folder:path}/resync")
async def resync_pipeline_db(show_folder: str) -> dict:
    """Force-rebuild pipeline.db from filesystem scan."""
    path = require_show_folder(show_folder)
    close_pipeline_db(path)
    from podcodex.core.pipeline_db import DB_FILENAME

    db_file = path / DB_FILENAME
    if db_file.exists():
        db_file.unlink()
    db = get_pipeline_db(path)
    episodes = scan_folder(path)
    if episodes:
        db.populate_from_scan(episodes)
    return {"status": "resynced", "episode_count": len(episodes)}


@router.get("/versions")
async def list_all_versions(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """List versions across all pipeline steps for an episode, newest first."""
    from podcodex.core._utils import AudioPaths
    from podcodex.core.versions import list_all_versions

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return list_all_versions(p.base)


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


_INTERESTING_EXTS = {
    ".mp3",
    ".m4a",
    ".wav",
    ".ogg",
    ".flac",  # audio
    ".vtt",
    ".srt",  # subtitles
    ".json",
    ".parquet",  # transcripts / pipeline outputs
}
_SKIP_PREFIXES = (".", "__")
_SKIP_NAMES = {"manifest.json"}


def _walk_episode_dir(root: Path, rel_prefix: str) -> list[str]:
    """Recursively collect interesting files under an episode dir.

    ``rel_prefix`` is the path (relative to the show folder) to prepend to
    each file name, so we skip allocating a Path per entry just to call
    ``relative_to``.
    """
    import os

    collected: list[str] = []
    try:
        with os.scandir(root) as it:
            for f in it:
                name = f.name
                if name.startswith(_SKIP_PREFIXES):
                    continue
                if f.is_dir(follow_symlinks=False):
                    collected.extend(
                        _walk_episode_dir(Path(f.path), f"{rel_prefix}/{name}")
                    )
                    continue
                if not f.is_file(follow_symlinks=False) or name in _SKIP_NAMES:
                    continue
                dot = name.rfind(".")
                if dot <= 0 or name[dot:].lower() not in _INTERESTING_EXTS:
                    continue
                collected.append(f"{rel_prefix}/{name}")
    except OSError:
        pass
    return collected


def _scan_episode_files(
    show_folder: Path, local_audio: dict[str, Path]
) -> dict[str, list[str]]:
    """Scan episode subdirectories for user-facing files.

    Returns a mapping of stem → list of filenames relative to show folder.
    Walks version subdirectories (``transcript/``, ``corrected/``,
    ``speaker_map/``, language folders, etc.) so the Pipeline file list
    surfaces version artifacts alongside legacy flat files.
    """
    import os

    result: dict[str, list[str]] = {}
    try:
        with os.scandir(show_folder) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                stem = entry.name
                if stem.startswith("."):
                    continue
                files = _walk_episode_dir(Path(entry.path), stem)
                if files:
                    files.sort()
                    result[stem] = files
    except OSError:
        pass

    # Prepend root audio (already discovered by _scan_audio_files).
    for stem, audio_path in local_audio.items():
        result.setdefault(stem, []).insert(0, audio_path.name)

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
