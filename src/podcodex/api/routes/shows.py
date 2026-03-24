"""Show and episode management routes."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from fastapi import APIRouter, HTTPException

from podcodex.api.schemas import EpisodeOut, ShowMeta, UnifiedEpisodeOut
from podcodex.ingest.folder import EpisodeInfo, scan_folder
from podcodex.ingest.rss import episode_stem, load_feed_cache
from podcodex.ingest.show import ShowMeta as _ShowMeta
from podcodex.ingest.show import load_show_meta, save_show_meta

router = APIRouter()

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
    path = Path(show_folder)
    if not path.is_dir():
        raise HTTPException(404, f"Show folder not found: {show_folder}")
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
        ),
    )
    return {"status": "saved"}


# ── Episode listing ──────────────────────────


@router.get("/{show_folder:path}/episodes", response_model=list[EpisodeOut])
async def list_episodes(show_folder: str) -> list[dict]:
    path = Path(show_folder)
    if not path.is_dir():
        raise HTTPException(404, f"Show folder not found: {show_folder}")
    episodes = scan_folder(path)
    return [_episode_to_dict(ep) for ep in episodes]


# ── Unified episodes (local + RSS merged) ───


def _is_downloaded(show_folder: Path, stem: str) -> bool:
    for ext in (".mp3", ".m4a", ".wav", ".ogg", ".flac"):
        if (show_folder / f"{stem}{ext}").exists():
            return True
    return False


@router.get(
    "/{show_folder:path}/unified",
    response_model=list[UnifiedEpisodeOut],
)
async def unified_episodes(show_folder: str) -> list[dict]:
    """Return a merged list of RSS + local episodes."""
    path = Path(show_folder)
    if not path.is_dir():
        raise HTTPException(404, f"Show folder not found: {show_folder}")

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
                "downloaded": _is_downloaded(path, stem) if stem else False,
                "transcribed": ep.transcribed if ep else False,
                "polished": ep.polished if ep else False,
                "indexed": ep.indexed if ep else False,
                "translations": ep.translations if ep else [],
                "artwork_url": r.artwork_url or "",
                "raw_transcript": ep.raw_transcript if ep else False,
                "validated_transcript": ep.validated_transcript if ep else False,
            }
        )

    # Local-only episodes (no RSS match)
    for ep in local.values():
        if ep.stem in seen_stems:
            continue
        result.append(
            {
                "id": ep.stem,
                "title": ep.title or ep.stem,
                "stem": ep.stem,
                "pub_date": None,
                "description": "",
                "audio_url": None,
                "duration": 0,
                "episode_number": None,
                "audio_path": str(ep.audio_path) if ep.audio_path else None,
                "downloaded": bool(ep.audio_path),
                "transcribed": ep.transcribed,
                "polished": ep.polished,
                "indexed": ep.indexed,
                "translations": ep.translations,
                "artwork_url": "",
                "raw_transcript": ep.raw_transcript,
                "validated_transcript": ep.validated_transcript,
            }
        )

    return result
