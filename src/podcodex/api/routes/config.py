"""App-level configuration — known shows, preferences."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from podcodex.api.schemas import (
    CreateFromRSSRequest,
    CreateFromRSSResponse,
    RegisterShowRequest,
)
from podcodex.core.constants import (
    ASSEMBLE_STRATEGIES,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SOURCE_LANG,
    DEFAULT_TARGET_LANG,
    DEFAULT_TTS_MODEL_SIZE,
    DEFAULT_WHISPER_MODEL,
    LLM_PROVIDERS,
    TTS_MODEL_SIZES,
    WHISPER_MODELS,
)
from podcodex.ingest.rss import feed_artwork, fetch_feed, save_feed_cache, search_itunes
from podcodex.ingest.show import ShowMeta as _ShowMeta
from podcodex.ingest.show import load_show_meta, save_show_meta

router = APIRouter()

CONFIG_PATH = Path.home() / ".config" / "podcodex" / "config.json"


class AppConfig(BaseModel):
    show_folders: list[str] = []
    default_save_path: str = ""  # suggested location for new shows


def _load() -> AppConfig:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            # Migrate from old podcast_dir format
            if "podcast_dir" in data and "show_folders" not in data:
                data["show_folders"] = []
                data["default_save_path"] = data.pop("podcast_dir", "")
            return AppConfig(**data)
        except (json.JSONDecodeError, OSError):
            pass
    return AppConfig()


def _save(cfg: AppConfig) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")


def _register_folder(cfg: AppConfig, folder_path: str) -> AppConfig:
    """Add a folder to known shows if not already tracked."""
    resolved = str(Path(folder_path).resolve())
    existing = {str(Path(p).resolve()) for p in cfg.show_folders}
    if resolved not in existing:
        cfg.show_folders.append(resolved)
        _save(cfg)
    return cfg


@router.get("/api/pipeline-config")
async def pipeline_config() -> dict:
    """Return all pipeline constants (models, providers, strategies).

    The React frontend fetches this once at startup so that labels,
    descriptions, and defaults live in Python — never duplicated in TS.
    """
    return {
        "whisper_models": WHISPER_MODELS,
        "default_whisper_model": DEFAULT_WHISPER_MODEL,
        "tts_model_sizes": TTS_MODEL_SIZES,
        "default_tts_model_size": DEFAULT_TTS_MODEL_SIZE,
        "assemble_strategies": ASSEMBLE_STRATEGIES,
        "llm_providers": LLM_PROVIDERS,
        "default_ollama_model": DEFAULT_OLLAMA_MODEL,
        "default_source_lang": DEFAULT_SOURCE_LANG,
        "default_target_lang": DEFAULT_TARGET_LANG,
    }


@router.get("/api/config", response_model=AppConfig)
async def get_config() -> AppConfig:
    return _load()


@router.put("/api/config", response_model=AppConfig)
async def put_config(cfg: AppConfig) -> AppConfig:
    _save(cfg)
    return cfg


# ── Podcast search ────────────────────────────


class PodcastSearchResultOut(BaseModel):
    name: str
    artist: str
    feed_url: str
    artwork_url: str = ""


@router.get("/api/podcasts/search", response_model=list[PodcastSearchResultOut])
async def search_podcasts(q: str, limit: int = 8) -> list[PodcastSearchResultOut]:
    """Search Apple Podcasts / iTunes for a podcast by name."""
    if not q.strip():
        return []
    results = search_itunes(q.strip(), limit=limit)
    return [PodcastSearchResultOut(**r.__dict__) for r in results]


# ── Create show from RSS ─────────────────────


@router.post("/api/shows/from-rss", response_model=CreateFromRSSResponse)
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

    # Save show metadata
    save_show_meta(
        show_path,
        _ShowMeta(
            name=folder_name,
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


# ── Register existing folder ─────────────────


@router.post("/api/shows/register")
async def register_show(req: RegisterShowRequest) -> dict:
    """Register an existing folder as a known show."""
    p = Path(req.path).expanduser().resolve()
    if not p.is_dir():
        raise HTTPException(400, f"Not a directory: {req.path}")
    cfg = _load()
    _register_folder(cfg, str(p))
    return {"status": "ok", "path": str(p)}


# ── Show listing ─────────────────────────────


class ShowSummary(BaseModel):
    name: str
    path: str
    episode_count: int = 0
    has_rss: bool = False
    artwork_url: str = ""


@router.get("/api/shows", response_model=list[ShowSummary])
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

        # Backfill artwork from RSS feed if missing
        if not artwork and meta and meta.rss_url:
            try:
                artwork = feed_artwork(meta.rss_url)
                if artwork:
                    meta.artwork_url = artwork
                    save_show_meta(child, meta)
            except Exception:
                pass

        audio_count = sum(
            1
            for f in child.iterdir()
            if f.is_file() and f.suffix in (".mp3", ".m4a", ".wav", ".ogg", ".flac")
        )

        has_rss = (child / "rss_cache.json").exists()
        shows.append(
            ShowSummary(
                name=name,
                path=str(child),
                episode_count=audio_count,
                has_rss=has_rss,
                artwork_url=artwork,
            )
        )
    return shows
