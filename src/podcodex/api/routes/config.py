"""App-level configuration — known shows, preferences."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel

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
from podcodex.ingest.rss import search_itunes

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
            logger.opt(exception=True).warning(
                "Failed to load config from {}, using defaults", CONFIG_PATH
            )
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


def _mask(value: str) -> str:
    """Show first 4 chars + asterisks for a secret value."""
    if len(value) <= 4:
        return "****"
    return value[:4] + "****"


def _detect_env_keys() -> dict[str, str]:
    """Return masked values for known API keys found in the environment."""
    import os

    detected: dict[str, str] = {}
    hf = os.environ.get("HF_TOKEN", "")
    if hf:
        detected["hf_token"] = _mask(hf)
    for provider, spec in LLM_PROVIDERS.items():
        env_var = spec.get("env_var", "")
        if env_var:
            val = os.environ.get(env_var, "")
            if val:
                detected[provider] = _mask(val)
    return detected


@router.get("/pipeline-config")
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
        "detected_keys": _detect_env_keys(),
    }


@router.get("/config", response_model=AppConfig)
async def get_config() -> AppConfig:
    return _load()


@router.put("/config", response_model=AppConfig)
async def put_config(cfg: AppConfig) -> AppConfig:
    _save(cfg)
    return cfg


# ── Podcast search ────────────────────────────


class PodcastSearchResultOut(BaseModel):
    name: str
    artist: str
    feed_url: str
    artwork_url: str = ""


@router.get("/podcasts/search", response_model=list[PodcastSearchResultOut])
async def search_podcasts(q: str, limit: int = 8) -> list[PodcastSearchResultOut]:
    """Search Apple Podcasts / iTunes for a podcast by name."""
    if not q.strip():
        return []
    results = search_itunes(q.strip(), limit=limit)
    return [PodcastSearchResultOut(**r.__dict__) for r in results]
