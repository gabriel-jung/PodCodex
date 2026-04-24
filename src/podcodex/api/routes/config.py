"""App-level configuration — known shows, preferences, user secrets."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from podcodex.core.app_paths import config_dir, secrets_env_path
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

CONFIG_PATH = config_dir() / "config.json"

# Env-var keys the Settings UI can manage. Order is the render order.
SECRET_KEYS: tuple[str, ...] = (
    "HF_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MISTRAL_API_KEY",
    "DISCORD_TOKEN",
)


class AppConfig(BaseModel):
    show_folders: list[str] = []
    default_save_path: str = ""  # suggested location for new shows


def _load() -> AppConfig:
    """Load app config from disk, migrating legacy formats if needed."""
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
    """Persist app config to disk as JSON (atomic write)."""
    from podcodex.core._utils import atomic_write

    atomic_write(
        CONFIG_PATH,
        lambda p: p.write_text(cfg.model_dump_json(indent=2), encoding="utf-8"),
        suffix=".json",
    )


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


# ── User-managed secrets (secrets.env) ────────────────────────────────


def _read_secrets_file() -> dict[str, str]:
    """Parse secrets.env into a dict. Empty dict if absent."""
    path = secrets_env_path()
    if not path.exists():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v}


def _write_secrets_file(values: dict[str, str]) -> None:
    """Atomically write secrets.env, 0600, only non-empty values."""
    from podcodex.core._utils import atomic_write

    def _writer(p: Path) -> None:
        lines = [
            "# PodCodex secrets — managed by the Settings UI. One KEY=value per line.",
            "# Do not commit this file.",
        ]
        for key, value in values.items():
            if not value:
                continue
            # python-dotenv parses quoted strings; quote to be safe with special chars.
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key}="{escaped}"')
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass

    atomic_write(secrets_env_path(), _writer, suffix=".env")


class SecretStatus(BaseModel):
    key: str  # env-var name, e.g. HF_TOKEN
    set: bool  # is a non-empty value available to the backend
    masked: str = ""  # first 4 chars + **** when set
    source: str = "none"  # "file" | "env" | "none"


class SecretsStatusResponse(BaseModel):
    path: str
    items: list[SecretStatus]


class SecretsUpdateRequest(BaseModel):
    # None = leave existing value untouched; "" = clear; str = set.
    values: dict[str, str | None]


def _status_from_file_values(file_values: dict[str, str]) -> SecretsStatusResponse:
    items: list[SecretStatus] = []
    for key in SECRET_KEYS:
        file_val = file_values.get(key, "")
        env_val = os.environ.get(key, "")
        if file_val:
            items.append(
                SecretStatus(key=key, set=True, masked=_mask(file_val), source="file")
            )
        elif env_val:
            items.append(
                SecretStatus(key=key, set=True, masked=_mask(env_val), source="env")
            )
        else:
            items.append(SecretStatus(key=key, set=False, source="none"))
    return SecretsStatusResponse(path=str(secrets_env_path()), items=items)


@router.get("/config/secrets", response_model=SecretsStatusResponse)
async def get_secrets_status() -> SecretsStatusResponse:
    """Report which managed secrets are set and where they come from."""
    return _status_from_file_values(_read_secrets_file())


@router.put("/config/secrets", response_model=SecretsStatusResponse)
async def put_secrets(req: SecretsUpdateRequest) -> SecretsStatusResponse:
    """Update managed secrets on disk and reload into the live environment.

    `values` semantics:
      - key absent or `None` → leave existing value unchanged
      - key with `""`        → clear (remove from file and from env)
      - key with string      → set/replace
    """
    unknown = set(req.values.keys()) - set(SECRET_KEYS)
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"Unknown secret keys: {sorted(unknown)}"
        )

    merged = _read_secrets_file()
    for key, value in req.values.items():
        if value is None:
            continue
        if value == "":
            merged.pop(key, None)
            os.environ.pop(key, None)
        else:
            merged[key] = value

    _write_secrets_file(merged)
    load_dotenv(secrets_env_path(), override=True)
    return _status_from_file_values(merged)


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
    """Return the current app configuration."""
    return _load()


@router.put("/config", response_model=AppConfig)
async def put_config(cfg: AppConfig) -> AppConfig:
    """Persist and return an updated app configuration."""
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
