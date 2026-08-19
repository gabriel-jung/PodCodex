"""Persisted app-level configuration (config.json).

Lives in ``core`` so leaf modules (``_ffmpeg``, future TTS / index code)
can read it without importing from ``api/routes`` — the FastAPI route at
``/api/config`` is the *transport* surface, not the source of truth.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable

from loguru import logger
from pydantic import BaseModel, Field

from podcodex.core.app_paths import config_dir

CONFIG_PATH = config_dir() / "config.json"


class PipelineTranscribeDefaults(BaseModel):
    """App-wide transcribe defaults (Settings → Pipeline)."""

    model_size: str = "large-v3-turbo"
    batch_size: int | None = None
    diarize: bool = False
    clean: bool = False
    num_speakers: str = ""
    language: str = ""


class PipelineLLMDefaults(BaseModel):
    """App-wide LLM defaults for the correct/translate steps."""

    mode: str = "manual"
    provider_profile: str = ""
    key_name: str = ""
    model: str = ""
    models_by_mode: dict[str, str] = Field(
        default_factory=lambda: {"api": "", "ollama": "", "manual": ""}
    )
    context: str = ""
    source_lang: str = "English"
    batch_minutes: float = 15


class PipelineAppDefaults(BaseModel):
    """App-wide pipeline defaults, the base layer under per-show overrides.

    Mirrors the frontend's ``ConfigBundle`` (snake_case, no HF token — that
    lives in the secrets file). The presets are stored user picks, not
    derived values, so they round-trip too.
    """

    transcribe: PipelineTranscribeDefaults = Field(
        default_factory=PipelineTranscribeDefaults
    )
    llm: PipelineLLMDefaults = Field(default_factory=PipelineLLMDefaults)
    engine: str = ""
    target_lang: str = "French"
    index_model: str = "bge-m3"
    index_chunker: str = "semantic"
    transcribe_preset: str = "gpu"
    llm_preset: str = "manual"
    llm_preset_touched: bool = False
    index_preset: str = "balanced"

    def status_defaults(self) -> dict:
        """The app-level half of the step-status ("outdated") comparison.

        Deliberately partial: only these fields participate in outdated
        detection at the app level. ``llm_mode`` / ``llm_provider_profile``
        stay at the unset sentinel ``""`` so a show-level override remains
        the only thing that makes them count; ``llm_models_by_mode`` is an
        input to model resolution, not compared directly. Consumers are
        ``_resolve_defaults`` + ``_step_statuses`` in ``api/routes/shows.py``.
        """
        return {
            "model_size": self.transcribe.model_size,
            "diarize": self.transcribe.diarize,
            "llm_mode": "",
            "llm_provider_profile": "",
            "llm_models_by_mode": dict(self.llm.models_by_mode),
            "target_lang": self.target_lang,
        }


class AppConfig(BaseModel):
    show_folders: list[str] = []
    default_save_path: str = ""  # suggested location for new shows
    # Absolute path to a non-PATH ffmpeg binary. Wired through Tauri to
    # the sidecar's PODCODEX_FFMPEG_EXE env, and read directly here so
    # the dev path (no Tauri) works too.
    ffmpeg_exe_override: str = ""
    # None = never saved (fresh install, or pre-migration client that still
    # holds defaults in localStorage). Readers fall back to the model's
    # built-in defaults; the frontend uses the None sentinel to run its
    # one-time localStorage migration.
    pipeline_defaults: PipelineAppDefaults | None = None


# Hit on every search/list_shows; mtime-keyed so writes auto-invalidate.
_LOAD_CACHE: tuple[float, AppConfig] | None = None

# Serializes every load-modify-save of config.json. Route handlers run on
# FastAPI's threadpool, so two mutations (register a show, save settings)
# can otherwise interleave their load/save windows and lose updates.
_config_lock = threading.RLock()


def load_config() -> AppConfig:
    """Load app config from disk, migrating legacy formats if needed."""
    global _LOAD_CACHE
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except FileNotFoundError:
        return AppConfig()
    except OSError:
        mtime = -1.0

    if _LOAD_CACHE is not None and _LOAD_CACHE[0] == mtime:
        return _LOAD_CACHE[1]

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if "podcast_dir" in data and "show_folders" not in data:
            data["show_folders"] = []
            data["default_save_path"] = data.pop("podcast_dir", "")
        cfg = AppConfig(**data)
    except (json.JSONDecodeError, OSError):
        logger.opt(exception=True).warning(
            "Failed to load config from {}, using defaults", CONFIG_PATH
        )
        return AppConfig()

    _LOAD_CACHE = (mtime, cfg)
    return cfg


def save_config(cfg: AppConfig) -> None:
    """Persist app config to disk as JSON (atomic write)."""
    from podcodex.core._utils import atomic_write

    with _config_lock:
        atomic_write(
            CONFIG_PATH,
            lambda p: p.write_text(cfg.model_dump_json(indent=2), encoding="utf-8"),
            suffix=".json",
        )
        global _LOAD_CACHE
        _LOAD_CACHE = None  # invalidate; next load_config() picks up new mtime


def mutate_config(fn: Callable[[AppConfig], bool | None]) -> AppConfig:
    """Atomically load-modify-save config under the process-wide lock.

    Every read-modify-write of config.json must go through here so
    concurrent handlers can't lose each other's updates. ``fn`` mutates the
    loaded config in place; return ``False`` to skip the save (no change).
    """
    with _config_lock:
        cfg = load_config()
        if fn(cfg) is not False:
            save_config(cfg)
        return cfg


def strip_user_path(raw: str) -> str:
    """Trim whitespace and surrounding quotes from a user-supplied path.

    Windows users often paste ``"C:\\path with spaces\\foo.exe"`` literally
    from a batch file or env var. Mirrors the equivalent strip in
    ``src-tauri/src/lib.rs:read_ffmpeg_override_from_config`` so both
    sides resolve the same way.
    """
    return raw.strip().strip('"').strip("'")
