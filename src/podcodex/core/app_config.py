"""Persisted app-level configuration (config.json).

Lives in ``core`` so leaf modules (``_ffmpeg``, future TTS / index code)
can read it without importing from ``api/routes`` — the FastAPI route at
``/api/config`` is the *transport* surface, not the source of truth.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from podcodex.core.app_paths import config_dir
from podcodex.core.constants import DEFAULT_WHISPER_MODEL
from podcodex.core.json_store import JsonModelStore

CONFIG_PATH = config_dir() / "config.json"


class PipelineTranscribeDefaults(BaseModel):
    """App-wide transcribe defaults (Settings → Pipeline)."""

    model_size: str = DEFAULT_WHISPER_MODEL
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
        ``_resolve_defaults`` + ``_step_statuses`` in ``core/episode_status.py``.
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


def _migrate_legacy(data: dict) -> dict:
    """Pre-multi-show configs had a single ``podcast_dir``."""
    if "podcast_dir" in data and "show_folders" not in data:
        data["show_folders"] = []
        data["default_save_path"] = data.pop("podcast_dir", "")
    return data


# Hit on every search/list_shows; mtime-keyed so writes auto-invalidate.
# The path is read at call time so tests can point CONFIG_PATH elsewhere.
_STORE = JsonModelStore(lambda: CONFIG_PATH, AppConfig, migrate=_migrate_legacy)


def load_config() -> AppConfig:
    """Load app config from disk, migrating legacy formats if needed.

    Returns a copy: mutating it changes nothing until :func:`save_config`.
    A file that cannot be read or validated loads as defaults (see
    ``JsonModelStore``), and is moved aside rather than overwritten on the
    next save.
    """
    return _STORE.load()


def save_config(cfg: AppConfig) -> None:
    """Persist app config to disk as JSON (atomic write)."""
    _STORE.save(cfg)


def mutate_config(fn: Callable[[AppConfig], bool | None]) -> AppConfig:
    """Atomically load-modify-save config under the process-wide lock.

    Every read-modify-write of config.json must go through here so
    concurrent handlers can't lose each other's updates. ``fn`` mutates the
    loaded config in place; return ``False`` to skip the save (no change).
    """
    return _STORE.mutate(fn)


def strip_user_path(raw: str) -> str:
    """Trim whitespace and surrounding quotes from a user-supplied path.

    Windows users often paste ``"C:\\path with spaces\\foo.exe"`` literally
    from a batch file or env var. Mirrors the equivalent strip in
    ``src-tauri/src/lib.rs:read_ffmpeg_override_from_config`` so both
    sides resolve the same way.
    """
    return raw.strip().strip('"').strip("'")
