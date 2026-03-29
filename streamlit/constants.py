"""
Shared constants for podcodex Streamlit UI.

Pipeline constants (models, providers, strategies) are imported from the
core package so they stay in sync with the desktop API and frontend.
Only Streamlit-specific constants (editor UI settings) are defined here.
"""

# Re-export pipeline constants from the single source of truth
from podcodex.core.constants import (  # noqa: F401
    ASSEMBLE_STRATEGIES,
    AUDIO_EXTENSIONS,
    DEFAULT_MAX_CHUNK_DURATION,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SILENCE_DURATION,
    DEFAULT_SOURCE_LANG,
    DEFAULT_TARGET_LANG,
    DEFAULT_TTS_MODEL_SIZE,
    DEFAULT_WHISPER_MODEL,
    LLM_PROVIDERS,
    TTS_MODEL_SIZES,
    VOICE_MAX_DURATION,
    VOICE_MIN_DURATION,
    VOICE_TOP_K,
    WHISPER_MODELS,
)

# Backwards-compat: Streamlit tabs use the list form
WHISPER_MODELS_LIST: list[str] = list(WHISPER_MODELS.keys())
TTS_MODEL_SIZES_LIST: list[str] = list(TTS_MODEL_SIZES.keys())

DEFAULT_LANGUAGE_CODE = "fr"

# ── Streamlit-only UI constants ──────────────────────────────────────────────

PAGE_SIZES = [10, 20, 50]
DEFAULT_PAGE_SIZE = 20
AUDIO_PADDING = 0.3  # seconds of context before/after segment audio preview
