"""
podcodex.core.constants — Single source of truth for pipeline settings.

Every model name, description, and default value used across the app is
defined here.  Both user interfaces (desktop and Streamlit) read from this
file, so you only ever need to update descriptions in one place.

The desktop API exposes these via ``GET /api/pipeline-config`` so the React
frontend can display them without duplicating any text.
"""

from __future__ import annotations


# ── Whisper transcription models ─────────────────────────────────────────────
#
# Keys   = model size identifiers passed to WhisperX.
# Values = short descriptions shown in the UI dropdown.

WHISPER_MODELS: dict[str, str] = {
    "large-v3": "Best quality — needs a GPU with ~10 GB memory",
    "medium": "Good quality/speed trade-off — needs ~5 GB GPU memory",
    "small": "Faster, slightly less accurate — needs ~2 GB GPU memory",
    "base": "Very fast, lower accuracy — works on most GPUs",
    "tiny": "Fastest, lowest accuracy — useful for quick tests",
}

DEFAULT_WHISPER_MODEL = "large-v3"

# ── Text-to-Speech (TTS) model sizes ────────────────────────────────────────
#
# Qwen-TTS comes in two sizes. The bigger model sounds more natural but
# requires a more powerful GPU.

TTS_MODEL_SIZES: dict[str, str] = {
    "1.7B": "Higher quality voice cloning — needs ~8 GB GPU memory",
    "0.6B": "Faster generation — needs ~4 GB GPU memory",
}

DEFAULT_TTS_MODEL_SIZE = "1.7B"

# ── Assembly strategies ──────────────────────────────────────────────────────
#
# After TTS generates each segment, the assembler stitches them into one
# audio file.  These strategies control how timing gaps are handled.

ASSEMBLE_STRATEGIES: dict[str, str] = {
    "original_timing": "Keep the original pause lengths between speakers",
    "silence": "Replace all pauses with a short fixed silence",
}

# ── Voice sample extraction defaults ─────────────────────────────────────────

VOICE_MIN_DURATION = 3.0  # minimum clip length (seconds) for a voice sample
VOICE_MAX_DURATION = 0.0  # 0 = no upper limit
VOICE_TOP_K = 3  # how many samples to keep per speaker
DEFAULT_MAX_CHUNK_DURATION = 20.0  # max seconds per TTS chunk
DEFAULT_SILENCE_DURATION = 0.5  # gap duration for the "silence" strategy

# ── LLM providers (for Polish & Translate) ───────────────────────────────────
#
# Each provider entry contains the API base URL, a sensible default model,
# and a human-friendly label for the UI.  "ollama" (local) is handled
# separately since it doesn't need a URL.

LLM_PROVIDERS: dict[str, dict[str, str]] = {
    "mistral": {
        "url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
        "label": "Mistral",
        "env_var": "MISTRAL_API_KEY",
    },
    "openai": {
        "url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "label": "OpenAI",
        "env_var": "OPENAI_API_KEY",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-20250514",
        "label": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
    },
    "custom": {
        "url": "",
        "model": "",
        "label": "Custom (OpenAI-compatible)",
        "env_var": "",
    },
}

DEFAULT_OLLAMA_MODEL = "qwen3:14b"  # default model when running locally via Ollama

DEFAULT_SOURCE_LANG = "French"
DEFAULT_TARGET_LANG = "English"

# ── Supported audio file formats ─────────────────────────────────────────────

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
