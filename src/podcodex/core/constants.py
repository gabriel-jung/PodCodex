"""
podcodex.core.constants — Single source of truth for pipeline settings.

Every model name, description, and default value used across the app is
defined here. The desktop API exposes these via ``GET /api/pipeline-config``
so the React frontend can display them without duplicating any text.
"""

from __future__ import annotations

from typing import Literal


# ── Whisper transcription models ─────────────────────────────────────────────
#
# Keys   = model size identifiers passed to WhisperX.
# Values = short descriptions shown in the UI dropdown.

WHISPER_MODELS: dict[str, str] = {
    "large-v3": "Best quality — ~5 GB VRAM (+ batch overhead)",
    "large-v3-turbo": "Near-best quality, 3× faster — ~4 GB VRAM",
    "medium": "Good quality/speed trade-off — ~3 GB VRAM",
    "small": "Faster, slightly less accurate — ~2 GB VRAM",
    "base": "Very fast, lower accuracy — ~1 GB VRAM",
    "tiny": "Fastest, lowest accuracy — ~1 GB VRAM",
}

# Minimum free VRAM (MB) needed to load each model.
# Used by check_vram() to fail fast instead of hanging on OOM.
WHISPER_VRAM_MB: dict[str, int] = {
    "large-v3": 5000,
    "large-v3-turbo": 4000,
    "medium": 3000,
    "small": 2000,
    "base": 1000,
    "tiny": 500,
}

# Pyannote diarization pipeline VRAM requirement (MB).
DIARIZATION_VRAM_MB = 1500

DEFAULT_WHISPER_MODEL = "large-v3-turbo"

# ── Text-to-Speech (TTS) model sizes ────────────────────────────────────────
#
# Qwen-TTS comes in two sizes. The bigger model sounds more natural but
# requires a more powerful GPU.

TTS_MODEL_SIZES: dict[str, str] = {
    "1.7B": "Higher quality voice cloning — needs ~8 GB GPU memory",
    "0.6B": "Faster generation — needs ~4 GB GPU memory",
}

TTS_VRAM_MB: dict[str, int] = {
    "1.7B": 6500,
    "0.6B": 4000,
}

DEFAULT_TTS_MODEL_SIZE = "1.7B"

# ── Assembly strategies ──────────────────────────────────────────────────────
#
# After TTS generates each segment, the assembler stitches them into one
# audio file.  These strategies control how timing gaps are handled.

AssembleStrategy = Literal["silence", "original_timing"]

ASSEMBLE_STRATEGIES: dict[str, str] = {
    "original_timing": "Keep the original pause lengths between speakers",
    "silence": "Use a short fixed pause between speakers, even shorter within a turn",
}

# ── Voice sample extraction defaults ─────────────────────────────────────────

VOICE_MIN_DURATION = 3.0  # minimum clip length (seconds) for a voice sample
VOICE_MAX_DURATION = 0.0  # 0 = no upper limit
VOICE_TOP_K = 3  # how many samples to keep per speaker
DEFAULT_MAX_CHUNK_DURATION = 20.0  # max seconds per TTS chunk
DEFAULT_SILENCE_DURATION = (
    0.2  # speaker-change pause for the "silence" strategy; within-turn auto-scales
)

# ── LLM providers (for Correct & Translate) ───────────────────────────────────
#
# Per-legacy-provider runtime fallbacks for the api mode in run_api(). Used
# only when the caller leaves ``model``/``api_key`` blank — the API path
# normally fills both via ``llm_resolver``. Base URLs and the full provider
# catalog (incl. openai-compatible built-ins like deepseek/gemini/groq) live
# in ``provider_profiles.BUILTIN_PROFILES``.

LLM_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {"model": "gpt-4o-mini", "env_var": "OPENAI_API_KEY"},
    "anthropic": {"model": "claude-sonnet-4-6", "env_var": "ANTHROPIC_API_KEY"},
    "mistral": {"model": "mistral-small-latest", "env_var": "MISTRAL_API_KEY"},
}

DEFAULT_OLLAMA_MODEL = "qwen3.5:27B"  # default model when running locally via Ollama

DEFAULT_SOURCE_LANG = "English"
DEFAULT_TARGET_LANG = "French"

# ── Supported audio file formats ─────────────────────────────────────────────

# Includes the formats yt-dlp's bestaudio can leave behind when the FFmpeg
# post-processor (mp3 conversion) doesn't run — e.g. when ffmpeg is missing.
# Without ``.opus``/``.webm``/``.aac``/``.wma`` here, ``_scan_audio_files``
# misses those files and the episode summary shows no audio_path even
# though the file is on disk and playable.
AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".opus",
    ".webm",
    ".aac",
    ".wma",
}
