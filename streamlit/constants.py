"""
Shared constants for podcodex Streamlit UI.

Centralizes magic strings and numbers that were previously scattered across tabs.
"""

# ── Whisper / transcription ──────────────────────────────────────────────────

WHISPER_MODELS = ["large-v3", "medium", "small"]
DEFAULT_LANGUAGE_CODE = "fr"

# ── TTS / synthesis ──────────────────────────────────────────────────────────

TTS_MODEL_SIZES = ["1.7B", "0.6B"]
DEFAULT_MAX_CHUNK_DURATION = 20.0
DEFAULT_SILENCE_DURATION = 0.5

# Voice sample extraction defaults
VOICE_MIN_DURATION = 3.0
VOICE_MAX_DURATION = 0.0  # 0 = no limit
VOICE_TOP_K = 3

# ── Editor ───────────────────────────────────────────────────────────────────

PAGE_SIZES = [10, 20, 50]
DEFAULT_PAGE_SIZE = 20
AUDIO_PADDING = 0.3  # seconds of context before/after segment audio preview

# ── LLM defaults ─────────────────────────────────────────────────────────────

DEFAULT_SOURCE_LANG = "French"
DEFAULT_TARGET_LANG = "English"
DEFAULT_OLLAMA_MODEL = "qwen3:14b"

# ── Audio file extensions ────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
