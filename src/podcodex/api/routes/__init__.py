"""API route modules — imported by the app factory."""

from podcodex.api.routes import (
    audio,
    config,
    filesystem,
    health,
    index,
    polish,
    rss,
    search,
    shows,
    synthesize,
    transcribe,
    translate,
    ws,
)

__all__ = [
    "audio",
    "config",
    "filesystem",
    "health",
    "index",
    "polish",
    "rss",
    "search",
    "shows",
    "synthesize",
    "transcribe",
    "translate",
    "ws",
]
