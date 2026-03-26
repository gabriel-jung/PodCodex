"""API route modules — imported by the app factory."""

from podcodex.api.routes import (
    audio,
    config,
    filesystem,
    health,
    polish,
    rss,
    shows,
    transcribe,
    translate,
    ws,
)

__all__ = [
    "audio",
    "config",
    "filesystem",
    "health",
    "polish",
    "rss",
    "shows",
    "transcribe",
    "translate",
    "ws",
]
