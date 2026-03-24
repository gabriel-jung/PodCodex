"""API route modules — imported by the app factory."""

from podcodex.api.routes import (
    audio,
    config,
    filesystem,
    health,
    rss,
    shows,
    transcribe,
)

__all__ = ["audio", "config", "filesystem", "health", "rss", "shows", "transcribe"]
