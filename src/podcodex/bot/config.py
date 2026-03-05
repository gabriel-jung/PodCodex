"""podcodex.bot.config — Bot configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BotConfig:
    token: str
    strategy: str = "bge_speaker"
    top_k: int = 5
    qdrant_url: str | None = None
