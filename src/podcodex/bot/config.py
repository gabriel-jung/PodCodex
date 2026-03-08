"""podcodex.bot.config — Bot configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass

from podcodex.rag.defaults import DEFAULT_MODEL, TOP_K


@dataclass
class BotConfig:
    token: str
    model: str = DEFAULT_MODEL
    top_k: int = TOP_K
    qdrant_url: str | None = None
