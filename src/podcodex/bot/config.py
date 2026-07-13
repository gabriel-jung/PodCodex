"""Bot configuration dataclasses: global (CLI) and per-server settings."""

from __future__ import annotations

from dataclasses import dataclass, field

from podcodex.rag.defaults import DEFAULT_CHUNKING, DEFAULT_MODEL, TOP_K


@dataclass
class BotConfig:
    """Global bot configuration (set via CLI flags, immutable at runtime)."""

    model: str = DEFAULT_MODEL
    chunker: str = DEFAULT_CHUNKING
    top_k: int = TOP_K
    index_path: str | None = None
    merge_strategy: str = "roundrobin"
    cooldown_seconds: float = 5.0
    dev_guild_id: int | None = None
    announce_interval_minutes: int = 10


@dataclass
class ServerSettings:
    """Per-server overrides persisted to server_config.json."""

    model: str = DEFAULT_MODEL
    chunker: str = DEFAULT_CHUNKING
    top_k: int = TOP_K
    allowed_shows: list[str] = field(default_factory=list)
    default_source: str = ""
    compact: bool = False
    announce_channel_id: int = 0  # 0 = announcements disabled for this server
