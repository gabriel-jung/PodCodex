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
    # Two lists, because the one they replace carried two meanings and every
    # command that touched it had to guess which was in play: unlocking a
    # protected show and pinning a default show are different acts.
    #
    # Password-protected show ids this guild has unlocked. Written by
    # /unlock and /lock only; the access checks read nothing else.
    unlocked_shows: list[str] = field(default_factory=list)
    # Show ids pinned as this guild's defaults, for commands that pick a
    # show when the user names none (/episodes). Written by /setup only.
    # A pin grants no access: a protected show still has to be unlocked.
    pinned_shows: list[str] = field(default_factory=list)
    # Legacy list, from before the split (and, further back, under the name
    # "default_shows"). It records nothing about which command wrote an
    # entry, so it is never reclassified in place: `AccessMixin` reads an
    # entry as an unlock while its show is password-protected and as a pin
    # while it is not, which is what the single list meant before. Entries
    # persist until an admin names that show through `/lock` or
    # `/setup show_remove|show_clear`.
    allowed_shows: list[str] = field(default_factory=list)
    default_source: str = ""
    compact: bool = False
    announce_channel_id: int = 0  # 0 = announcements disabled for this server
