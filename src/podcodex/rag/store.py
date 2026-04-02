"""
podcodex.rag.store — Collection naming utilities for podcast RAG.

Collection naming: "{normalized_show}__{model}__{chunker}"
    e.g. "my_podcast__bge-m3__semantic"
"""

from __future__ import annotations

import re


def _normalize_show(show: str) -> str:
    """Lowercase and collapse non-alphanumeric runs to underscores."""
    return re.sub(r"[^a-z0-9]+", "_", show.lower()).strip("_")


def collection_name(show: str, model: str, chunker: str = "semantic") -> str:
    """Build the canonical collection name: ``{normalized_show}__{model}__{chunker}``.

    Args:
        show: Human-readable show name (normalized internally).
        model: Embedding model key (e.g. ``"bge-m3"``).
        chunker: Chunking strategy key (default ``"semantic"``).

    Returns:
        A deterministic, filesystem-safe collection name string.
    """
    return f"{_normalize_show(show)}__{model}__{chunker}"
