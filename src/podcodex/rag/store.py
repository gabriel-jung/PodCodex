"""
podcodex.rag.store — Collection naming utilities for podcast RAG.

Collection naming: "{show_id}__{model}__{chunker}"
    e.g. "my_podcast_3f2a91c4__bge-m3__semantic"

The name is an internal detail of ``IndexStore``. Callers go from a show to
its collection through ``IndexStore.resolve_collection``, never by rebuilding
the string: a caller that rebuilds it is what a show rename used to orphan.
"""

from __future__ import annotations

import re


def _normalize_show(show: str) -> str:
    """Lowercase and collapse non-alphanumeric runs to underscores.

    Show ids are minted already normalized (see ``ingest.show.ensure_show_id``),
    so this is a no-op for them. It still matters for the legacy names of
    collections written before ids existed, which the migration has to be able
    to reconstruct in order to find and rename them.
    """
    return re.sub(r"[^a-z0-9]+", "_", show.lower()).strip("_")


def collection_name(show: str, model: str, chunker: str = "semantic") -> str:
    """Build the canonical collection name: ``{show}__{model}__{chunker}``.

    Args:
        show: Show id (normalized internally; already-normalized ids pass
            through unchanged). Pass a display name only when reconstructing
            a legacy name during migration.
        model: Embedding model key (e.g. ``"bge-m3"``).
        chunker: Chunking strategy key (default ``"semantic"``).

    Returns:
        A deterministic, filesystem-safe collection name string.
    """
    return f"{_normalize_show(show)}__{model}__{chunker}"
