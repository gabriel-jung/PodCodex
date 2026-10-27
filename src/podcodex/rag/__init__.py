"""podcodex.rag — Retrieval-Augmented Generation for podcast transcripts."""

from __future__ import annotations


def rag_available() -> bool:
    """Whether the ``rag`` extra is installed, without importing it.

    The check for optional-rag callers. Importing ``rag.index_store`` is not
    one: its module-scope pyarrow and numpy come with the pipeline extra too,
    while lancedb is only imported when a store opens.
    """
    from importlib.util import find_spec

    return find_spec("lancedb") is not None
