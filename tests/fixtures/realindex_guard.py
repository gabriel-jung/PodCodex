"""Pytest plugin: fail any test whose index resolution lands on the real index.

Loaded for every run via ``addopts`` in ``pyproject.toml``.

Why this exists: ``IndexStore()`` with no path resolves the developer's own
``<data_dir>/index``, and opening it is not passive. The store stamps an
ownership marker on a new index, heals collection metadata, and runs the
one-time show-id migration. A test that forgets to isolate the index
therefore rewrites real ``show.toml`` files and real ``_collections`` rows,
silently and outside any tmp_path. That happened once; this makes it loud.

Isolate a test by setting ``PODCODEX_INDEX`` (``tests/fixtures/api_client``
does it for you) or by passing an explicit path to ``IndexStore``.

The real resolver is wrapped rather than replaced, so the ``PODCODEX_INDEX``
override, which is handled *inside* that function, still works and only
genuine escapes are caught.
"""

from __future__ import annotations

import os
from pathlib import Path

import podcodex.rag.index_store as _index_store

_orig_resolve = _index_store._resolve_default_index_path


def _real_index_path() -> Path | None:
    try:
        return Path(_index_store._canonical_index_path()).resolve()
    except Exception:
        return None


def _guarded() -> tuple[Path, str]:
    path, reason = _orig_resolve()
    if os.environ.get("PODCODEX_ALLOW_REAL_INDEX"):
        return path, reason
    real = _real_index_path()
    try:
        landed = Path(path).resolve()
    except OSError:
        return path, reason
    if real is not None and landed == real:
        raise RuntimeError(
            f"This test resolved the REAL index at {landed} (reason: {reason}). "
            "Opening it stamps ownership, heals collection metadata and runs the "
            "show-id migration against real data. Set PODCODEX_INDEX to a tmp_path "
            "(tests/fixtures/api_client.make_client does this) or pass an explicit "
            "path to IndexStore. Set PODCODEX_ALLOW_REAL_INDEX=1 to override."
        )
    return path, reason


_index_store._resolve_default_index_path = _guarded
