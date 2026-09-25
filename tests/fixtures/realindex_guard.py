"""Pytest plugin: keep tests off the developer's real index, config and data.

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

import atexit
import os
import shutil
import tempfile
from pathlib import Path

# Before anything imports podcodex: point the config and data dirs at a
# session scratch dir, so a test that forgets to isolate them writes there
# instead of the developer's real config.json, api_keys.json or data dir
# (app_config computes CONFIG_PATH at import). Tests that need a specific
# dir still set their own; tests that need these unset delenv them.
_SCRATCH = Path(tempfile.mkdtemp(prefix="podcodex-tests-"))
atexit.register(shutil.rmtree, _SCRATCH, ignore_errors=True)
os.environ["XDG_CONFIG_HOME"] = str(_SCRATCH / "config")
os.environ["PODCODEX_DATA_DIR"] = str(_SCRATCH / "data")

import podcodex.rag.index_store as _index_store  # noqa: E402

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
