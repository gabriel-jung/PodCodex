"""Shared data-dir isolation for tests.

Loaded as a pytest plugin through ``addopts`` (there is no root conftest),
so ``isolated_data_dir`` is available to every test module without an
import.

``app_paths.data_dir`` and ``config_dir`` are lru_cached, so pointing the env
at a tmp dir is not enough on its own: the caches must be cleared on the way
in and out, or the next test resolves whichever dir the first one used.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from podcodex.core import app_paths


@pytest.fixture
def isolated_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point ``data_dir()`` at a fresh tmp dir; the model cache follows it."""
    monkeypatch.setenv("PODCODEX_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("PODCODEX_CACHE_DIR", raising=False)
    app_paths.data_dir.cache_clear()
    yield tmp_path
    app_paths.data_dir.cache_clear()
