"""Shared isolation for tests: env, data dir and index.

Loaded as a pytest plugin through ``addopts`` (there is no root conftest),
so these fixtures are available to every test module without an import.

``app_paths.data_dir`` and ``config_dir`` are lru_cached, so pointing the env
at a tmp dir is not enough on its own: the caches must be cleared on the way
in and out, or the next test resolves whichever dir the first one used. The
same holds for ``get_index_store``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from podcodex.core import app_paths


@pytest.fixture(autouse=True)
def _restore_environ() -> Iterator[None]:
    """Put ``os.environ`` back exactly as it was before each test.

    ``monkeypatch.delenv(name, raising=False)`` on an unset variable records
    no restore, so when the code under test then writes that variable itself
    (the device override, ``SSL_CERT_FILE``, the HF cache vars) the value
    outlives the test and every later module sees it. Snapshotting here
    catches that whatever the test did.
    """
    before = dict(os.environ)
    yield
    if os.environ != before:
        os.environ.clear()
        os.environ.update(before)


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


@pytest.fixture
def isolated_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point ``get_index_store()`` at ``tmp_path / "index"`` for one test."""
    from podcodex.rag import index_store

    path = tmp_path / "index"
    monkeypatch.setenv("PODCODEX_INDEX", str(path))
    index_store.get_index_store.cache_clear()
    yield path
    index_store.get_index_store.cache_clear()
