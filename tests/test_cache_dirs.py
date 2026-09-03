"""Tests for podcodex.core.cache directory resolution.

The model cache must sit under the one definition of the data dir
(``app_paths.data_dir()``). A second definition here is what used to send the
bot's HF cache to ``~/.podcodex/models`` inside the container, a path on no
compose volume, so BGE-M3 was re-downloaded on every recreate.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from podcodex.core import app_paths, cache


@pytest.fixture
def isolated_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    monkeypatch.delenv("PODCODEX_CACHE_DIR", raising=False)
    monkeypatch.setenv("PODCODEX_DATA_DIR", str(tmp_path))
    app_paths.data_dir.cache_clear()
    yield tmp_path
    app_paths.data_dir.cache_clear()


def test_cache_dir_lives_under_the_data_dir(isolated_data_dir: Path) -> None:
    assert cache.get_cache_dir() == isolated_data_dir / "models"


def test_cache_dir_follows_app_paths_without_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No PODCODEX_DATA_DIR (the bot, MCP, dev): still the platform data dir."""
    monkeypatch.delenv("PODCODEX_CACHE_DIR", raising=False)
    monkeypatch.delenv("PODCODEX_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr("sys.platform", "linux")
    app_paths.data_dir.cache_clear()
    try:
        assert cache.get_cache_dir() == app_paths.data_dir() / "models"
    finally:
        app_paths.data_dir.cache_clear()


def test_explicit_override_still_wins(
    isolated_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("PODCODEX_CACHE_DIR", str(override))
    assert cache.get_cache_dir() == override


def test_hf_cache_vars_all_point_at_one_hub(
    isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HF_HOME, HF_HUB_CACHE and TRANSFORMERS_CACHE must not split-brain."""
    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(var, raising=False)
    import os

    hf_dir = cache.get_hf_cache_dir()
    assert hf_dir == isolated_data_dir / "models" / "huggingface"
    assert os.environ["HF_HOME"] == str(hf_dir)
    assert os.environ["HF_HUB_CACHE"] == str(hf_dir / "hub")
    assert os.environ["TRANSFORMERS_CACHE"] == str(hf_dir / "hub")


HF_VARS = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")


def _hf_env(monkeypatch: pytest.MonkeyPatch, setter) -> dict[str, str]:
    """Run one cache setter in an environment with no HF vars, return them."""
    import os

    for var in HF_VARS:
        monkeypatch.delenv(var, raising=False)
    setter()
    return {var: os.environ.get(var, "") for var in HF_VARS}


def test_both_hf_cache_setters_agree(
    isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``get_hf_cache_dir`` and the sidecar's ``_wire_ml_caches`` compute the
    ``<data_dir>/models/huggingface`` layout independently. If they ever
    diverge, ``snapshot_download`` and the transformers loader look in
    different directories and every model appears to be missing."""
    from podcodex.api.server import _wire_ml_caches

    from_core = _hf_env(monkeypatch, cache.get_hf_cache_dir)
    from_sidecar = _hf_env(monkeypatch, _wire_ml_caches)

    assert from_core == from_sidecar
    assert from_core["HF_HOME"] == str(isolated_data_dir / "models" / "huggingface")
    assert from_core["HF_HUB_CACHE"] == from_core["TRANSFORMERS_CACHE"]
    assert from_core["HF_HUB_CACHE"] == str(
        isolated_data_dir / "models" / "huggingface" / "hub"
    )


def test_sidecar_cache_wiring_is_a_noop_without_a_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dev checkout running server.py directly keeps its system HF cache."""
    import os

    from podcodex.api.server import _wire_ml_caches

    monkeypatch.delenv("PODCODEX_DATA_DIR", raising=False)
    for var in HF_VARS:
        monkeypatch.delenv(var, raising=False)
    _wire_ml_caches()
    assert not any(os.environ.get(var) for var in HF_VARS)


def test_delete_cached_model_removes_a_hub_layout_model(
    isolated_data_dir: Path,
) -> None:
    hub = cache.get_hf_cache_dir() / "hub"
    target = hub / "models--BAAI--bge-m3"
    (target / "blobs").mkdir(parents=True)
    (target / "blobs" / "weights.bin").write_bytes(b"x" * 16)

    assert cache.delete_cached_model("models--BAAI--bge-m3") is True
    assert not target.exists()
    assert hub.is_dir()


def test_delete_cached_model_removes_a_flat_layout_model(
    isolated_data_dir: Path,
) -> None:
    """faster-whisper passes ``cache_dir=`` and lands one level higher."""
    hf_root = cache.get_hf_cache_dir()
    target = hf_root / "models--Systran--faster-whisper-large-v3"
    target.mkdir(parents=True)
    (target / "model.bin").write_bytes(b"y")

    assert cache.delete_cached_model("models--Systran--faster-whisper-large-v3") is True
    assert not target.exists()


def test_delete_cached_model_ignores_an_unknown_id(isolated_data_dir: Path) -> None:
    assert cache.delete_cached_model("models--nobody--nothing") is False


def test_delete_cached_model_refuses_a_traversing_id(isolated_data_dir: Path) -> None:
    """The id reaches here from a request body, and the call is an rmtree."""
    hf_root = cache.get_hf_cache_dir()
    outside = hf_root.parent / "sentence-transformers"
    outside.mkdir(parents=True, exist_ok=True)
    (outside / "keep.txt").write_text("keep", encoding="utf-8")

    assert cache.delete_cached_model("../sentence-transformers") is False
    assert cache.delete_cached_model("hub/../../sentence-transformers") is False
    assert (outside / "keep.txt").exists()
