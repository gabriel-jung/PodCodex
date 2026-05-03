"""Tests for podcodex.core.user_settings — JSON persistence + device override."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from podcodex.core import app_paths, user_settings


@pytest.fixture
def isolated_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point ``data_dir()`` at a fresh tmp dir and clear its lru_cache."""
    monkeypatch.setenv("PODCODEX_DATA_DIR", str(tmp_path))
    app_paths.data_dir.cache_clear()
    yield tmp_path
    app_paths.data_dir.cache_clear()


def test_load_returns_empty_when_no_file(isolated_data_dir: Path) -> None:
    assert user_settings.load() == {}


def test_load_returns_empty_when_file_corrupt(isolated_data_dir: Path) -> None:
    (isolated_data_dir / "settings.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    assert user_settings.load() == {}


def test_load_returns_empty_when_top_level_not_dict(isolated_data_dir: Path) -> None:
    (isolated_data_dir / "settings.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert user_settings.load() == {}


def test_save_then_load_roundtrip(isolated_data_dir: Path) -> None:
    user_settings.save({"device_override": "cpu", "other": 42})
    assert user_settings.load() == {"device_override": "cpu", "other": 42}


def test_save_writes_atomically(isolated_data_dir: Path) -> None:
    user_settings.save({"device_override": "cpu"})
    settings_file = isolated_data_dir / "settings.json"
    assert settings_file.exists()
    # tmp file should not linger on success
    assert not (isolated_data_dir / "settings.json.tmp").exists()
    # contents are well-formed JSON
    json.loads(settings_file.read_text(encoding="utf-8"))


def test_get_device_override_default_auto(isolated_data_dir: Path) -> None:
    assert user_settings.get_device_override() == "auto"


def test_get_device_override_reads_persisted(isolated_data_dir: Path) -> None:
    user_settings.save({"device_override": "cpu"})
    assert user_settings.get_device_override() == "cpu"


def test_get_device_override_falls_back_on_invalid_value(
    isolated_data_dir: Path,
) -> None:
    user_settings.save({"device_override": "metal"})
    assert user_settings.get_device_override() == "auto"


def test_set_device_override_persists(isolated_data_dir: Path) -> None:
    user_settings.set_device_override("cpu")
    assert user_settings.load()["device_override"] == "cpu"


def test_set_device_override_auto_clears_key(isolated_data_dir: Path) -> None:
    user_settings.save({"device_override": "cpu", "keep": "yes"})
    user_settings.set_device_override("auto")
    data = user_settings.load()
    assert "device_override" not in data
    assert data == {"keep": "yes"}


def test_set_device_override_rejects_invalid(isolated_data_dir: Path) -> None:
    with pytest.raises(ValueError, match="invalid device_override"):
        user_settings.set_device_override("metal")  # type: ignore[arg-type]
