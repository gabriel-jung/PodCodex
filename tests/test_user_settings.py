"""Tests for podcodex.core.user_settings — JSON persistence + device override."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcodex.core import user_settings


def test_load_returns_empty_when_file_corrupt(isolated_data_dir: Path) -> None:
    (isolated_data_dir / "settings.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    assert user_settings.load() == {}


def test_load_returns_empty_when_top_level_not_dict(isolated_data_dir: Path) -> None:
    (isolated_data_dir / "settings.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert user_settings.load() == {}


def test_get_device_override_default_auto(isolated_data_dir: Path) -> None:
    assert user_settings.get_device_override() == "auto"


def test_get_device_override_falls_back_on_invalid_value(
    isolated_data_dir: Path,
) -> None:
    user_settings.save({"device_override": "metal"})
    assert user_settings.get_device_override() == "auto"


def test_set_device_override_auto_clears_key(isolated_data_dir: Path) -> None:
    user_settings.save({"device_override": "cpu", "keep": "yes"})
    user_settings.set_device_override("auto")
    data = user_settings.load()
    assert "device_override" not in data
    assert data == {"keep": "yes"}


def test_set_device_override_rejects_invalid(isolated_data_dir: Path) -> None:
    with pytest.raises(ValueError, match="invalid device_override"):
        user_settings.set_device_override("metal")  # type: ignore[arg-type]
