"""Tests for bootstrap._apply_persisted_device_override.

Covers the precedence rule: ``PODCODEX_DEVICE`` env wins over the
persisted ``device_override`` setting; persisted only fills in when env
is unset (or empty).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcodex import bootstrap
from podcodex.core import user_settings


@pytest.fixture
def isolated(isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("PODCODEX_DEVICE", raising=False)
    return isolated_data_dir


def test_no_persisted_no_env_leaves_env_unset(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    bootstrap._apply_persisted_device_override()
    assert "PODCODEX_DEVICE" not in os.environ


def test_persisted_cpu_promotes_to_env(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    user_settings.set_device_override("cpu")
    bootstrap._apply_persisted_device_override()
    assert os.environ["PODCODEX_DEVICE"] == "cpu"


def test_persisted_auto_does_not_set_env(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    user_settings.set_device_override("auto")
    bootstrap._apply_persisted_device_override()
    assert "PODCODEX_DEVICE" not in os.environ


def test_explicit_env_wins_over_persisted(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    user_settings.set_device_override("cpu")
    monkeypatch.setenv("PODCODEX_DEVICE", "cuda")
    bootstrap._apply_persisted_device_override()
    assert os.environ["PODCODEX_DEVICE"] == "cuda"
