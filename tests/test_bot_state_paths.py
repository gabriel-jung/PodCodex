"""Where the bot keeps its per-guild state.

``server_config.json`` (each server's /setup defaults, unlocked shows and
announce channel) plus ``search_cache.db`` and ``announce_state.db`` used to
default to the working directory, which under Docker is the discarded
container layer: every ``docker compose up -d --build`` wiped them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcodex.bot.bot import _STATE_FILES, _resolve_state_path


def test_default_lands_in_the_data_dir(
    isolated_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _resolve_state_path(None)
    assert path == isolated_data_dir / "bot" / "server_config.json"
    # The two SQLite stores are placed beside it, so they move with it.
    assert path.parent.is_dir()


def test_explicit_override_wins(isolated_data_dir: Path, tmp_path: Path) -> None:
    override = tmp_path / "custom" / "server_config.json"
    assert _resolve_state_path(str(override)) == override


def test_legacy_working_directory_state_is_migrated_once(
    isolated_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "app"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    (cwd / "server_config.json").write_text(json.dumps({"1": {"top_k": 7}}))
    (cwd / "search_cache.db").write_bytes(b"cache")
    (cwd / "announce_state.db").write_bytes(b"announce")

    path = _resolve_state_path(None)

    assert json.loads(path.read_text()) == {"1": {"top_k": 7}}
    for name in _STATE_FILES:
        assert (path.parent / name).is_file()
        assert not (cwd / name).exists()

    # Idempotent: a later run with a fresh legacy file must not clobber the
    # migrated config.
    (cwd / "server_config.json").write_text(json.dumps({"2": {"top_k": 1}}))
    assert _resolve_state_path(None) == path
    assert json.loads(path.read_text()) == {"1": {"top_k": 7}}
