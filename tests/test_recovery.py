"""Tests for podcodex.core.recovery — startup temp-file reaper."""

from __future__ import annotations

import os
import time
from pathlib import Path

from podcodex.core.recovery import reap_stale_temp_files


def _touch(path: Path, age_seconds: float = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    if age_seconds:
        past = time.time() - age_seconds
        os.utime(path, (past, past))
    return path


def test_reap_removes_old_tmp_files(tmp_path: Path):
    _touch(tmp_path / ".tmp_abc.json", age_seconds=3600)
    _touch(tmp_path / "segments.json.tmp", age_seconds=3600)
    kept = _touch(tmp_path / "segments.json")

    removed = reap_stale_temp_files([tmp_path], older_than_sec=60)

    assert removed == 2
    assert kept.exists(), "non-temp file must be preserved"


def test_reap_skips_recent_tmp_files(tmp_path: Path):
    fresh = _touch(tmp_path / ".tmp_abc.json")
    removed = reap_stale_temp_files([tmp_path], older_than_sec=60)
    assert removed == 0
    assert fresh.exists(), "fresh temp files must not be reaped (race safety)"


def test_reap_handles_missing_roots(tmp_path: Path):
    # No exception, no removal, when a root doesn't exist.
    ghost = tmp_path / "does" / "not" / "exist"
    assert reap_stale_temp_files([ghost]) == 0


def test_reap_walks_nested_dirs(tmp_path: Path):
    _touch(tmp_path / "shows" / "foo" / ".versions" / ".tmp_v1.json", age_seconds=3600)
    assert reap_stale_temp_files([tmp_path], older_than_sec=60) == 1
