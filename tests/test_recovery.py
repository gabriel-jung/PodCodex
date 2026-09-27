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
    kept = _touch(tmp_path / "segments.json")

    removed = reap_stale_temp_files([tmp_path], older_than_sec=60)

    assert removed == 1
    assert kept.exists(), "non-temp file must be preserved"


def test_reap_leaves_other_apps_tmp_files_alone(tmp_path: Path):
    """Show folders can hold other programs' files; a bare `*.tmp` pattern
    deleted them. Only PodCodex's `.tmp_` prefix is reaped."""
    foreign = _touch(tmp_path / "project.aup3.tmp", age_seconds=3600)
    assert reap_stale_temp_files([tmp_path], older_than_sec=60) == 0
    assert foreign.exists()


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


def test_tagged_temp_files_are_still_reapable(tmp_path: Path):
    """Callers tag their temp files ("prompts_", "claude_cfg_"); a process
    killed mid-write leaves one behind, and the startup reaper must find it."""
    import os
    import time

    from podcodex.core._utils import atomic_write
    from podcodex.core.recovery import reap_stale_temp_files

    seen: list[str] = []

    def crash(tmp: Path) -> None:
        seen.append(tmp.name)
        raise RuntimeError("killed mid-write")

    try:
        atomic_write(tmp_path / "prompts.json", crash, tag="prompts_")
    except RuntimeError:
        pass
    # What a kill (rather than an exception) leaves: the temp file itself.
    orphan = tmp_path / seen[0]
    orphan.write_text("partial")
    old = time.time() - 7 * 86400
    os.utime(orphan, (old, old))

    assert reap_stale_temp_files([tmp_path]) == 1
    assert not orphan.exists()


def test_atomic_write_keeps_normal_permissions(tmp_path):
    """mkstemp creates 0600 and os.replace published it, so every episode
    file became owner-only."""
    import os
    import stat

    from podcodex.core._utils import _default_file_mode, atomic_write

    fresh = tmp_path / "fresh.json"
    atomic_write(fresh, lambda p: p.write_text("{}"))
    assert stat.S_IMODE(fresh.stat().st_mode) == _default_file_mode()

    secret = tmp_path / "secret.json"
    secret.write_text("{}")
    os.chmod(secret, 0o600)
    atomic_write(secret, lambda p: p.write_text("{}"))
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
