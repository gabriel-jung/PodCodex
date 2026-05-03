"""Startup recovery — clean up stale artifacts from prior crashes.

Atomic writes use `<dir>/.tmp_*` temp files and swap them into place with
`os.replace`. A clean exception path unlinks the temp file, but a hard
kill (OOM, power loss, SIGKILL) can leave orphans behind. This module
reaps them at backend startup so they don't accumulate.

Strictly cosmetic: no file ever referenced by the destination is removed,
and the age threshold guarantees we never race with an in-flight write.
"""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

# Files older than this are assumed abandoned. The longest atomic_write
# call in the pipeline is the version save during a batch — well under
# this threshold in practice.
_STALE_AGE_SECONDS = 30 * 60  # 30 minutes
_TEMP_PATTERNS = (".tmp_*", "*.tmp")


def reap_stale_temp_files(
    roots: Iterable[Path], older_than_sec: int = _STALE_AGE_SECONDS
) -> int:
    """Delete `.tmp_*` / `*.tmp` orphans older than ``older_than_sec`` under ``roots``.

    Returns the count of files removed. Missing or unreadable roots are
    silently skipped so this is always safe to call on startup.
    """
    cutoff = time.time() - older_than_sec
    removed = 0
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for pattern in _TEMP_PATTERNS:
            for candidate in root_path.rglob(pattern):
                try:
                    st = candidate.stat()
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode) or st.st_mtime >= cutoff:
                    continue
                try:
                    candidate.unlink()
                    removed += 1
                    logger.debug(f"[recovery] removed stale temp: {candidate}")
                except OSError as exc:
                    logger.debug(f"[recovery] could not remove {candidate}: {exc}")
    return removed


def run_startup_recovery() -> None:
    """Run the full set of startup recovery steps, logging a summary.

    Walks every registered show folder plus the user config directory,
    reaping stale atomic-write temp files from any crash that skipped the
    normal cleanup path.
    """
    from podcodex.api.routes.config import _load as _load_cfg
    from podcodex.core.app_paths import config_dir

    try:
        cfg = _load_cfg()
    except Exception:
        cfg = None

    roots: list[Path] = [config_dir()]
    if cfg:
        for folder in cfg.show_folders:
            p = Path(folder)
            if p.is_dir():
                roots.append(p)

    # Also sweep the LanceDB index dir under <data_dir>/index.
    from podcodex.core.app_paths import data_dir

    candidate = data_dir() / "index"
    if candidate.is_dir() and candidate not in roots:
        roots.append(candidate)
    env_index = os.environ.get("PODCODEX_INDEX")
    if env_index:
        p = Path(env_index)
        if p.is_dir() and p not in roots:
            roots.append(p)

    n = reap_stale_temp_files(roots)
    if n:
        logger.info(f"Startup recovery: reaped {n} stale temp file(s)")
