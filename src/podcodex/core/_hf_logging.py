"""HuggingFace cache diagnostics for the bundled sidecar.

The bundled app loads ~5 models (whisperx, pyannote, two embedders, TTS)
on first use. When startup feels slow, the question is always "did it
download or hit cache?" — this module exposes the answer in the log.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from loguru import logger


def log_cached_models() -> None:
    """Dump the currently-cached HuggingFace models to the log.

    Called once at bootstrap so every session starts with a clear
    picture of what is already on disk vs. what will need a download.
    """
    try:
        from podcodex.core.cache import list_cached_models

        models = list_cached_models()
    except Exception as exc:  # noqa: BLE001
        logger.warning("hf-cache: list failed: {!r}", exc)
        return

    if not models:
        logger.info("hf-cache: empty (first run — models will be downloaded on demand)")
        return

    total_mb = sum(m["size_mb"] for m in models)
    logger.info("hf-cache: {} models, {:.0f} MB total", len(models), total_mb)
    for m in models:
        logger.info("  - {} ({:.0f} MB)", m["name"], m["size_mb"])


@contextmanager
def timed_load(label: str):
    """Context manager that logs ``loading <label>`` / ``loaded <label> in <s>``.

    Slow loads (>~5s) almost always indicate a download — the log
    timestamp gap shows it without needing to instrument HF Hub itself.
    """
    logger.info("loading {}", label)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info("loaded {} in {:.1f}s", label, elapsed)
