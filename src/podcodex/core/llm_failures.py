"""Plain-JSON record of per-batch LLM outcomes for auto (api/ollama) runs.

Not versioned. One file per episode (``llm_failures.json`` in the episode
folder, beside the step directories), keyed by pipeline step (``corrected``
or a translation language key). Each step section lists every batch of the
last auto run with its status, so the user can inspect a batch the pipeline
silently rejected (count drift / parse failure) and fix it by hand.

The file is kept until the user dismisses it (``clear_step``); a clean
re-run overwrites the step section rather than removing it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from podcodex.core._utils import write_json

FAILURES_FILENAME = "llm_failures.json"


def failures_path(base: Path) -> Path:
    """Path to the episode's ``llm_failures.json`` (beside the step folders)."""
    return base.parent / FAILURES_FILENAME


def _read(path: Path) -> dict[str, Any]:
    """Parse a failures file, or ``{}`` when absent / unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("Could not read {}", path)
        return {}


def _has_rejects(section: Any) -> bool:
    """True when a step section records at least one rejected batch."""
    return isinstance(section, dict) and section.get("rejected", 0) > 0


def _base_for(audio_path: str | Path | None, output_dir: str | Path | None) -> Path:
    from podcodex.core._utils import AudioPaths

    return AudioPaths.from_audio(audio_path, output_dir=output_dir).base


def load_failures(base: Path) -> dict[str, Any]:
    """Read the failures file. Returns ``{}`` when absent or unreadable."""
    return _read(failures_path(base))


def save_batch_records(
    base: Path,
    step: str,
    *,
    model: str,
    mode: str,
    records: list[dict],
) -> None:
    """Replace the *step* section with the latest run's per-batch records."""
    data = load_failures(base)
    data[step] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "mode": mode,
        "total_batches": len(records),
        "rejected": sum(1 for r in records if r.get("status") == "rejected"),
        "batches": records,
    }
    write_json(failures_path(base), data)


def clear_step(base: Path, step: str) -> bool:
    """Drop the *step* section. Removes the file when nothing remains.

    Returns True when a section was removed, False when there was none.
    """
    data = load_failures(base)
    if step not in data:
        return False
    del data[step]
    path = failures_path(base)
    if data:
        write_json(path, data)
    else:
        path.unlink(missing_ok=True)
    return True


def step_has_rejects(base: Path, step: str) -> bool:
    """True when the *step* section records at least one rejected batch."""
    return _has_rejects(load_failures(base).get(step))


def rejected_steps(episode_dir: Path) -> list[str]:
    """Steps under *episode_dir* whose last auto run had a rejected batch.

    Reads the failures file directly by directory — the episode-list builder
    has the episode folder, not an AudioPaths base.
    """
    data = _read(episode_dir / FAILURES_FILENAME)
    return [step for step, section in data.items() if _has_rejects(section)]


def record_run(
    audio_path: str | Path | None,
    output_dir: str | Path | None,
    step: str,
    *,
    model: str,
    mode: str,
    records: list[dict],
) -> None:
    """Resolve the episode and persist one auto run's per-batch records.

    No-op when *records* is empty (manual mode, or no batches ran) or when
    the episode cannot be located.
    """
    if not records or not (audio_path or output_dir):
        return
    save_batch_records(
        _base_for(audio_path, output_dir), step, model=model, mode=mode, records=records
    )


def get_step(audio_path: str | None, output_dir: str | None, step: str) -> dict | None:
    """Return the ``llm_failures.json`` section for one step, or None."""
    if not (audio_path or output_dir):
        return None
    return load_failures(_base_for(audio_path, output_dir)).get(step)


def clear_step_for(audio_path: str | None, output_dir: str | None, step: str) -> bool:
    """Drop one step's section, addressed by episode ref. True when removed."""
    if not (audio_path or output_dir):
        return False
    return clear_step(_base_for(audio_path, output_dir), step)
