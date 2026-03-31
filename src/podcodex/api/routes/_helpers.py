"""Shared helpers for API route modules."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException
from loguru import logger

from podcodex.api.schemas import TaskResponse
from podcodex.core._utils import (
    UNKNOWN_SPEAKERS,
    save_segments_json as _core_save_segments,
)

# ── Shared constants ────────────────────────────

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".opus", ".wma"}


# ── Path helpers ────────────────────────────────


def require_show_folder(show_folder: str) -> Path:
    """Resolve a show folder path, raising 404 if it doesn't exist."""
    path = Path(show_folder)
    if not path.is_dir():
        raise HTTPException(404, f"Show folder not found: {show_folder}")
    return path


def is_downloaded(show_folder: Path, stem: str) -> bool:
    """Check if an audio file with the given stem exists in the show folder."""
    return any((show_folder / f"{stem}{ext}").exists() for ext in AUDIO_EXTS)


# ── Task submission ─────────────────────────────


def submit_task(step: str, audio_path: str, fn, *args) -> TaskResponse:
    """Submit a background task.

    If a task is already running on this audio_path, return its task_id
    instead of raising an error — lets the UI reconnect after navigation.
    """
    from podcodex.api.tasks import task_manager

    try:
        info = task_manager.submit(step, audio_path, fn, *args)
    except ValueError:
        # Return existing running task so the UI can reconnect
        existing = task_manager.get_active(audio_path)
        if existing:
            return TaskResponse(task_id=existing.task_id)
        raise HTTPException(409, "A task is already running on this file") from None
    return TaskResponse(task_id=info.task_id)


# Extend the core set with empty string (relevant for flagging UI segments).
_UNKNOWN_SPEAKERS = UNKNOWN_SPEAKERS | {""}


def read_segments(path: Path) -> list[dict] | None:
    """Read segments from a JSON file.

    Handles both formats:
    - Plain array: [{speaker, text, start, end}, ...]
    - Wrapped: {meta: {...}, segments: [...]}
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read segments from {}: {}", path, exc)
        return None

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    return None


def is_flagged(seg: dict) -> bool:
    """Determine whether a segment should be flagged for review."""
    speaker = seg.get("speaker", "")
    if speaker == "[BREAK]":
        return False
    if speaker in _UNKNOWN_SPEAKERS:
        return True
    if speaker == "[remove]":
        return True
    # Low speech density: < 2 chars/s
    dur = seg.get("end", 0) - seg.get("start", 0)
    if dur > 0 and len(seg.get("text", "")) / dur < 2:
        return True
    return False


def annotate_flags(segments: list[dict]) -> list[dict]:
    """Add a ``flagged`` field to each segment."""
    for seg in segments:
        seg["flagged"] = is_flagged(seg)
    return segments


def save_segments_json(
    path: Path,
    segments: list[dict],
    label: str = "Segments",
    provenance: dict | None = None,
) -> int:
    """Write segments to a JSON file. Returns the segment count."""
    _core_save_segments(path, segments, label, provenance=provenance)
    return len(segments)


def load_segments_or_404(path: Path, label: str = "segments") -> list[dict]:
    """Load and annotate segments from a path, raising 404 if not found."""
    data = read_segments(path)
    if data is None:
        raise HTTPException(404, f"No {label} found")
    return annotate_flags(data)


def load_best_source(audio_path: str, output_dir: str | None = None) -> list[dict]:
    """Load the best available source segments (polished → transcript fallback).

    Raises ValueError if no source segments are found.
    """
    from podcodex.core.polish import load_polished
    from podcodex.core.transcribe import load_transcript

    try:
        segments = load_polished(audio_path, output_dir=output_dir)
    except (FileNotFoundError, ValueError):
        segments = None
    if not segments:
        segments = load_transcript(audio_path, output_dir=output_dir)
    if not segments:
        raise ValueError("No source segments found (need transcript or polished)")
    return segments
