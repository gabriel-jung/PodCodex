"""Shared helpers for API route modules."""

from __future__ import annotations

import json
from pathlib import Path

from podcodex.core._utils import (
    UNKNOWN_SPEAKERS,
    save_segments_json as _core_save_segments,
)

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
    except (json.JSONDecodeError, OSError):
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
    path: Path, segments: list[dict], label: str = "Segments"
) -> int:
    """Write segments to a JSON file. Returns the segment count."""
    _core_save_segments(path, segments, label)
    return len(segments)


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
