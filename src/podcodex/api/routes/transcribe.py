"""Transcription routes — load/save transcripts and speaker maps."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from podcodex.core._utils import AudioPaths

router = APIRouter()


def _read_segments(path: Path) -> list[dict] | None:
    """Read transcript segments from a JSON file.

    Handles both formats:
    - Plain array: [{speaker, text, start, end}, ...]
    - Wrapped: {meta: {...}, segments: [...]}
    """
    if not path.exists():
        return None
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    return None


@router.get("/segments")
async def get_segments(
    audio_path: str = Query(..., description="Absolute path to audio file"),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load transcript segments (prefers validated over raw)."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    data = _read_segments(p.transcript_best)
    if data is None:
        raise HTTPException(404, "No transcript found")
    return data


@router.get("/segments/raw")
async def get_segments_raw(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load raw (unvalidated) transcript segments."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    data = _read_segments(p.transcript_raw)
    if data is None:
        raise HTTPException(404, "No raw transcript found")
    return data


@router.get("/version-info")
async def version_info(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Return which transcript versions exist."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return {
        "has_raw": p.transcript_raw.exists(),
        "has_validated": p.transcript.exists(),
    }
