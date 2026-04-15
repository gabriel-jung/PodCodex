"""Episode export endpoints — text, SRT, VTT, ZIP."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

from podcodex.core._utils import (
    AudioPaths,
    normalize_lang,
    segments_to_srt,
    segments_to_text,
    segments_to_vtt,
)
from podcodex.core.pipeline_db import DB_FILENAME
from podcodex.core.versions import load_latest

router = APIRouter()

_EXCLUDE_NAMES = frozenset(
    {
        DB_FILENAME,
        DB_FILENAME + "-wal",
        DB_FILENAME + "-shm",
    }
)
_EXCLUDE_DIRS = frozenset({".versions"})


def _load_segments(audio_path: str, output_dir: str | None, source: str) -> list[dict]:
    """Load segments for the given source (transcript, corrected, or a language code)."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    step = normalize_lang(source)
    segments = load_latest(p.base, step)
    if segments is not None:
        return segments

    raise HTTPException(404, f"No segments found for source={source}")


@router.get("/text")
def export_text(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
    source: str = Query("transcript"),
):
    """Export segments as plain text."""
    segments = _load_segments(audio_path, output_dir, source)
    text = segments_to_text(segments)
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")


@router.get("/srt")
def export_srt(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
    source: str = Query("transcript"),
):
    """Export segments as SRT subtitles."""
    segments = _load_segments(audio_path, output_dir, source)
    srt = segments_to_srt(segments)
    return PlainTextResponse(srt, media_type="text/plain; charset=utf-8")


@router.get("/vtt")
def export_vtt(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
    source: str = Query("transcript"),
):
    """Export segments as WebVTT subtitles."""
    segments = _load_segments(audio_path, output_dir, source)
    vtt = segments_to_vtt(segments)
    return PlainTextResponse(vtt, media_type="text/vtt; charset=utf-8")


@router.get("/zip")
def export_zip(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
):
    """Export the entire episode output directory as a ZIP archive."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    episode_dir = Path(p.base).parent
    if not episode_dir.exists():
        raise HTTPException(404, "Episode directory not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(episode_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.name in _EXCLUDE_NAMES:
                continue
            if any(d in f.parts for d in _EXCLUDE_DIRS):
                continue
            arcname = f.relative_to(episode_dir.parent)
            zf.write(f, arcname)

    buf.seek(0)
    stem = Path(audio_path).stem
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{stem}.zip"'},
    )
