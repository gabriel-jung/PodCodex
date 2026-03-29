"""Episode export endpoints — text, SRT, VTT, ZIP."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

from podcodex.core._utils import (
    AudioPaths,
    read_json,
    segments_to_srt,
    segments_to_text,
    segments_to_vtt,
)

router = APIRouter()


def _load_segments(audio_path: str, output_dir: str | None, source: str) -> list[dict]:
    """Load segments for the given source (transcript, polished, or a language code)."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    if source == "transcript":
        candidates = [f"{p.base}.transcript.json", f"{p.base}.transcript.raw.json"]
    elif source == "polished":
        candidates = [f"{p.base}.polished.json", f"{p.base}.polished.raw.json"]
    else:
        # Treat as language code
        candidates = [f"{p.base}.{source}.json", f"{p.base}.{source}.raw.json"]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            data = read_json(path)
            # JSON files store segments under a "segments" key
            if isinstance(data, dict) and "segments" in data:
                return data["segments"]
            if isinstance(data, list):
                return data
            return []

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
            if f.is_file():
                arcname = f.relative_to(episode_dir.parent)
                zf.write(f, arcname)

    buf.seek(0)
    stem = Path(audio_path).stem
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{stem}.zip"'},
    )
