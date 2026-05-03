"""Episode export endpoints — text, SRT, VTT, ZIP."""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

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


def _attachment_headers(audio_path: str, ext: str) -> dict[str, str]:
    # Force a download instead of inline rendering — the Tauri webview ignores
    # the <a download> attribute on same-origin text responses.
    stem = Path(audio_path).stem or "export"
    return {"Content-Disposition": f'attachment; filename="{stem}.{ext}"'}


@router.get("/text")
def export_text(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
    source: str = Query("transcript"),
):
    """Export segments as plain text."""
    segments = _load_segments(audio_path, output_dir, source)
    text = segments_to_text(segments)
    return PlainTextResponse(
        text,
        media_type="text/plain; charset=utf-8",
        headers=_attachment_headers(audio_path, "txt"),
    )


@router.get("/srt")
def export_srt(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
    source: str = Query("transcript"),
):
    """Export segments as SRT subtitles."""
    segments = _load_segments(audio_path, output_dir, source)
    srt = segments_to_srt(segments)
    return PlainTextResponse(
        srt,
        media_type="application/x-subrip; charset=utf-8",
        headers=_attachment_headers(audio_path, "srt"),
    )


@router.get("/vtt")
def export_vtt(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
    source: str = Query("transcript"),
):
    """Export segments as WebVTT subtitles."""
    segments = _load_segments(audio_path, output_dir, source)
    vtt = segments_to_vtt(segments)
    return PlainTextResponse(
        vtt,
        media_type="text/vtt; charset=utf-8",
        headers=_attachment_headers(audio_path, "vtt"),
    )


def _write_episode_zip(audio_path: str, output_dir: str | None, target) -> None:
    """Write the episode dir as a ZIP archive into ``target`` (a path or file-like)."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    episode_dir = Path(p.base).parent
    if not episode_dir.exists():
        raise HTTPException(404, "Episode directory not found")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(episode_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.name in _EXCLUDE_NAMES:
                continue
            if any(d in f.parts for d in _EXCLUDE_DIRS):
                continue
            arcname = f.relative_to(episode_dir.parent)
            zf.write(f, arcname)


@router.get("/zip")
def export_zip(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
):
    """Export the entire episode output directory as a ZIP archive."""
    buf = io.BytesIO()
    _write_episode_zip(audio_path, output_dir, buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers=_attachment_headers(audio_path, "zip"),
    )


class ExportSaveRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    source: str = "transcript"
    format: Literal["txt", "srt", "vtt", "zip", "audio"]
    dest: str


@router.post("/save")
def export_save(req: ExportSaveRequest) -> dict:
    """Write an export directly to ``dest`` on disk.

    Used by the desktop app: the renderer opens a native Save-As dialog,
    then posts the chosen path here so the backend can write the file
    without bouncing the bytes through the webview.
    """
    dest = Path(req.dest).expanduser()
    if not dest.parent.is_dir():
        raise HTTPException(400, f"Destination directory does not exist: {dest.parent}")

    if req.format == "zip":
        _write_episode_zip(req.audio_path, req.output_dir, dest)
    elif req.format == "audio":
        src = Path(req.audio_path)
        if not src.is_file():
            raise HTTPException(404, f"Audio file not found: {req.audio_path}")
        shutil.copyfile(src, dest)
    else:
        segments = _load_segments(req.audio_path, req.output_dir, req.source)
        if req.format == "txt":
            content = segments_to_text(segments)
        elif req.format == "srt":
            content = segments_to_srt(segments)
        else:
            content = segments_to_vtt(segments)
        dest.write_text(content, encoding="utf-8")

    return {"status": "saved", "path": str(dest)}
