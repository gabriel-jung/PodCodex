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

from podcodex.core._utils import AudioPaths, normalize_lang
from podcodex.core.constants import AUDIO_EXTENSIONS
from podcodex.core.subtitles import segments_to_srt, segments_to_text, segments_to_vtt
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


def _declared_speakers(audio_path: str, output_dir: str | None) -> set[str]:
    """The show's declared speakers, for the exports' placeholder check.

    A show can legitimately declare a speaker called "Narrator"; without this
    their lines would be the only ones exported with no name (see
    core/_utils.is_unattributed).
    """
    from podcodex.ingest.show import load_show_meta

    try:
        p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
        meta = load_show_meta(p.show_dir)
    except Exception:
        return set()
    return set(meta.speakers) if meta else set()


def _confined_paths(
    audio_path: str, output_dir: str | None
) -> tuple[str | None, str | None]:
    """Confine the caller-supplied episode paths to a registered show folder.

    Without this an absolute ``output_dir`` makes the episode directory any
    directory on disk, which ``/zip`` then streams back wholesale. Same guard
    ``/api/audio/file`` applies to its own ``path`` param.
    """
    from podcodex.api.routes._helpers import resolve_inside_show_root

    if not audio_path and not output_dir:
        raise HTTPException(400, "audio_path or output_dir is required")
    # An empty audio_path is an episode with no audio (a YouTube subtitle
    # import): Path("").resolve() is the process cwd, which no show contains,
    # so resolving it refused every export of such an episode.
    safe_audio = str(resolve_inside_show_root(audio_path)) if audio_path else None
    safe_out = str(resolve_inside_show_root(output_dir)) if output_dir else None
    return safe_audio, safe_out


def _load_segments(
    audio_path: str,
    output_dir: str | None,
    source: str,
    version_id: str | None = None,
) -> list[dict]:
    """Load segments for the given source (transcript, corrected, or a language code).

    *version_id* exports that version (the one the viewer shows) instead of
    the step's default.
    """
    audio_path, output_dir = _confined_paths(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    step = normalize_lang(source)
    from podcodex.api.routes._helpers import load_version_or_404, shape_step_segments

    # Shaped like the viewer shows them (a translation carries the current
    # speaker map), so the exported file matches the screen.
    if version_id:
        segments = load_version_or_404(p.base, step, version_id)
    else:
        segments = load_latest(p.base, step)
    if segments is not None:
        return shape_step_segments(p.base, step, segments)

    raise HTTPException(404, f"No segments found for source={source}")


def _attachment_headers(
    audio_path: str, ext: str, output_dir: str | None = None
) -> dict[str, str]:
    # Force a download instead of inline rendering: the Tauri webview ignores
    # the <a download> attribute on same-origin text responses.
    from urllib.parse import quote

    stem = Path(audio_path).stem or (Path(output_dir).name if output_dir else "")
    name = f"{stem or 'export'}.{ext}"
    # Headers are Latin-1: an ASCII fallback plus the RFC 5987 UTF-8 form, or
    # a stem with accents or CJK characters fails the response with a 500.
    ascii_name = name.encode("ascii", "replace").decode("ascii").replace('"', "_")
    return {
        "Content-Disposition": (
            f'attachment; filename="{ascii_name}"; '
            f"filename*=UTF-8''{quote(name, safe='')}"
        )
    }


@router.get("/text")
def export_text(
    audio_path: str = Query(""),
    output_dir: str | None = Query(None),
    source: str = Query("transcript"),
    version_id: str | None = Query(None),
):
    """Export segments as plain text."""
    segments = _load_segments(audio_path, output_dir, source, version_id)
    text = segments_to_text(
        segments, declared=_declared_speakers(audio_path, output_dir)
    )
    return PlainTextResponse(
        text,
        media_type="text/plain; charset=utf-8",
        headers=_attachment_headers(audio_path, "txt", output_dir),
    )


@router.get("/srt")
def export_srt(
    audio_path: str = Query(""),
    output_dir: str | None = Query(None),
    source: str = Query("transcript"),
    version_id: str | None = Query(None),
):
    """Export segments as SRT subtitles."""
    segments = _load_segments(audio_path, output_dir, source, version_id)
    srt = segments_to_srt(segments, declared=_declared_speakers(audio_path, output_dir))
    return PlainTextResponse(
        srt,
        media_type="application/x-subrip; charset=utf-8",
        headers=_attachment_headers(audio_path, "srt", output_dir),
    )


@router.get("/vtt")
def export_vtt(
    audio_path: str = Query(""),
    output_dir: str | None = Query(None),
    source: str = Query("transcript"),
    version_id: str | None = Query(None),
):
    """Export segments as WebVTT subtitles."""
    segments = _load_segments(audio_path, output_dir, source, version_id)
    vtt = segments_to_vtt(segments, declared=_declared_speakers(audio_path, output_dir))
    return PlainTextResponse(
        vtt,
        media_type="text/vtt; charset=utf-8",
        headers=_attachment_headers(audio_path, "vtt", output_dir),
    )


def _write_episode_zip(audio_path: str, output_dir: str | None, target) -> None:
    """Write the episode dir as a ZIP archive into ``target`` (a path or file-like)."""
    audio_path, output_dir = _confined_paths(audio_path, output_dir)
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
    audio_path: str = Query(""),
    output_dir: str | None = Query(None),
):
    """Export the entire episode output directory as a ZIP archive."""
    buf = io.BytesIO()
    _write_episode_zip(audio_path, output_dir, buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers=_attachment_headers(audio_path, "zip", output_dir),
    )


class ExportSaveRequest(BaseModel):
    audio_path: str = ""
    output_dir: str | None = None
    source: str = "transcript"
    version_id: str | None = None
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
        # Confined like every other branch: the accepted risk is the chosen
        # destination, not reading any file on disk as the source.
        from podcodex.api.routes._helpers import resolve_inside_show_root

        if not req.audio_path:
            raise HTTPException(400, "This episode has no audio file")
        src = resolve_inside_show_root(req.audio_path)
        if src.suffix.lower() not in AUDIO_EXTENSIONS:
            raise HTTPException(400, f"Not an audio file: {src.name}")
        if not src.is_file():
            raise HTTPException(404, f"Audio file not found: {req.audio_path}")
        shutil.copyfile(src, dest)
    else:
        segments = _load_segments(
            req.audio_path, req.output_dir, req.source, req.version_id
        )
        declared = _declared_speakers(req.audio_path, req.output_dir)
        if req.format == "txt":
            content = segments_to_text(segments, declared=declared)
        elif req.format == "srt":
            content = segments_to_srt(segments, declared=declared)
        else:
            content = segments_to_vtt(segments, declared=declared)
        dest.write_text(content, encoding="utf-8")

    return {"status": "saved", "path": str(dest)}
