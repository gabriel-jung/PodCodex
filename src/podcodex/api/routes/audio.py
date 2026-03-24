"""Audio file serving — full files and segment clips."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/api/audio/file")
async def serve_audio_file(
    path: str = Query(..., description="Absolute path to audio file"),
):
    """Serve a full audio file with range-request support."""
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, f"Audio file not found: {path}")

    media_type = {
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }.get(p.suffix.lower(), "application/octet-stream")

    return FileResponse(p, media_type=media_type)


@router.get("/api/audio/clip")
async def serve_audio_clip(
    path: str = Query(...),
    start: float = Query(0.0),
    end: float = Query(0.0),
    padding: float = Query(0.3),
):
    """Serve a short audio clip extracted from a full file.

    Requires soundfile to be installed. Falls back to full file if not available.
    """
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, f"Audio file not found: {path}")

    try:
        import io

        import soundfile as sf

        info = sf.info(str(p))
        sr = info.samplerate

        clip_start = max(0, start - padding)
        clip_end = min(info.duration, end + padding) if end > 0 else info.duration

        start_frame = int(clip_start * sr)
        n_frames = int((clip_end - clip_start) * sr)

        data, _ = sf.read(str(p), start=start_frame, frames=n_frames, dtype="float32")

        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV")
        buf.seek(0)

        from fastapi.responses import StreamingResponse

        return StreamingResponse(buf, media_type="audio/wav")
    except ImportError:
        # soundfile not installed — serve full file
        return FileResponse(p, media_type="audio/mpeg")
