"""Audio file serving — full files and segment clips."""

from __future__ import annotations


from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger

from podcodex.api.routes._helpers import AUDIO_EXTS, resolve_inside_show_root

router = APIRouter()


@router.get("/file")
def serve_audio_file(
    path: str = Query(..., description="Absolute path to audio file"),
):
    """Serve a full audio file with range-request support."""
    p = resolve_inside_show_root(path)
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


@router.get("/clip")
def serve_audio_clip(
    path: str = Query(...),
    start: float = Query(0.0),
    end: float = Query(0.0),
    padding: float = Query(0.3),
):
    """Serve a short audio clip extracted from a full file.

    Requires soundfile to be installed. Falls back to full file if not available.
    """
    p = resolve_inside_show_root(path)
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


@router.delete("/file")
def delete_audio_file(
    path: str = Query(..., description="Absolute path to audio file"),
):
    """Delete an audio file from disk."""
    p = resolve_inside_show_root(path)
    if not p.is_file():
        raise HTTPException(404, f"Audio file not found: {path}")

    if p.suffix.lower() not in AUDIO_EXTS:
        raise HTTPException(400, f"Not an audio file: {p.name}")

    show_folder = p.parent
    stem = p.stem
    p.unlink()

    from podcodex.ingest.folder import invalidate_scan_cache

    invalidate_scan_cache(show_folder)

    # Unlinking the audio can leave a status row with nothing behind it: no
    # output dir, no versions, no audio. /unified lists episodes straight from
    # those rows, so the episode would stay on screen forever with every step
    # showing "not started" and no way to act on it. Drop the row in that case
    # only; when transcripts remain the episode is still real.
    #
    # Gated on the parent actually being a registered show root. The path guard
    # above accepts anything *under* one, and .wav is an audio extension, so a
    # synthesized `{show}/{stem}/synthesize/{id}.wav` reaches here too. Opening
    # a pipeline DB against that directory would create a stray pipeline.db in
    # it, because PipelineDB mkdirs and connects.
    row_removed = False
    if _is_show_root(show_folder):
        from podcodex.core.delete_episode import drop_db_row, episode_has_leftovers

        try:
            if not episode_has_leftovers(show_folder, stem):
                row_removed = drop_db_row(show_folder, stem)
        except Exception:
            # Best-effort cleanup: the audio really was deleted, so the request
            # succeeded. A surviving row self-heals on the next resync.
            logger.opt(exception=True).warning(
                "audio delete: orphan-row cleanup failed for {!r}", stem
            )

    return {"status": "deleted", "path": str(p), "episode_removed": row_removed}


def _is_show_root(path: Path) -> bool:
    """True when ``path`` is itself a registered show folder, not merely inside one."""
    from podcodex.api.routes.config import _load as _load_cfg

    for folder in _load_cfg().show_folders:
        try:
            if path.samefile(Path(folder)):
                return True
        except OSError:
            continue
    return False
