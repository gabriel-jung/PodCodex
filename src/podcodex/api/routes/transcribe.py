"""Transcription routes — load/save transcripts, speaker maps, run pipeline."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel

from podcodex.api.routes._helpers import annotate_flags, read_segments, submit_task
from podcodex.api.schemas import Segment, TaskResponse
from podcodex.core._utils import AudioPaths, merge_consecutive_segments, write_json

router = APIRouter()


# ── Load / save ──────────────────────────────────────────


@router.get("/segments")
async def get_segments(
    audio_path: str = Query(..., description="Absolute path to audio file"),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load transcript segments (prefers validated over raw)."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    data = read_segments(p.transcript_best)
    if data is None:
        raise HTTPException(404, "No transcript found")
    return annotate_flags(data)


@router.get("/segments/raw")
async def get_segments_raw(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load raw (unvalidated) transcript segments."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    data = read_segments(p.transcript_raw)
    if data is None:
        raise HTTPException(404, "No raw transcript found")
    return annotate_flags(data)


@router.put("/segments")
async def save_segments(
    segments: list[Segment],
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Save validated transcript segments."""
    from podcodex.core.transcribe import save_transcript

    seg_dicts = [s.model_dump() for s in segments]
    save_transcript(audio_path, seg_dicts, output_dir=output_dir)
    return {"status": "saved", "count": len(seg_dicts)}


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


# ── Speaker map ──────────────────────────────────────────


@router.get("/speaker-map")
async def get_speaker_map(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict[str, str]:
    """Read the speaker name mapping."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    if not p.speaker_map.exists():
        return {}
    try:
        return json.loads(p.speaker_map.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@router.put("/speaker-map")
async def save_speaker_map(
    mapping: dict[str, str],
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Save the speaker name mapping."""
    from podcodex.core.transcribe import save_speaker_map

    save_speaker_map(audio_path, mapping, output_dir=output_dir)
    return {"status": "saved"}


# ── Upload ────────────────────────────────────────────────


@router.post("/upload")
async def upload_transcript(
    file: UploadFile = File(...),
    audio_path: str = Query(..., description="Absolute path to audio file"),
    output_dir: str | None = Query(None),
) -> dict:
    """Upload a transcript JSON file and save as raw transcript."""
    content = await file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON file")

    # Accept either a plain list of segments or {segments: [...], ...} wrapper
    if isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
    elif isinstance(data, list):
        segments = data
    else:
        raise HTTPException(
            400, "Expected a JSON array of segments or {segments: [...]}"
        )

    if not isinstance(segments, list):
        raise HTTPException(400, "Segments must be a JSON array")

    # Validate each segment has at least text and speaker
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise HTTPException(400, f"Segment {i} is not an object")
        if "text" not in seg:
            raise HTTPException(400, f"Segment {i} missing 'text' field")
        # Ensure required fields have defaults
        seg.setdefault("speaker", "UNKNOWN")
        seg.setdefault("start", 0.0)
        seg.setdefault("end", 0.0)

    # Merge consecutive same-speaker segments (same as the transcription pipeline)
    segments = merge_consecutive_segments(segments)

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    p.base.parent.mkdir(parents=True, exist_ok=True)
    write_json(p.transcript_raw, {"segments": segments})
    return {"status": "uploaded", "count": len(segments)}


# ── Pipeline execution ───────────────────────────────────


class TranscribeRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    model_size: str = "large-v3-turbo"
    language: str = ""
    batch_size: int = 16
    force: bool = False
    diarize: bool = True
    hf_token: str | None = None
    num_speakers: int | None = None
    show: str = ""
    episode: str = ""


@router.post("/start", response_model=TaskResponse)
async def start_transcribe(req: TranscribeRequest) -> TaskResponse:
    """Start the transcription pipeline as a background task."""

    def run_transcribe(progress_cb, req_data):
        from podcodex.core.transcribe import (
            assign_speakers,
            diarize_file,
            export_transcript,
            transcribe_file,
        )

        progress_cb(0.0, "Transcribing audio...")
        transcribe_file(
            req_data.audio_path,
            model_size=req_data.model_size,
            language=req_data.language or None,
            batch_size=req_data.batch_size,
            force=req_data.force,
            output_dir=req_data.output_dir,
        )

        if req_data.diarize:
            progress_cb(0.25, "Diarizing speakers...")
            diarize_file(
                req_data.audio_path,
                hf_token=req_data.hf_token,
                num_speakers=req_data.num_speakers,
                force=req_data.force,
                output_dir=req_data.output_dir,
            )

            progress_cb(0.5, "Assigning speakers...")
            assign_speakers(
                req_data.audio_path,
                force=req_data.force,
                output_dir=req_data.output_dir,
            )

        progress_cb(0.75, "Exporting transcript...")
        segments = export_transcript(
            req_data.audio_path,
            output_dir=req_data.output_dir,
            show=req_data.show,
            episode=req_data.episode,
            diarized=req_data.diarize,
        )
        return {"count": len(segments)}

    return submit_task("transcribe", req.audio_path, run_transcribe, req)
