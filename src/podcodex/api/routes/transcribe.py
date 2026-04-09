"""Transcription routes — load/save transcripts, speaker maps, run pipeline."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import (
    build_provenance,
    submit_task,
    transcribe_prov_params,
)
from podcodex.api.schemas import Segment, TaskResponse
from podcodex.api.routes._versions import register_version_routes
from podcodex.core._utils import AudioPaths
from podcodex.core.pipeline_db import mark_step
from podcodex.core.versions import save_version

router = APIRouter()
register_version_routes(router, "transcript")


# ── Load / save ──────────────────────────────────────────


@router.get("/segments")
async def get_segments(
    audio_path: str = Query(..., description="Absolute path to audio file"),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load transcript segments (latest version, falls back to legacy files)."""
    from podcodex.api.routes._helpers import annotate_flags
    from podcodex.core.versions import load_latest

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    segments = load_latest(p.base, "transcript")
    if segments is None:
        from podcodex.api.routes._helpers import read_segments

        segments = read_segments(p.transcript_best)
    if segments is None:
        raise HTTPException(404, "No transcript found")
    return annotate_flags(segments)


@router.put("/segments")
async def save_segments(
    segments: list[Segment],
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Save validated transcript segments."""
    from podcodex.core.transcribe import save_transcript

    seg_dicts = [s.model_dump() for s in segments]
    provenance = build_provenance("transcript", ptype="validated", manual_edit=True)
    save_transcript(
        audio_path,
        seg_dicts,
        output_dir=output_dir,
        provenance=provenance,
    )
    return {"status": "saved", "count": len(seg_dicts)}


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
    """Upload a transcript file (JSON, SRT, or VTT) and save as raw transcript."""
    from podcodex.core._utils import srt_to_segments, vtt_to_segments

    content = await file.read()
    filename = (file.filename or "").lower()

    if filename.endswith(".vtt"):
        text = content.decode("utf-8")
        segments = vtt_to_segments(text)
        if not segments:
            raise HTTPException(400, "No segments found in VTT file")
        source = "vtt"
        original_text = text
    elif filename.endswith(".srt"):
        text = content.decode("utf-8")
        segments = srt_to_segments(text)
        if not segments:
            raise HTTPException(400, "No segments found in SRT file")
        source = "srt"
        original_text = text
    else:
        original_text = None
        # JSON (default)
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
        source = "json"

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    p.base.parent.mkdir(parents=True, exist_ok=True)

    # Save original subtitle file for reference
    if original_text is not None:
        ext = "vtt" if source == "vtt" else "srt"
        orig_path = p.base / f"{p.base.name}.subtitles.{ext}"
        orig_path.write_text(original_text, encoding="utf-8")

    provenance = build_provenance(
        "transcript",
        params=transcribe_prov_params(
            diarize=False, source=source, filename=file.filename
        ),
    )
    save_version(p.base, "transcript", segments, provenance)
    mark_step(
        p.show_dir, p.base.name, transcribed=True, provenance={"transcript": provenance}
    )

    return {"status": "uploaded", "count": len(segments)}


# ── Import from existing file ────────────────────────────


@router.post("/import")
async def import_transcript(
    audio_path: str = Query(..., description="Absolute path to audio file"),
    file_path: str = Query(
        ..., description="Absolute path to VTT/SRT/JSON file to import"
    ),
    output_dir: str | None = Query(None),
) -> dict:
    """Import a transcript from an existing file on disk (VTT, SRT, or JSON)."""
    from pathlib import Path

    from podcodex.core._utils import srt_to_segments, vtt_to_segments

    src = Path(file_path)
    if not src.exists():
        raise HTTPException(404, f"File not found: {file_path}")

    content = src.read_text(encoding="utf-8")
    filename = src.name.lower()

    if filename.endswith(".vtt"):
        segments = vtt_to_segments(content)
        if not segments:
            raise HTTPException(400, "No segments found in VTT file")
        source = "vtt"
    elif filename.endswith(".srt"):
        segments = srt_to_segments(content)
        if not segments:
            raise HTTPException(400, "No segments found in SRT file")
        source = "srt"
    elif filename.endswith(".json"):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid JSON file")
        if isinstance(data, dict) and "segments" in data:
            segments = data["segments"]
        elif isinstance(data, list):
            segments = data
        else:
            raise HTTPException(
                400, "Expected a JSON array of segments or {segments: [...]}"
            )
        source = "json"
    else:
        raise HTTPException(400, "Unsupported file type. Use VTT, SRT, or JSON.")

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    p.base.parent.mkdir(parents=True, exist_ok=True)

    provenance = build_provenance(
        "transcript",
        params=transcribe_prov_params(diarize=False, source=source, filename=src.name),
    )
    save_version(p.base, "transcript", segments, provenance)
    mark_step(
        p.show_dir, p.base.name, transcribed=True, provenance={"transcript": provenance}
    )

    return {"status": "imported", "count": len(segments)}


# ── Pipeline execution ───────────────────────────────────


class TranscribeRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    model_size: str = "large-v3-turbo"
    language: str = ""
    batch_size: int | None = None
    force: bool = False
    diarize: bool = True
    hf_token: str | None = None
    num_speakers: int | None = None
    show: str = ""
    episode: str = ""
    clean: bool = False

    @field_validator("batch_size")
    @classmethod
    def batch_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be at least 1")
        return v

    @field_validator("num_speakers")
    @classmethod
    def num_speakers_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("num_speakers must be at least 1")
        return v


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

        from podcodex.core._utils import default_batch_size

        batch_size = req_data.batch_size or default_batch_size()
        progress_cb(0.0, f"Transcribing audio (batch_size={batch_size})...")
        transcribe_file(
            req_data.audio_path,
            model_size=req_data.model_size,
            language=req_data.language or None,
            batch_size=batch_size,
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
        provenance = build_provenance(
            "transcript",
            model=req_data.model_size,
            params=transcribe_prov_params(
                req_data.diarize,
                source="whisper",
                model=req_data.model_size,
                language=req_data.language or None,
                batch_size=req_data.batch_size,
                num_speakers=req_data.num_speakers,
                clean=req_data.clean,
            ),
        )
        segments = export_transcript(
            req_data.audio_path,
            output_dir=req_data.output_dir,
            show=req_data.show,
            episode=req_data.episode,
            diarized=req_data.diarize,
            clean=req_data.clean,
            provenance=provenance,
        )
        return {"count": len(segments)}

    return submit_task("transcribe", req.audio_path, run_transcribe, req)
