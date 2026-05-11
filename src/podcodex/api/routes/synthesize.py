"""Synthesis pipeline routes — voice extraction, TTS generation, assembly."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import (
    load_best_source,
    require_audio_or_output,
    submit_subprocess_task,
)
from podcodex.api.schemas import TaskResponse
from podcodex.core._ffmpeg import ffmpeg_exe
from podcodex.core._utils import AudioPaths, SAMPLE_RATE
from podcodex.core.constants import AssembleStrategy

router = APIRouter()


# ── Status ───────────────────────────────────────────────


@router.get("/status")
async def synthesis_status(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> dict:
    """Check which synthesis artifacts exist on disk."""
    from podcodex.core.versions import list_versions

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    voice_dir = p.voice_samples_dir
    tts_dir = p.tts_segments_dir
    # Synthesized if either the legacy single-file path exists OR any
    # versioned synth row is registered in the DB.
    synthesized = p.synthesized.exists() or bool(list_versions(p.base, "synthesize"))
    return {
        "voice_samples_extracted": voice_dir.is_dir() and any(voice_dir.glob("*.wav")),
        "tts_segments_generated": tts_dir.is_dir() and any(tts_dir.glob("*.wav")),
        "synthesized": synthesized,
    }


# ── Voice sample extraction ─────────────────────────────


class ExtractVoicesRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    min_duration: float | None = None
    max_duration: float | None = None
    top_k: int = 3

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        """Validate that top_k is at least 1."""
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v


@router.post("/extract-voices", response_model=TaskResponse)
async def extract_voices(req: ExtractVoicesRequest) -> TaskResponse:
    """Extract voice samples for cloning as a background task."""

    return submit_subprocess_task(
        "extract_voices",
        req.audio_path,
        entry_path="podcodex.core.synthesize_job:run_extract",
        kwargs={
            "audio_path": req.audio_path,
            "output_dir": req.output_dir,
            "min_duration": req.min_duration,
            "max_duration": req.max_duration,
            "top_k": req.top_k,
        },
        req=req,
    )


class VoiceSelection(BaseModel):
    speaker: str
    start: float
    end: float
    text: str = ""


class ExtractSelectedRequest(BaseModel):
    audio_path: str | None = None
    output_dir: str | None = None
    selections: list[VoiceSelection]


@router.post("/extract-selected")
async def extract_selected(req: ExtractSelectedRequest) -> dict:
    """Extract specific user-chosen segments as voice samples."""
    from podcodex.core.synthesize import extract_selected_samples

    if not req.audio_path:
        raise HTTPException(400, "Audio source required for extraction")
    if not req.selections:
        raise HTTPException(400, "No segments selected")

    samples = extract_selected_samples(
        req.audio_path,
        [s.model_dump() for s in req.selections],
        output_dir=req.output_dir,
    )

    # Convert Path objects for JSON
    result = {}
    for speaker, entries in samples.items():
        result[speaker] = [
            {
                "file": str(e["file"]),
                "duration": e["duration"],
                "text": e.get("text", ""),
            }
            for e in entries
        ]

    total = sum(len(v) for v in result.values())
    return {
        "status": "ok",
        "speakers": len(result),
        "total_samples": total,
        "samples": result,
    }


# ── Upload custom voice sample ─────────────────────────


@router.post("/upload-sample")
async def upload_voice_sample(
    audio_path: str | None = Form(None),
    output_dir: str | None = Form(None),
    speaker: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    """Upload an external audio file as a voice sample for a speaker."""
    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    samples_dir = p.ensure_voice_samples_dir()

    # Filename uses a uuid suffix to dodge collisions when two uploads for
    # the same speaker arrive concurrently (a sequential next_idx counter
    # raced under that pattern). The ``_custom_`` infix marks uploads so
    # ``extract_selected_samples`` won't unlink them when the user re-runs
    # extraction on the same speaker.
    import uuid

    out_path = samples_dir / f"{speaker}_custom_{uuid.uuid4().hex[:8]}.wav"

    # Save uploaded file to a temp location, then convert to 16kHz mono WAV
    import tempfile

    suffix = Path(file.filename or "upload.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                ffmpeg_exe(),
                "-y",
                "-i",
                tmp_path,
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(
            400, f"Failed to convert audio: {exc.stderr.decode()[:200]}"
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    import soundfile as sf

    info = sf.info(str(out_path))
    return {
        "file": str(out_path),
        "duration": info.duration,
        "text": f"[uploaded: {file.filename}]",
        "speaker": speaker,
    }


# ── Voice samples (load from disk) ──────────────────────


@router.get("/voice-samples")
async def get_voice_samples(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
    source_version_id: str | None = Query(None),
) -> dict[str, list[dict]]:
    """Load extracted voice samples from disk.

    When ``source_version_id`` is set, derives the speaker list from that
    pinned version (matching the source picker). Otherwise falls back to
    the latest best source — keeps backward compatibility.
    """
    from podcodex.core._utils import fill_narrator_speaker, real_speakers
    from podcodex.core.synthesize import load_voice_samples
    from podcodex.core.transcribe import load_transcript
    from podcodex.core.versions import load_version_by_id

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    segments: list[dict] | None = None
    if source_version_id:
        resolved = load_version_by_id(p.base, source_version_id)
        if resolved:
            segments, _ = resolved
    if segments is None:
        segments = load_transcript(audio_path, output_dir=output_dir) or []
    speakers = real_speakers(fill_narrator_speaker(segments))

    samples = load_voice_samples(str(p.base.parent), speakers)

    # Convert Path objects to strings for JSON serialisation
    return {
        speaker: [
            {
                "file": str(entry["file"]),
                "duration": entry["duration"],
                "text": entry.get("text", ""),
            }
            for entry in entries
        ]
        for speaker, entries in samples.items()
    }


# ── TTS generation ───────────────────────────────────────


class GenerateRequest(BaseModel):
    audio_path: str | None = None
    output_dir: str | None = None
    model_size: str = "1.7B"
    language: str = "English"
    source_lang: str | None = None
    source_version_id: str | None = None
    max_chunk_duration: float = 20.0
    force: bool = False
    only_speakers: list[str] | None = None
    # Stable keys "<speaker>:<start>:<end>" of segments the UI kept checked.
    # When provided, generation is restricted to these; everything else is
    # dropped from the output. None = keep everything.
    keep_segment_keys: list[str] | None = None

    @field_validator("max_chunk_duration")
    @classmethod
    def max_chunk_duration_positive(cls, v: float) -> float:
        """Validate that max_chunk_duration is a positive number."""
        if v <= 0:
            raise ValueError("max_chunk_duration must be positive")
        return v


@router.post("/generate", response_model=TaskResponse)
async def generate_tts(req: GenerateRequest) -> TaskResponse:
    """Generate TTS audio for all segments as a background task.

    Supports incremental generation: previously generated segments are skipped
    if their text and voice sample haven't changed.  Use ``force=true`` to
    regenerate everything, or ``only_speakers`` to target specific speakers.
    """
    require_audio_or_output(req.audio_path, req.output_dir)
    assert req.audio_path or req.output_dir
    task_key = req.audio_path or req.output_dir

    return submit_subprocess_task(
        "generate_tts",
        task_key,
        entry_path="podcodex.core.synthesize_job:run_generate",
        kwargs={
            "audio_path": req.audio_path,
            "output_dir": req.output_dir,
            "source_lang": req.source_lang or "",
            "source_version_id": req.source_version_id,
            "model_size": req.model_size,
            "language": req.language,
            "max_chunk_duration": req.max_chunk_duration,
            "force": req.force,
            "only_speakers": req.only_speakers,
            "keep_segment_keys": req.keep_segment_keys,
        },
        req=req,
    )


# ── Generated segments (load from disk) ─────────────────


@router.get("/generated-segments")
async def get_generated_segments(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
    source_version_id: str | None = Query(None),
) -> list[dict]:
    """Load generated TTS segments from disk."""
    from podcodex.core.synthesize import load_generated_segments
    from podcodex.core.versions import load_version_by_id

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    segments: list[dict] | None = None
    if source_version_id:
        resolved = load_version_by_id(p.base, source_version_id)
        if resolved:
            segments, _ = resolved
    if segments is None:
        try:
            segments = load_best_source(audio_path, output_dir)
        except ValueError:
            raise HTTPException(404, "No source segments found")

    generated = load_generated_segments(str(p.base.parent), segments)
    # Convert Path objects for JSON, include manifest metadata
    return [
        {
            "speaker": seg.get("speaker", ""),
            "text": seg.get("text", ""),
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "audio_file": str(seg["audio_file"]),
            "duration": seg.get("end", 0) - seg.get("start", 0),
            "voice_sample": seg.get("voice_sample", ""),
            "generated_at": seg.get("generated_at", ""),
        }
        for seg in generated
    ]


# ── Assembly ─────────────────────────────────────────────


class AssembleRequest(BaseModel):
    audio_path: str | None = None
    output_dir: str | None = None
    strategy: AssembleStrategy = "original_timing"
    silence_duration: float = 0.5
    language: str = ""
    model_size: str | None = None
    source_version_id: str | None = None
    # Same scope semantics as GenerateRequest. When set, drops every segment
    # whose seg_key is not in the list — so re-generating a narrower scope
    # actually shortens the assembled output instead of pulling in stale
    # on-disk segments from earlier full runs.
    keep_segment_keys: list[str] | None = None

    @field_validator("silence_duration")
    @classmethod
    def silence_duration_non_negative(cls, v: float) -> float:
        """Validate that silence_duration is zero or positive."""
        if v < 0:
            raise ValueError("silence_duration must be non-negative")
        return v


def _versioned_synth_path(base, version_id: str):
    """Versioned filename next to the legacy single-file synthesized path."""
    return base.parent / f"{base.name}.synthesized.{version_id}.wav"


@router.get("/versions")
async def list_synthesize_versions(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """List all assembled synthesis versions for an episode (newest first)."""
    from podcodex.api.routes._helpers import require_audio_or_output
    from podcodex.core.versions import backfill_version_sizes, list_versions

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    versions = list_versions(p.base, "synthesize")
    backfill_version_sizes(p.base, versions)
    return versions


@router.get("/versions/{version_id}")
async def get_synthesize_version(
    version_id: str,
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> dict:
    """Return the on-disk path + duration for a specific synth version."""
    from podcodex.api.routes._helpers import require_audio_or_output
    from podcodex.core.versions import synthesize_version_path

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    path = synthesize_version_path(p.base, version_id)
    if path is None:
        raise HTTPException(404, f"Synthesize version {version_id} not found")
    import soundfile as sf

    info = sf.info(str(path))
    return {"path": str(path), "duration": info.duration, "version_id": version_id}


@router.delete("/versions/{version_id}")
async def remove_synthesize_version(
    version_id: str,
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> dict:
    """Delete a synth version (.wav + DB row)."""
    from podcodex.api.routes._helpers import require_audio_or_output
    from podcodex.core.versions import delete_synthesize_version

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    removed = delete_synthesize_version(p.base, version_id)
    if not removed:
        raise HTTPException(404, f"Synthesize version {version_id} not found")
    return {"status": "deleted", "version_id": version_id}


@router.post("/assemble")
async def assemble(req: AssembleRequest) -> dict:
    """Assemble generated TTS segments into a versioned final episode audio file."""
    from podcodex.core.synthesize import assemble_episode, load_generated_segments
    from podcodex.core.versions import new_version_id, save_synthesize_version

    from podcodex.core._utils import seg_key

    require_audio_or_output(req.audio_path, req.output_dir)
    p = AudioPaths.from_audio(req.audio_path, output_dir=req.output_dir)

    try:
        segments = load_best_source(req.audio_path, req.output_dir)
    except ValueError:
        raise HTTPException(404, "No source segments found")

    if req.keep_segment_keys is not None:
        wanted = set(req.keep_segment_keys)
        segments = [seg for seg in segments if seg_key(seg) in wanted]
        if not segments:
            raise HTTPException(400, "Selection dropped every segment")

    generated = load_generated_segments(str(p.base.parent), segments)
    if not generated:
        raise HTTPException(404, "No generated TTS segments found")

    now, version_id = new_version_id("raw")
    versioned_path = _versioned_synth_path(p.base, version_id)

    try:
        out_path = assemble_episode(
            generated,
            req.audio_path,
            output_dir=req.output_dir,
            strategy=req.strategy,
            silence_duration=req.silence_duration,
            output_path=versioned_path,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    import soundfile as sf

    info = sf.info(str(out_path))
    save_synthesize_version(
        p.base,
        out_path,
        version_id=version_id,
        now=now,
        strategy=req.strategy,
        silence_duration=req.silence_duration,
        source_version_id=req.source_version_id,
        language=req.language,
        model_size=req.model_size,
        segment_count=len(generated),
        duration_s=info.duration,
    )
    # Flip the per-show pipeline DB flag. The /unified episodes route reads
    # synthesized from this table (not from disk on every request), so without
    # an explicit mark here the overview StageCard stays stuck on "not started"
    # forever after the first populate_from_scan.
    from podcodex.core.pipeline_db import get_pipeline_db

    show_folder = p.show_dir
    get_pipeline_db(show_folder).mark(p.base.name, synthesized=True)

    # Drop folder scan cache so any disk-based fallback also reflects the
    # fresh state on the next read.
    from podcodex.ingest.folder import invalidate_scan_cache

    invalidate_scan_cache(show_folder)
    return {
        "path": str(out_path),
        "duration": info.duration,
        "version_id": version_id,
    }
