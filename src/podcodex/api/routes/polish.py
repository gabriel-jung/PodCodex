"""Polish pipeline routes — load/save polished segments, run polish."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import (
    load_segments_or_404,
    submit_task,
)
from podcodex.api.schemas import Segment, TaskResponse
from podcodex.core._utils import AudioPaths

router = APIRouter()


# ── Load / save ──────────────────────────────────────────


@router.get("/segments")
async def get_polished_segments(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load polished segments (latest version, falls back to legacy files)."""
    from podcodex.api.routes._helpers import annotate_flags
    from podcodex.core.versions import load_latest

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    segments = load_latest(p.base, "polished")
    if segments is not None:
        return annotate_flags(segments)
    return load_segments_or_404(p.polished_best, "polished segments")


@router.put("/segments")
async def save_polished_segments(
    segments: list[Segment],
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Save validated polished segments."""
    from podcodex.core.polish import save_polished

    seg_dicts = [s.model_dump() for s in segments]
    provenance = {
        "step": "polished",
        "type": "validated",
        "model": None,
        "params": {},
        "manual_edit": True,
    }
    save_polished(audio_path, seg_dicts, output_dir=output_dir, provenance=provenance)
    return {"status": "saved", "count": len(seg_dicts)}


# ── Version history ──────────────────────────────────────


@router.get("/versions")
async def list_polish_versions(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """List all archived polished versions (newest first)."""
    from podcodex.core.versions import list_versions

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return list_versions(p.base, "polished")


@router.get("/versions/{version_id}")
async def load_polish_version(
    version_id: str,
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load segments from a specific archived polished version."""
    from podcodex.core.versions import load_version

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    try:
        return load_version(p.base, "polished", version_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Version {version_id} not found")


# ── Pipeline execution ───────────────────────────────────


class PolishRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    mode: str = "ollama"
    provider: str | None = None
    model: str = ""
    context: str = ""
    source_lang: str = "French"
    batch_minutes: float = 15.0
    engine: str = "Whisper"
    api_base_url: str = ""
    api_key: str | None = None

    @field_validator("batch_minutes")
    @classmethod
    def batch_minutes_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("batch_minutes must be positive")
        return v


@router.post("/start", response_model=TaskResponse)
async def start_polish(req: PolishRequest) -> TaskResponse:
    """Start the polish pipeline as a background task."""

    def run_polish(progress_cb, req_data):
        from podcodex.core.polish import polish_segments, save_polished_raw
        from podcodex.core.transcribe import load_transcript

        progress_cb(0.0, "Loading transcript...")
        segments = load_transcript(req_data.audio_path, output_dir=req_data.output_dir)
        if not segments:
            raise ValueError("No transcript found to polish")

        progress_cb(0.1, "Starting polish...")

        def on_batch(batch_num, total):
            frac = 0.1 + 0.8 * (batch_num / total)
            progress_cb(frac, f"Batch {batch_num} of {total}")

        polished = polish_segments(
            segments,
            mode=req_data.mode,
            context=req_data.context,
            source_lang=req_data.source_lang,
            model=req_data.model,
            batch_minutes=req_data.batch_minutes,
            provider=req_data.provider,
            engine=req_data.engine,
            api_base_url=req_data.api_base_url,
            api_key=req_data.api_key,
            original_segments=segments,
            merge=False,  # transcript is already merged on load/upload
            on_batch=on_batch,
        )

        progress_cb(0.95, "Saving...")
        provenance = {
            "step": "polished",
            "type": "raw",
            "model": req_data.model or "qwen3:4b",
            "params": {
                "mode": req_data.mode,
                "provider": req_data.provider,
                "source_lang": req_data.source_lang,
                "batch_minutes": req_data.batch_minutes,
                "engine": req_data.engine,
            },
            "manual_edit": False,
        }
        save_polished_raw(
            req_data.audio_path,
            polished,
            output_dir=req_data.output_dir,
            provenance=provenance,
        )
        return {"count": len(polished)}

    return submit_task("polish", req.audio_path, run_polish, req)


# ── Skip polish ──────────────────────────────────────────


class SkipRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None


@router.post("/skip")
async def skip_polish(req: SkipRequest) -> dict:
    """Copy transcript segments directly to polished output (skip LLM)."""
    from podcodex.core.polish import save_polished_raw
    from podcodex.core.transcribe import load_transcript

    segments = load_transcript(req.audio_path, output_dir=req.output_dir)
    if not segments:
        raise HTTPException(404, "No transcript found to copy")

    provenance = {
        "step": "polished",
        "type": "raw",
        "model": None,
        "params": {"skipped": True},
        "manual_edit": False,
    }
    save_polished_raw(
        req.audio_path,
        segments,
        output_dir=req.output_dir,
        provenance=provenance,
    )
    return {"status": "saved", "count": len(segments)}


# ── Manual mode ──────────────────────────────────────────


class ManualPromptsRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    context: str = ""
    source_lang: str = "French"
    batch_minutes: float = 15.0
    engine: str = "Whisper"


@router.post("/manual-prompts")
async def generate_manual_prompts(req: ManualPromptsRequest) -> list[dict]:
    """Generate batched prompts for manual LLM correction."""
    from podcodex.core.polish import build_manual_prompts_batched
    from podcodex.core.transcribe import load_transcript

    segments = load_transcript(req.audio_path, output_dir=req.output_dir)
    if not segments:
        raise HTTPException(404, "No transcript found")

    batches = build_manual_prompts_batched(
        segments,
        batch_minutes=req.batch_minutes,
        context=req.context,
        source_lang=req.source_lang,
        engine=req.engine,
    )
    return [
        {"batch_index": i, "prompt": prompt, "segment_count": len(batch_segs)}
        for i, (batch_segs, prompt) in enumerate(batches)
    ]


class ApplyManualRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    corrections: list[dict]


@router.post("/apply-manual")
async def apply_manual_corrections(req: ApplyManualRequest) -> dict:
    """Apply manually-obtained LLM corrections and save as raw."""
    from podcodex.core._utils import validate_manual
    from podcodex.core.polish import save_polished_raw
    from podcodex.core.transcribe import load_transcript

    original = load_transcript(req.audio_path, output_dir=req.output_dir)
    if not original:
        raise HTTPException(404, "No transcript found")

    polished = validate_manual(req.corrections, original)
    provenance = {
        "step": "polished",
        "type": "raw",
        "model": None,
        "params": {"mode": "manual"},
        "manual_edit": True,
    }
    save_polished_raw(
        req.audio_path,
        polished,
        output_dir=req.output_dir,
        provenance=provenance,
    )
    return {"status": "saved", "count": len(polished)}
