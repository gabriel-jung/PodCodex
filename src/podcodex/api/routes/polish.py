"""Polish pipeline routes — load/save polished segments, run polish."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from podcodex.api.routes._helpers import (
    ApplyManualRequest,
    LLMRequest,
    ManualPromptsRequest,
    batch_progress,
    build_provenance,
    submit_task,
)
from podcodex.api.routes._versions import register_version_routes
from podcodex.api.schemas import Segment, TaskResponse
from podcodex.core._utils import AudioPaths

router = APIRouter()
register_version_routes(router, "polished")


# ── Load / save ──────────────────────────────────────────


@router.get("/segments")
async def get_polished_segments(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load polished segments from the version DB."""
    from podcodex.api.routes._helpers import annotate_flags
    from podcodex.core.versions import load_latest

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    segments = load_latest(p.base, "polished")
    if segments is None:
        raise HTTPException(404, "No polished segments found")
    return annotate_flags(segments)


@router.put("/segments")
async def save_polished_segments(
    segments: list[Segment],
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Save validated polished segments."""
    from podcodex.core.polish import save_polished

    seg_dicts = [s.model_dump() for s in segments]
    provenance = build_provenance("polished", ptype="validated", manual_edit=True)
    save_polished(audio_path, seg_dicts, output_dir=output_dir, provenance=provenance)
    return {"status": "saved", "count": len(seg_dicts)}


# ── Pipeline execution ───────────────────────────────────


class PolishRequest(LLMRequest):
    engine: str = "Whisper"


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
            on_batch=batch_progress(progress_cb),
        )

        progress_cb(0.95, "Saving...")
        provenance = build_provenance(
            "polished",
            model=req_data.model or "qwen3:4b",
            params={
                "mode": req_data.mode,
                "provider": req_data.provider,
                "source_lang": req_data.source_lang,
                "batch_minutes": req_data.batch_minutes,
                "engine": req_data.engine,
            },
        )
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

    provenance = build_provenance("polished", params={"skipped": True})
    save_polished_raw(
        req.audio_path,
        segments,
        output_dir=req.output_dir,
        provenance=provenance,
    )
    return {"status": "saved", "count": len(segments)}


# ── Manual mode ──────────────────────────────────────────


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
    provenance = build_provenance(
        "polished",
        params={"mode": "manual"},
        manual_edit=True,
    )
    save_polished_raw(
        req.audio_path,
        polished,
        output_dir=req.output_dir,
        provenance=provenance,
    )
    return {"status": "saved", "count": len(polished)}
