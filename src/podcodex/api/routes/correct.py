"""Correct pipeline routes — load/save corrected segments, run correction."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from podcodex.api.routes._helpers import (
    ApplyManualRequest,
    LLMRequest,
    ManualPromptsRequest,
    batch_progress,
    build_edit_provenance,
    build_provenance,
    enrich_correct_kwargs,
    format_prompt_batches,
    llm_prov_params,
    require_audio_or_output,
    submit_task,
)
from podcodex.api.routes._versions import register_version_routes
from podcodex.api.schemas import Segment, TaskResponse
from podcodex.core._utils import AudioPaths

router = APIRouter()
register_version_routes(router, "corrected")


# ── Load / save ──────────────────────────────────────────


@router.get("/segments")
async def get_corrected_segments(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
    limit: int | None = Query(None, ge=1, description="Max segments to return"),
) -> list[dict]:
    """Load corrected segments from the version DB."""
    from podcodex.api.routes._helpers import annotate_flags
    from podcodex.core.versions import load_latest

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    segments = load_latest(p.base, "corrected")
    if segments is None:
        raise HTTPException(404, "No corrected segments found")
    if limit is not None:
        segments = segments[:limit]
    return annotate_flags(segments)


@router.put("/segments")
async def save_corrected_segments(
    segments: list[Segment],
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> dict:
    """Save validated corrected segments."""
    from podcodex.core.correct import save_corrected

    require_audio_or_output(audio_path, output_dir)
    seg_dicts = [s.model_dump() for s in segments]
    provenance = build_edit_provenance("corrected", audio_path, output_dir)
    save_corrected(audio_path, seg_dicts, output_dir=output_dir, provenance=provenance)
    return {"status": "saved", "count": len(seg_dicts)}


# ── Pipeline execution ───────────────────────────────────


@router.post("/start", response_model=TaskResponse)
async def start_correct(req: LLMRequest) -> TaskResponse:
    """Start the correct pipeline as a background task."""
    # Resolve profile + key up front so a bad pick fails the request, not
    # the background task. Ollama mode tolerates an empty key_name.
    if req.mode == "api":
        from podcodex.core.llm_resolver import LLMResolutionError, resolve_llm

        try:
            resolved = resolve_llm(req.provider_profile, req.key_name)
        except LLMResolutionError as exc:
            raise HTTPException(400, str(exc))
    else:
        resolved = None

    def run_correct(progress_cb, req_data):
        from podcodex.core.correct import correct_segments, save_corrected
        from podcodex.core.transcribe import load_transcript
        from podcodex.core.versions import load_version

        progress_cb(0.0, "Loading transcript...")
        if req_data.source_version_id:
            p = AudioPaths.from_audio(
                req_data.audio_path, output_dir=req_data.output_dir
            )
            segments = load_version(p.base, "transcript", req_data.source_version_id)
        else:
            segments = load_transcript(
                req_data.audio_path, output_dir=req_data.output_dir
            )
        if not segments:
            raise ValueError("No transcript found to correct")

        # Auto-detect transcript source and language from provenance
        tc_kwargs = enrich_correct_kwargs(
            req_data.audio_path, req_data.output_dir, req_data.source_lang
        )

        progress_cb(0.1, "Starting correction...")

        corrected = correct_segments(
            segments,
            mode=req_data.mode,
            context=req_data.context,
            source_lang=tc_kwargs["source_lang"],
            model=req_data.model,
            batch_minutes=req_data.batch_minutes,
            provider=resolved.provider if resolved else None,
            engine=tc_kwargs["engine"],
            engine_model=tc_kwargs["engine_model"],
            api_base_url=resolved.api_base_url if resolved else "",
            api_key=resolved.api_key if resolved else None,
            original_segments=segments,
            merge=False,  # transcript is already merged on load/upload
            on_batch=batch_progress(progress_cb),
        )

        progress_cb(0.95, "Saving...")
        provenance = build_provenance(
            "corrected",
            model=req_data.model,
            audio_path=req_data.audio_path,
            output_dir=req_data.output_dir,
            params=llm_prov_params(
                req_data.mode,
                provider_profile=req_data.provider_profile,
                key_name=req_data.key_name,
                source_lang=tc_kwargs["source_lang"],
                batch_minutes=req_data.batch_minutes,
            ),
        )
        save_corrected(
            req_data.audio_path,
            corrected,
            output_dir=req_data.output_dir,
            provenance=provenance,
        )
        return {"count": len(corrected)}

    return submit_task("correct", req.audio_path, run_correct, req)


# ── Manual mode ──────────────────────────────────────────


@router.post("/manual-prompts")
async def generate_manual_prompts(req: ManualPromptsRequest) -> list[dict]:
    """Generate batched prompts for manual LLM correction."""
    from podcodex.core.correct import (
        build_manual_prompts_batched,
        transcript_provenance_info,
    )
    from podcodex.core.transcribe import load_transcript
    from podcodex.core.versions import get_latest_provenance, load_version

    p = AudioPaths.from_audio(req.audio_path, output_dir=req.output_dir)

    if req.source_version_id:
        segments = load_version(p.base, "transcript", req.source_version_id)
    else:
        segments = load_transcript(req.audio_path, output_dir=req.output_dir)
    if not segments:
        raise HTTPException(404, "No transcript found")

    tc_info = transcript_provenance_info(get_latest_provenance(p.base, "transcript"))
    source_lang = tc_info["language"] or req.source_lang

    batches = build_manual_prompts_batched(
        segments,
        batch_minutes=req.batch_minutes,
        context=req.context,
        source_lang=source_lang,
        engine=tc_info["source"],
        engine_model=tc_info["model"],
    )
    return format_prompt_batches(batches)


@router.post("/apply-manual")
async def apply_manual_corrections(req: ApplyManualRequest) -> dict:
    """Apply manually-obtained LLM corrections and save as raw."""
    from podcodex.core._utils import validate_manual
    from podcodex.core.correct import save_corrected
    from podcodex.core.transcribe import load_transcript

    original = load_transcript(req.audio_path, output_dir=req.output_dir)
    if not original:
        raise HTTPException(404, "No transcript found")

    corrected = validate_manual(req.corrections, original)
    # Applying manual LLM prompts is still an LLM correction, not a hand-edit —
    # manual_edit stays False so the review workflow can distinguish an
    # unreviewed LLM pass from a user-validated one.
    provenance = build_provenance(
        "corrected",
        params=llm_prov_params("manual"),
        audio_path=req.audio_path,
        output_dir=req.output_dir,
    )
    save_corrected(
        req.audio_path,
        corrected,
        output_dir=req.output_dir,
        provenance=provenance,
    )
    return {"status": "saved", "count": len(corrected)}
