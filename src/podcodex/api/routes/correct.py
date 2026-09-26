"""Correct pipeline routes — load/save corrected segments, run correction."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from podcodex.api.routes._helpers import (
    ApplyBatchesRequest,
    ApplyManualRequest,
    LLMRequest,
    ManualPromptsRequest,
    batch_progress,
    build_edit_provenance,
    build_provenance,
    enrich_correct_kwargs,
    format_prompt_batches,
    llm_prov_params,
    reconcile_batches,
    require_audio_or_output,
    submit_task,
)
from podcodex.core.llm_failures import clear_step_for, get_step, stamp_run_version
from podcodex.api.routes._versions import register_version_routes
from podcodex.api.schemas import Segment, TaskResponse
from podcodex.core._utils import AudioPaths
from podcodex.core.source import load_source

router = APIRouter()
register_version_routes(router, "corrected")


# ── Load / save ──────────────────────────────────────────


@router.get("/segments")
def get_corrected_segments(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
    limit: int | None = Query(None, ge=1, description="Max segments to return"),
) -> list[dict]:
    """Load corrected segments from the version DB."""
    from podcodex.api.routes._helpers import shape_step_segments
    from podcodex.core.versions import load_latest

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    segments = load_latest(p.base, "corrected")
    if segments is None:
        raise HTTPException(404, "No corrected segments found")
    if limit is not None:
        segments = segments[:limit]
    return shape_step_segments(p.base, "corrected", segments)


@router.put("/segments")
def save_corrected_segments(
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
def start_correct(req: LLMRequest) -> TaskResponse:
    """Start the correct pipeline as a background task."""
    # Resolve profile + key up front so a bad pick fails the request, not
    # the background task. Ollama mode tolerates an empty key_name.
    from podcodex.core.llm_resolver import LLMResolutionError, resolve_llm_run

    try:
        llm = resolve_llm_run(req.mode, req.provider_profile, req.key_name, req.model)
    except LLMResolutionError as exc:
        raise HTTPException(400, str(exc))

    def run_correct(progress_cb, req_data):
        from podcodex.core.correct import correct_and_save

        progress_cb(0.0, "Loading transcript...")
        source = load_source(
            req_data.audio_path,
            req_data.output_dir,
            req_data.source_version_id,
            step="transcript",
        )
        progress_cb(0.1, "Starting correction...")
        corrected, _version_id = correct_and_save(
            source,
            llm,
            audio_path=req_data.audio_path,
            output_dir=req_data.output_dir,
            context=req_data.context,
            source_lang=req_data.source_lang,
            batch_minutes=req_data.batch_minutes,
            provider_profile=req_data.provider_profile,
            key_name=req_data.key_name,
            on_batch=batch_progress(progress_cb),
        )
        return {"count": len(corrected)}

    return submit_task("correct", req.audio_path, run_correct, req)


# ── Manual mode ──────────────────────────────────────────


@router.post("/manual-prompts")
def generate_manual_prompts(req: ManualPromptsRequest) -> list[dict]:
    """Generate batched prompts for manual LLM correction."""
    from podcodex.core.correct import build_manual_prompts_batched

    try:
        source = load_source(
            req.audio_path, req.output_dir, req.source_version_id, step="transcript"
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    tc_kwargs = enrich_correct_kwargs(
        req.audio_path, req.output_dir, req.source_lang, source
    )

    batches = build_manual_prompts_batched(
        source.segments,
        batch_minutes=req.batch_minutes,
        batch_count=req.batch_count,
        context=req.context,
        **tc_kwargs,
    )
    return format_prompt_batches(batches)


@router.get("/llm-failures")
def get_correct_failures(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> dict | None:
    """Per-batch records of the last auto correction run, or None if none."""
    require_audio_or_output(audio_path, output_dir)
    return get_step(audio_path, output_dir, "corrected")


@router.delete("/llm-failures")
def dismiss_correct_failures(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
) -> dict:
    """Dismiss the recorded correction batch results for this episode."""
    require_audio_or_output(audio_path, output_dir)
    return {"cleared": clear_step_for(audio_path, output_dir, "corrected")}


@router.post("/apply-manual")
def apply_manual_corrections(req: ApplyManualRequest) -> dict:
    """Apply manually-obtained LLM corrections and save as raw."""
    from podcodex.core.llm import validate_manual
    from podcodex.core.correct import save_corrected

    try:
        source = load_source(
            req.audio_path, req.output_dir, req.source_version_id, step="transcript"
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))

    try:
        corrected = validate_manual(req.corrections, source.segments)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    # Applying manual LLM prompts is still an LLM correction, not a hand-edit —
    # manual_edit stays False so the review workflow can distinguish an
    # unreviewed LLM pass from a user-validated one.
    provenance = build_provenance(
        "corrected",
        params=llm_prov_params("manual"),
        audio_path=req.audio_path,
        output_dir=req.output_dir,
        source=source,
    )
    save_corrected(
        req.audio_path,
        corrected,
        output_dir=req.output_dir,
        provenance=provenance,
    )
    return {"status": "saved", "count": len(corrected)}


@router.post("/apply-batches")
def apply_batches_correction(req: ApplyBatchesRequest) -> dict:
    """Apply hand-reconciled batches from a failed auto correction run.

    One new version for all fixes (not one per batch); provenance keeps the
    original run's model so the version does not read as a manual edit.
    """
    from podcodex.core.correct import save_corrected
    from podcodex.core.llm_failures import resolve_batches

    p, patched, section, chain = reconcile_batches(req, "corrected")
    provenance = build_provenance(
        "corrected",
        model=section.get("model"),
        params=llm_prov_params(
            section.get("mode", "manual"),
            batch_fixes=len(req.fixes),
            **({"source_chain": chain} if chain else {}),
        ),
        audio_path=req.audio_path,
        output_dir=req.output_dir,
    )
    version_id = save_corrected(
        req.audio_path, patched, output_dir=req.output_dir, provenance=provenance
    )
    # The remaining rejected batches now live in the patched version.
    stamp_run_version(req.audio_path, req.output_dir, "corrected", version_id)
    rejected = resolve_batches(p.base, "corrected", [fix.batch for fix in req.fixes])
    return {"status": "saved", "count": len(patched), "rejected": rejected}
