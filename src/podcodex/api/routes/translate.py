"""Translate pipeline routes — load/save translations, run translate."""

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
    format_prompt_batches,
    llm_prov_params,
    load_best_source,
    reconcile_batches,
    require_audio_or_output,
    submit_task,
)
from podcodex.core.llm_failures import clear_step_for, get_step, stamp_run_version
from podcodex.api.routes._versions import register_version_routes
from podcodex.api.schemas import Segment, TaskResponse
from podcodex.core._utils import AudioPaths, normalize_lang
from podcodex.core.source import load_source

router = APIRouter()
register_version_routes(router, lang_param=True)


# ── Load / save ──────────────────────────────────────────


@router.get("/segments")
def get_translated_segments(
    audio_path: str | None = Query(None),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load translated segments from the version DB.

    Re-applies the latest speaker map on the way out so renames performed
    after the translation was generated (e.g. SPEAKER_00 → Chris Fisher)
    show through. The saved translation file is untouched.
    """
    from podcodex.api.routes._helpers import shape_step_segments
    from podcodex.core.versions import load_latest

    require_audio_or_output(audio_path, output_dir)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    lang_norm = normalize_lang(lang)
    segments = load_latest(p.base, lang_norm)
    if segments is None:
        raise HTTPException(404, f"No translation found for '{lang}'")
    return shape_step_segments(p.base, lang_norm, segments)


@router.put("/segments")
def save_translated_segments(
    segments: list[Segment],
    audio_path: str | None = Query(None),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Save validated translated segments."""
    from podcodex.core.translate import save_translation

    require_audio_or_output(audio_path, output_dir)
    lang_norm = normalize_lang(lang)
    seg_dicts = [s.model_dump() for s in segments]
    provenance = build_edit_provenance(lang_norm, audio_path, output_dir)
    save_translation(
        audio_path, seg_dicts, lang, output_dir=output_dir, provenance=provenance
    )
    return {"status": "saved", "count": len(seg_dicts)}


# ── Pipeline execution ───────────────────────────────────


class TranslateRequest(LLMRequest):
    target_lang: str = "French"


@router.post("/start", response_model=TaskResponse)
def start_translate(req: TranslateRequest) -> TaskResponse:
    """Start the translate pipeline as a background task."""
    from podcodex.core.llm_resolver import LLMResolutionError, resolve_llm_run

    try:
        llm = resolve_llm_run(req.mode, req.provider_profile, req.key_name, req.model)
    except LLMResolutionError as exc:
        raise HTTPException(400, str(exc))

    def run_translate(progress_cb, req_data):
        """Load source segments, run translation in batches, and save the raw output."""
        from podcodex.core.translate import translate_and_save

        progress_cb(0.0, "Loading source segments...")
        source = load_source(
            req_data.audio_path, req_data.output_dir, req_data.source_version_id
        )
        progress_cb(0.1, "Starting translation...")
        translated, _version_id = translate_and_save(
            source,
            llm,
            audio_path=req_data.audio_path,
            output_dir=req_data.output_dir,
            target_lang=req_data.target_lang,
            context=req_data.context,
            source_lang=req_data.source_lang,
            batch_minutes=req_data.batch_minutes,
            provider_profile=req_data.provider_profile,
            key_name=req_data.key_name,
            on_batch=batch_progress(progress_cb),
        )
        return {"count": len(translated), "lang": req_data.target_lang}

    return submit_task(
        "translate",
        req.audio_path,
        run_translate,
        req,
        subject=normalize_lang(req.target_lang),
    )


# ── Manual mode ──────────────────────────────────────────


@router.post("/manual-prompts")
def generate_manual_prompts(req: ManualPromptsRequest) -> list[dict]:
    """Generate batched prompts for manual translation."""
    from podcodex.core.translate import build_manual_prompts_batched

    try:
        segments = load_best_source(
            req.audio_path, req.output_dir, req.source_version_id
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))

    batches = build_manual_prompts_batched(
        segments,
        batch_minutes=req.batch_minutes,
        batch_count=req.batch_count,
        context=req.context,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
    )
    return format_prompt_batches(batches)


@router.get("/llm-failures")
def get_translate_failures(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
    lang: str = Query(..., description="Target language of the translation"),
) -> dict | None:
    """Per-batch records of the last auto translation run, or None if none."""
    require_audio_or_output(audio_path, output_dir)
    return get_step(audio_path, output_dir, normalize_lang(lang))


@router.delete("/llm-failures")
def dismiss_translate_failures(
    audio_path: str | None = Query(None),
    output_dir: str | None = Query(None),
    lang: str = Query(..., description="Target language of the translation"),
) -> dict:
    """Dismiss the recorded translation batch results for this language."""
    require_audio_or_output(audio_path, output_dir)
    return {"cleared": clear_step_for(audio_path, output_dir, normalize_lang(lang))}


@router.post("/apply-manual")
def apply_manual_corrections(req: ApplyManualRequest) -> dict:
    """Apply manually-obtained translation corrections and save as raw."""
    from podcodex.core.llm import validate_manual
    from podcodex.core.translate import save_translation

    try:
        source = load_source(req.audio_path, req.output_dir, req.source_version_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))

    try:
        translated = validate_manual(req.corrections, source.segments)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    lang_norm = normalize_lang(req.lang)
    # manual_edit stays False: applying a manual LLM prompt is still an LLM
    # pass, not a reviewed hand-edit. Marking it edited made an unreviewed
    # paste outrank every later auto translation. The correct route already
    # does it this way; params llm_mode=manual records the provenance.
    provenance = build_provenance(
        lang_norm,
        params=llm_prov_params("manual"),
        audio_path=req.audio_path,
        output_dir=req.output_dir,
        source=source,
    )
    save_translation(
        req.audio_path,
        translated,
        req.lang,
        output_dir=req.output_dir,
        provenance=provenance,
    )
    return {"status": "saved", "count": len(translated)}


@router.post("/apply-batches")
def apply_batches_translation(req: ApplyBatchesRequest) -> dict:
    """Apply hand-reconciled batches from a failed auto translation run.

    One new version for all fixes (not one per batch); provenance keeps the
    original run's model so the version does not read as a manual edit.
    """
    from podcodex.core.llm_failures import resolve_batches
    from podcodex.core.translate import save_translation

    lang_norm = normalize_lang(req.lang)
    p, patched, section, chain = reconcile_batches(req, lang_norm)
    provenance = build_provenance(
        lang_norm,
        model=section.get("model"),
        params=llm_prov_params(
            section.get("mode", "manual"),
            batch_fixes=len(req.fixes),
            **({"source_chain": chain} if chain else {}),
        ),
        audio_path=req.audio_path,
        output_dir=req.output_dir,
    )
    version_id = save_translation(
        req.audio_path,
        patched,
        req.lang,
        output_dir=req.output_dir,
        provenance=provenance,
    )
    # The remaining rejected batches now live in the patched version.
    stamp_run_version(req.audio_path, req.output_dir, lang_norm, version_id)
    rejected = resolve_batches(p.base, lang_norm, [fix.batch for fix in req.fixes])
    return {"status": "saved", "count": len(patched), "rejected": rejected}
