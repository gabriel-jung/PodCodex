"""Translate pipeline routes — load/save translations, run translate."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from podcodex.api.routes._helpers import (
    ApplyManualRequest,
    LLMRequest,
    ManualPromptsRequest,
    batch_progress,
    build_provenance,
    format_prompt_batches,
    llm_prov_params,
    load_best_source,
    submit_task,
)
from podcodex.api.routes._versions import register_version_routes
from podcodex.api.schemas import Segment, TaskResponse
from podcodex.core._utils import AudioPaths, normalize_lang

router = APIRouter()
register_version_routes(router, lang_param=True)


# ── Load / save ──────────────────────────────────────────


@router.get("/segments")
async def get_translated_segments(
    audio_path: str = Query(...),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load translated segments from the version DB."""
    from podcodex.api.routes._helpers import annotate_flags
    from podcodex.core.versions import load_latest

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    lang_norm = normalize_lang(lang)
    segments = load_latest(p.base, lang_norm)
    if segments is None:
        raise HTTPException(404, f"No translation found for '{lang}'")
    return annotate_flags(segments)


@router.put("/segments")
async def save_translated_segments(
    segments: list[Segment],
    audio_path: str = Query(...),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Save validated translated segments."""
    from podcodex.core.translate import save_translation

    lang_norm = normalize_lang(lang)
    seg_dicts = [s.model_dump() for s in segments]
    provenance = build_provenance(lang_norm, ptype="validated", manual_edit=True)
    save_translation(
        audio_path, seg_dicts, lang, output_dir=output_dir, provenance=provenance
    )
    return {"status": "saved", "count": len(seg_dicts)}


@router.get("/languages")
async def list_languages(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[str]:
    """List available translation languages."""
    from podcodex.core.translate import list_translations

    return list_translations(audio_path, output_dir=output_dir)


# ── Pipeline execution ───────────────────────────────────


class TranslateRequest(LLMRequest):
    target_lang: str = "English"


@router.post("/start", response_model=TaskResponse)
async def start_translate(req: TranslateRequest) -> TaskResponse:
    """Start the translate pipeline as a background task."""

    def run_translate(progress_cb, req_data):
        """Load source segments, run translation in batches, and save the raw output."""
        from podcodex.core.translate import save_translation_raw, translate_segments

        progress_cb(0.0, "Loading source segments...")
        segments = load_best_source(req_data.audio_path, req_data.output_dir)

        progress_cb(0.1, "Starting translation...")

        translated = translate_segments(
            segments,
            mode=req_data.mode,
            context=req_data.context,
            source_lang=req_data.source_lang,
            target_lang=req_data.target_lang,
            model=req_data.model,
            batch_minutes=req_data.batch_minutes,
            provider=req_data.provider,
            api_base_url=req_data.api_base_url,
            api_key=req_data.api_key,
            original_segments=segments,
            merge=False,  # source segments are already merged on load/upload
            on_batch=batch_progress(progress_cb),
        )

        progress_cb(0.95, "Saving...")
        lang_norm = normalize_lang(req_data.target_lang)
        provenance = build_provenance(
            lang_norm,
            model=req_data.model,
            audio_path=req_data.audio_path,
            output_dir=req_data.output_dir,
            params=llm_prov_params(
                req_data.mode,
                req_data.provider,
                source_lang=req_data.source_lang,
                target_lang=req_data.target_lang,
                batch_minutes=req_data.batch_minutes,
            ),
        )
        save_translation_raw(
            req_data.audio_path,
            translated,
            req_data.target_lang,
            output_dir=req_data.output_dir,
            provenance=provenance,
        )
        return {"count": len(translated), "lang": req_data.target_lang}

    return submit_task("translate", req.audio_path, run_translate, req)


# ── Manual mode ──────────────────────────────────────────


@router.post("/manual-prompts")
async def generate_manual_prompts(req: ManualPromptsRequest) -> list[dict]:
    """Generate batched prompts for manual translation."""
    from podcodex.core.translate import build_manual_prompts_batched
    from podcodex.core.versions import load_version

    if req.source_version_id:
        p = AudioPaths.from_audio(req.audio_path, output_dir=req.output_dir)
        # Determine which step the version belongs to (corrected or transcript)
        try:
            segments = load_version(p.base, "corrected", req.source_version_id)
        except FileNotFoundError:
            segments = load_version(p.base, "transcript", req.source_version_id)
    else:
        try:
            segments = load_best_source(req.audio_path, req.output_dir)
        except ValueError:
            raise HTTPException(404, "No source segments found")

    batches = build_manual_prompts_batched(
        segments,
        batch_minutes=req.batch_minutes,
        context=req.context,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
    )
    return format_prompt_batches(batches)


@router.post("/apply-manual")
async def apply_manual_corrections(req: ApplyManualRequest) -> dict:
    """Apply manually-obtained translation corrections and save as raw."""
    from podcodex.core._utils import validate_manual
    from podcodex.core.translate import save_translation_raw

    try:
        original = load_best_source(req.audio_path, req.output_dir)
    except ValueError:
        raise HTTPException(404, "No source segments found")

    translated = validate_manual(req.corrections, original)
    lang_norm = normalize_lang(req.lang)
    provenance = build_provenance(
        lang_norm,
        params=llm_prov_params("manual"),
        manual_edit=True,
    )
    save_translation_raw(
        req.audio_path,
        translated,
        req.lang,
        output_dir=req.output_dir,
        provenance=provenance,
    )
    return {"status": "saved", "count": len(translated)}
