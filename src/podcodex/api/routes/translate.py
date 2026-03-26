"""Translate pipeline routes — load/save translations, run translate."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from podcodex.api.routes._helpers import (
    annotate_flags,
    load_best_source,
    read_segments,
    save_segments_json,
)
from podcodex.api.schemas import Segment, TaskResponse
from podcodex.api.tasks import task_manager
from podcodex.core._utils import AudioPaths

router = APIRouter()


# ── Load / save ──────────────────────────────────────────


@router.get("/segments")
async def get_translated_segments(
    audio_path: str = Query(...),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load translated segments (prefers validated over raw)."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    data = read_segments(p.translation_best(lang))
    if data is None:
        raise HTTPException(404, f"No translation found for '{lang}'")
    return annotate_flags(data)


@router.get("/segments/raw")
async def get_translated_segments_raw(
    audio_path: str = Query(...),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load raw translated segments."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    data = read_segments(p.translation_raw(lang))
    if data is None:
        raise HTTPException(404, f"No raw translation found for '{lang}'")
    return annotate_flags(data)


@router.put("/segments")
async def save_translated_segments(
    segments: list[Segment],
    audio_path: str = Query(...),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Save validated translated segments."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    count = save_segments_json(
        p.translation(lang), [s.model_dump() for s in segments], f"Translation ({lang})"
    )
    return {"status": "saved", "count": count}


@router.get("/version-info")
async def translate_version_info(
    audio_path: str = Query(...),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Return which translation versions exist for a language."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return {
        "has_raw": p.translation_raw(lang).exists(),
        "has_validated": p.translation(lang).exists(),
    }


@router.get("/languages")
async def list_languages(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[str]:
    """List available translation languages."""
    from podcodex.core.translate import list_translations

    return list_translations(audio_path, output_dir=output_dir)


# ── Pipeline execution ───────────────────────────────────


class TranslateRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    mode: str = "ollama"
    provider: str | None = None
    model: str = ""
    context: str = ""
    source_lang: str = "French"
    target_lang: str = "English"
    batch_size: int = 10


@router.post("/start", response_model=TaskResponse)
async def start_translate(req: TranslateRequest) -> TaskResponse:
    """Start the translate pipeline as a background task."""

    def run_translate(progress_cb, req_data):
        from podcodex.core.translate import save_translation_raw, translate_segments

        progress_cb(0.0, "Loading source segments...")
        segments = load_best_source(req_data.audio_path, req_data.output_dir)

        progress_cb(0.1, "Starting translation...")

        def on_batch(batch_num, total):
            frac = 0.1 + 0.8 * (batch_num / total)
            progress_cb(frac, f"Batch {batch_num} of {total}")

        translated = translate_segments(
            segments,
            mode=req_data.mode,
            context=req_data.context,
            source_lang=req_data.source_lang,
            target_lang=req_data.target_lang,
            model=req_data.model,
            batch_size=req_data.batch_size,
            provider=req_data.provider,
            original_segments=segments,
            merge=False,  # source segments are already merged on load/upload
            on_batch=on_batch,
        )

        progress_cb(0.95, "Saving...")
        save_translation_raw(
            req_data.audio_path,
            translated,
            req_data.target_lang,
            output_dir=req_data.output_dir,
        )
        return {"count": len(translated), "lang": req_data.target_lang}

    try:
        info = task_manager.submit("translate", req.audio_path, run_translate, req)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return TaskResponse(task_id=info.task_id)


# ── Manual mode ──────────────────────────────────────────


class ManualPromptsRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    context: str = ""
    source_lang: str = "French"
    target_lang: str = "English"
    batch_minutes: float = 15.0


@router.post("/manual-prompts")
async def generate_manual_prompts(req: ManualPromptsRequest) -> list[dict]:
    """Generate batched prompts for manual translation."""
    from podcodex.core.translate import build_manual_prompts_batched

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
    return [
        {"batch_index": i, "prompt": prompt, "segment_count": len(batch_segs)}
        for i, (batch_segs, prompt) in enumerate(batches)
    ]


class ApplyManualRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    lang: str = "English"
    corrections: list[dict]


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
    save_translation_raw(
        req.audio_path, translated, req.lang, output_dir=req.output_dir
    )
    return {"status": "saved", "count": len(translated)}
