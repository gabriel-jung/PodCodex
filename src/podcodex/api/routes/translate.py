"""Translate pipeline routes — load/save translations, run translate."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import (
    load_best_source,
    load_segments_or_404,
    submit_task,
)
from podcodex.api.schemas import Segment, TaskResponse
from podcodex.core._utils import AudioPaths

router = APIRouter()


# ── Load / save ──────────────────────────────────────────


@router.get("/segments")
async def get_translated_segments(
    audio_path: str = Query(...),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load translated segments (latest version, falls back to legacy files)."""
    from podcodex.api.routes._helpers import annotate_flags
    from podcodex.core.versions import load_latest

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    lang_norm = lang.lower().strip().replace(" ", "_")
    segments = load_latest(p.base, lang_norm)
    if segments is not None:
        return annotate_flags(segments)
    return load_segments_or_404(p.translation_best(lang), f"translation for '{lang}'")


@router.put("/segments")
async def save_translated_segments(
    segments: list[Segment],
    audio_path: str = Query(...),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Save validated translated segments."""
    from podcodex.core.translate import save_translation

    lang_norm = lang.lower().strip().replace(" ", "_")
    seg_dicts = [s.model_dump() for s in segments]
    provenance = {
        "step": lang_norm,
        "type": "validated",
        "model": None,
        "params": {},
        "manual_edit": True,
    }
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


# ── Version history ──────────────────────────────────────


@router.get("/versions")
async def list_translate_versions(
    audio_path: str = Query(...),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """List all archived translation versions for a language (newest first)."""
    from podcodex.core.versions import list_versions

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    lang_norm = lang.lower().strip().replace(" ", "_")
    return list_versions(p.base, lang_norm)


@router.get("/versions/{version_id}")
async def load_translate_version(
    version_id: str,
    audio_path: str = Query(...),
    lang: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load segments from a specific archived translation version."""
    from podcodex.core.versions import load_version

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    lang_norm = lang.lower().strip().replace(" ", "_")
    try:
        return load_version(p.base, lang_norm, version_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Version {version_id} not found")


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
    batch_minutes: float = 15.0
    api_base_url: str = ""
    api_key: str | None = None

    @field_validator("batch_minutes")
    @classmethod
    def batch_minutes_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("batch_minutes must be positive")
        return v


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
            batch_minutes=req_data.batch_minutes,
            provider=req_data.provider,
            api_base_url=req_data.api_base_url,
            api_key=req_data.api_key,
            original_segments=segments,
            merge=False,  # source segments are already merged on load/upload
            on_batch=on_batch,
        )

        progress_cb(0.95, "Saving...")
        lang_norm = req_data.target_lang.lower().strip().replace(" ", "_")
        provenance = {
            "step": lang_norm,
            "type": "raw",
            "model": req_data.model,
            "params": {
                "mode": req_data.mode,
                "provider": req_data.provider,
                "source_lang": req_data.source_lang,
                "target_lang": req_data.target_lang,
                "batch_minutes": req_data.batch_minutes,
            },
            "manual_edit": False,
        }
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
    lang_norm = req.lang.lower().strip().replace(" ", "_")
    provenance = {
        "step": lang_norm,
        "type": "raw",
        "model": None,
        "params": {"mode": "manual"},
        "manual_edit": True,
    }
    save_translation_raw(
        req.audio_path,
        translated,
        req.lang,
        output_dir=req.output_dir,
        provenance=provenance,
    )
    return {"status": "saved", "count": len(translated)}
