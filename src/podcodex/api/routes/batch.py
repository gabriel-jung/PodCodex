"""Batch pipeline routes — run pipeline steps across multiple episodes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import submit_task
from podcodex.api.schemas import TaskResponse

router = APIRouter()


class BatchRequest(BaseModel):
    show_folder: str
    audio_paths: list[str]
    # Step toggles
    transcribe: bool = True
    polish: bool = True
    translate: bool = True
    index: bool = True
    # Transcribe config
    model_size: str = "large-v3-turbo"
    language: str = ""
    batch_size: int = 16
    diarize: bool = True
    hf_token: str | None = None
    num_speakers: int | None = None
    # Polish/Translate config (LLM)
    llm_mode: str = "ollama"
    llm_provider: str | None = None
    llm_model: str = ""
    llm_api_base_url: str = ""
    llm_api_key: str | None = None
    context: str = ""
    source_lang: str = "French"
    target_lang: str = "English"
    llm_batch_minutes: float = 15.0
    engine: str = "Whisper"
    # Index config
    show_name: str = ""
    index_model_keys: list[str] = ["bge-m3"]
    index_chunkings: list[str] = ["semantic"]

    @field_validator("batch_size")
    @classmethod
    def batch_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be at least 1")
        return v

    @field_validator("llm_batch_minutes")
    @classmethod
    def batch_minutes_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("llm_batch_minutes must be positive")
        return v


# Step weights for progress calculation (must sum to 1.0 when all enabled)
_STEP_WEIGHTS = {
    "transcribe": 0.5,
    "polish": 0.2,
    "translate": 0.2,
    "index": 0.1,
}


def _enabled_weight(req: BatchRequest) -> float:
    """Sum of weights for enabled steps."""
    return sum(w for k, w in _STEP_WEIGHTS.items() if getattr(req, k)) or 1.0


@router.post("/start", response_model=TaskResponse)
async def start_batch(req: BatchRequest) -> TaskResponse:
    """Start a batch pipeline run across multiple episodes."""
    if not req.audio_paths:
        raise HTTPException(400, "No episodes selected")

    # Check no individual tasks are running on any of the requested paths
    from podcodex.api.tasks import task_manager

    for ap in req.audio_paths:
        existing = task_manager.get_active(ap)
        if existing:
            raise HTTPException(
                409,
                f"Task {existing.task_id} already running on {Path(ap).stem}",
            )

    return submit_task("batch", req.show_folder, _run_batch, req)


def _batch_transcribe(
    audio_path, stem, req, status, cancelled, ep_progress, i, step_offset
):
    """Run transcribe sub-steps. Returns True if work was done."""
    from podcodex.core.transcribe import (
        assign_speakers,
        diarize_file,
        export_transcript,
        transcribe_file,
    )

    sw = _STEP_WEIGHTS["transcribe"]
    did_work = False

    if not status["transcribed"]:
        did_work = True
        ep_progress(i, step_offset, sw, 0.0, "Transcribing...")
        transcribe_file(
            audio_path,
            model_size=req.model_size,
            language=req.language or None,
            batch_size=req.batch_size,
        )

    if not cancelled() and req.diarize and not status["diarized"]:
        did_work = True
        ep_progress(i, step_offset, sw, 0.4, "Diarizing...")
        diarize_file(
            audio_path,
            hf_token=req.hf_token,
            num_speakers=req.num_speakers,
        )

    if not cancelled() and req.diarize and not status["assigned"]:
        did_work = True
        ep_progress(i, step_offset, sw, 0.7, "Assigning speakers...")
        assign_speakers(audio_path)

    if not cancelled() and not status["exported"]:
        did_work = True
        ep_progress(i, step_offset, sw, 0.9, "Exporting transcript...")
        export_transcript(
            audio_path,
            show=req.show_name,
            episode=stem,
            diarized=req.diarize,
        )

    return did_work


def _batch_polish(audio_path, p, req, cancelled, ep_progress, i, step_offset):
    """Run polish step. Returns True if work was done."""
    sw = _STEP_WEIGHTS["polish"]
    if not p.has_polished() and (p.transcript.exists() or p.transcript_raw.exists()):
        ep_progress(i, step_offset, sw, 0.0, "Polishing...")

        from podcodex.core.polish import polish_segments, save_polished_raw
        from podcodex.core.transcribe import load_transcript

        segments = load_transcript(audio_path)
        if segments:
            polished = polish_segments(
                segments,
                mode=req.llm_mode,
                context=req.context,
                source_lang=req.source_lang,
                model=req.llm_model,
                batch_minutes=req.llm_batch_minutes,
                provider=req.llm_provider,
                engine=req.engine,
                api_base_url=req.llm_api_base_url,
                api_key=req.llm_api_key,
                original_segments=segments,
                merge=False,
            )
            save_polished_raw(audio_path, polished)
            return True
    return False


def _batch_translate(audio_path, p, req, cancelled, ep_progress, i, step_offset):
    """Run translate step. Returns True if work was done."""
    sw = _STEP_WEIGHTS["translate"]
    if not p.has_translation(req.target_lang) and (
        p.has_polished() or p.transcript.exists() or p.transcript_raw.exists()
    ):
        ep_progress(i, step_offset, sw, 0.0, "Translating...")

        from podcodex.api.routes._helpers import load_best_source
        from podcodex.core.translate import save_translation_raw, translate_segments

        segments = load_best_source(audio_path)
        translated = translate_segments(
            segments,
            mode=req.llm_mode,
            context=req.context,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            model=req.llm_model,
            batch_minutes=req.llm_batch_minutes,
            provider=req.llm_provider,
            api_base_url=req.llm_api_base_url,
            api_key=req.llm_api_key,
            original_segments=segments,
            merge=False,
        )
        save_translation_raw(audio_path, translated, req.target_lang)
        return True
    return False


def _batch_index(audio_path, stem, p, req, cancelled, ep_progress, i, step_offset):
    """Run index step. Returns True if work was done."""
    import json

    sw = _STEP_WEIGHTS["index"]
    marker = p.base.parent / ".rag_indexed"

    if not marker.exists() and p.transcript_best.exists():
        ep_progress(i, step_offset, sw, 0.0, "Indexing...")

        from podcodex.cli import _resolve_source, _source_label, vectorize_batch
        from podcodex.rag.localstore import LocalStore

        transcript_path = p.transcript_best
        source_path = _resolve_source(transcript_path, "auto")
        source_label = _source_label(source_path, transcript_path)

        data = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            transcript = {
                "meta": {
                    "show": req.show_name,
                    "episode": stem,
                    "source": source_label,
                },
                "segments": data,
            }
        else:
            transcript = data
            transcript.setdefault("meta", {})
            transcript["meta"].setdefault("show", req.show_name)
            transcript["meta"].setdefault("episode", stem)
            transcript["meta"].setdefault("source", source_label)

        db_path = p.vectors_db
        local = LocalStore(db_path)
        try:
            vectorize_batch(
                transcript,
                req.show_name,
                stem,
                req.index_model_keys,
                req.index_chunkings,
                local,
            )
        finally:
            local.close()
        marker.touch()
        return True
    return False


def _run_batch(progress_cb, req: BatchRequest):
    """Sequential batch pipeline: loop episodes, run enabled steps."""
    from podcodex.core._utils import AudioPaths
    from podcodex.core.transcribe import processing_status
    from podcodex.ingest.folder import invalidate_scan_cache

    from podcodex.api.tasks import task_manager

    cancel = getattr(progress_cb, "cancel_event", None)
    batch_task_id = task_manager.get_active(req.show_folder)
    batch_task_id = batch_task_id.task_id if batch_task_id else "batch"
    total = len(req.audio_paths)
    weight = _enabled_weight(req)
    completed = 0
    failed = 0
    skipped = 0
    errors: list[dict] = []

    def _cancelled() -> bool:
        return bool(cancel and cancel.is_set())

    def ep_progress(
        ep_idx: int, step_offset: float, step_weight: float, frac: float, msg: str
    ):
        ep_frac = (step_offset + frac * step_weight) / weight
        overall = (ep_idx + ep_frac) / total
        progress_cb(overall, f"[{ep_idx + 1}/{total}] {msg}")

    for i, audio_path in enumerate(req.audio_paths):
        if _cancelled():
            progress_cb(i / total, "Cancelled")
            break

        stem = Path(audio_path).stem

        if task_manager.get_active(audio_path):
            progress_cb((i + 1) / total, f"[{i + 1}/{total}] Skipped (task running)")
            skipped += 1
            continue

        task_manager.lock(audio_path, batch_task_id)
        progress_cb(i / total, f"[{i + 1}/{total}] Starting...")
        ep_had_work = False

        try:
            status = processing_status(audio_path)
            p = AudioPaths.from_audio(audio_path)
            step_offset = 0.0

            if req.transcribe and not _cancelled():
                if _batch_transcribe(
                    audio_path,
                    stem,
                    req,
                    status,
                    _cancelled,
                    ep_progress,
                    i,
                    step_offset,
                ):
                    ep_had_work = True
                step_offset += _STEP_WEIGHTS["transcribe"]

            if req.polish and not _cancelled():
                if _batch_polish(
                    audio_path, p, req, _cancelled, ep_progress, i, step_offset
                ):
                    ep_had_work = True
                step_offset += _STEP_WEIGHTS["polish"]

            if req.translate and req.target_lang and not _cancelled():
                if _batch_translate(
                    audio_path, p, req, _cancelled, ep_progress, i, step_offset
                ):
                    ep_had_work = True
                step_offset += _STEP_WEIGHTS["translate"]

            if req.index and not _cancelled():
                if _batch_index(
                    audio_path, stem, p, req, _cancelled, ep_progress, i, step_offset
                ):
                    ep_had_work = True

            if _cancelled():
                if ep_had_work:
                    completed += 1
                progress_cb((i + 1) / total, f"[{i + 1}/{total}] Cancelled")
                break
            elif not ep_had_work:
                skipped += 1
                progress_cb(
                    (i + 1) / total, f"[{i + 1}/{total}] Skipped (already done)"
                )
            else:
                completed += 1
                progress_cb((i + 1) / total, f"[{i + 1}/{total}] Done")

        except Exception as exc:
            logger.exception("Batch: episode {} failed", stem)
            failed += 1
            errors.append({"episode": stem, "error": str(exc)})
            progress_cb((i + 1) / total, f"[{i + 1}/{total}] Failed: {exc}")

        finally:
            task_manager.unlock(audio_path)

        try:
            invalidate_scan_cache(Path(audio_path).parent)
        except Exception:
            logger.opt(exception=True).debug(
                "Failed to invalidate scan cache for {}", audio_path
            )

    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
    }
