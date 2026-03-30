"""Batch pipeline routes — run pipeline steps across multiple episodes."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from podcodex.api.routes._helpers import submit_task
from podcodex.api.schemas import TaskResponse

router = APIRouter()
logger = logging.getLogger(__name__)


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


def _run_batch(progress_cb, req: BatchRequest):  # noqa: C901
    """Sequential batch pipeline: loop episodes, run enabled steps."""
    from podcodex.core._utils import AudioPaths
    from podcodex.core.transcribe import (
        assign_speakers,
        diarize_file,
        export_transcript,
        processing_status,
        transcribe_file,
    )
    from podcodex.ingest.folder import invalidate_scan_cache

    from podcodex.api.tasks import task_manager

    cancel = getattr(progress_cb, "cancel_event", None)
    # The batch task_id is the lock value set by submit() on req.show_folder.
    # Read it once — cancel() may remove it later.
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
        """Report compound progress: episode position + step fraction."""
        ep_frac = (step_offset + frac * step_weight) / weight
        overall = (ep_idx + ep_frac) / total
        progress_cb(overall, f"[{ep_idx + 1}/{total}] {msg}")

    for i, audio_path in enumerate(req.audio_paths):
        # Check cancellation between episodes
        if _cancelled():
            progress_cb(i / total, "Cancelled")
            break

        stem = Path(audio_path).stem

        # Skip if an individual task is running on this episode
        if task_manager.get_active(audio_path):
            progress_cb((i + 1) / total, f"[{i + 1}/{total}] Skipped (task running)")
            skipped += 1
            continue

        # Lock this episode for the duration of processing
        task_manager.lock(audio_path, batch_task_id)

        progress_cb(i / total, f"[{i + 1}/{total}] Starting...")
        ep_had_work = False

        try:
            status = processing_status(audio_path)
            p = AudioPaths.from_audio(audio_path)
            step_offset = 0.0

            # ── Transcribe ──
            if req.transcribe and not _cancelled():
                sw = _STEP_WEIGHTS["transcribe"]
                if not status["transcribed"]:
                    ep_had_work = True
                    ep_progress(i, step_offset, sw, 0.0, "Transcribing...")
                    transcribe_file(
                        audio_path,
                        model_size=req.model_size,
                        language=req.language or None,
                        batch_size=req.batch_size,
                    )

                if not _cancelled() and req.diarize and not status["diarized"]:
                    ep_had_work = True
                    ep_progress(i, step_offset, sw, 0.4, "Diarizing...")
                    diarize_file(
                        audio_path,
                        hf_token=req.hf_token,
                        num_speakers=req.num_speakers,
                    )

                if not _cancelled() and req.diarize and not status["assigned"]:
                    ep_had_work = True
                    ep_progress(i, step_offset, sw, 0.7, "Assigning speakers...")
                    assign_speakers(audio_path)

                if not _cancelled() and not status["exported"]:
                    ep_had_work = True
                    ep_progress(i, step_offset, sw, 0.9, "Exporting transcript...")
                    export_transcript(
                        audio_path,
                        show=req.show_name,
                        episode=stem,
                        diarized=req.diarize,
                    )

                step_offset += sw

            # ── Polish ──
            if req.polish and not _cancelled():
                sw = _STEP_WEIGHTS["polish"]
                if not p.has_polished() and (
                    p.transcript.exists() or p.transcript_raw.exists()
                ):
                    ep_had_work = True
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

                step_offset += sw

            # ── Translate ──
            if req.translate and req.target_lang and not _cancelled():
                sw = _STEP_WEIGHTS["translate"]
                if not p.has_translation(req.target_lang) and (
                    p.has_polished()
                    or p.transcript.exists()
                    or p.transcript_raw.exists()
                ):
                    ep_had_work = True
                    ep_progress(i, step_offset, sw, 0.0, "Translating...")

                    from podcodex.api.routes._helpers import load_best_source
                    from podcodex.core.translate import (
                        save_translation_raw,
                        translate_segments,
                    )

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

                step_offset += sw

            # ── Index ──
            if req.index and not _cancelled():
                sw = _STEP_WEIGHTS["index"]
                marker = p.base.parent / ".rag_indexed"

                if not marker.exists() and p.transcript_best.exists():
                    ep_had_work = True
                    ep_progress(i, step_offset, sw, 0.0, "Indexing...")

                    import json

                    from podcodex.cli import (
                        _resolve_source,
                        _source_label,
                        vectorize_batch,
                    )
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

            if _cancelled():
                # Partially processed — count what was done but don't mark complete
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
            logger.exception("Batch: episode %s failed", stem)
            failed += 1
            errors.append({"episode": stem, "error": str(exc)})
            progress_cb((i + 1) / total, f"[{i + 1}/{total}] Failed: {exc}")

        finally:
            # Release per-episode lock
            task_manager.unlock(audio_path)

        # Invalidate scan cache after each episode so UI updates
        try:
            ap = Path(audio_path)
            invalidate_scan_cache(ap.parent)
        except Exception:
            pass

    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
    }
