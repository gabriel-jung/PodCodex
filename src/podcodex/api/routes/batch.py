"""Batch pipeline routes — run pipeline steps across multiple episodes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import (
    build_provenance,
    enrich_correct_kwargs,
    llm_prov_params,
    submit_task,
    transcribe_prov_params,
)
from podcodex.core._utils import normalize_lang
from podcodex.api.schemas import TaskResponse

router = APIRouter()


class BatchRequest(BaseModel):
    show_folder: str
    audio_paths: list[str]
    # Step toggles
    transcribe: bool = True
    correct: bool = True
    translate: bool = True
    index: bool = True
    # Transcribe config
    model_size: str = "large-v3-turbo"
    language: str = ""
    batch_size: int | None = None
    diarize: bool = True
    clean: bool = False
    hf_token: str | None = None
    num_speakers: int | None = None
    # Transcribe source: "audio" (WhisperX) or "subtitles" (cached VTT)
    transcribe_source: str = "audio"
    sub_lang: str = "en"
    # Correct/Translate config (LLM)
    llm_mode: str = "ollama"
    llm_provider: str | None = None
    llm_model: str = ""
    llm_api_base_url: str = ""
    llm_api_key: str | None = None
    context: str = ""
    source_lang: str = "French"
    target_lang: str = "English"
    llm_batch_minutes: float = 15.0
    engine: str = "whisper"
    force: bool = False
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
    "correct": 0.2,
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
    logger.debug("Batch start — language={!r}, diarize={}", req.language, req.diarize)

    # Check no individual tasks are running on any of the requested paths
    from podcodex.api.tasks import task_manager

    for ap in req.audio_paths:
        existing = task_manager.get_active(ap)
        if existing:
            raise HTTPException(
                409,
                f"Task {existing.task_id} already running on {Path(ap).stem}",
            )

    return submit_task("batch", f"batch:{req.show_folder}", _run_batch, req)


def _batch_transcribe(audio_path, stem, p, req, cancelled, ep_progress, i, step_offset):
    """Run transcribe sub-steps. Returns True if work was done.

    Each sub-step checks the version DB for matching params to decide
    whether to skip.
    """
    from podcodex.core.transcribe import (
        assign_speakers,
        diarize_file,
        export_transcript,
        transcribe_file,
    )
    from podcodex.core.versions import has_matching_version, has_version

    from podcodex.core._utils import default_batch_size

    sw = _STEP_WEIGHTS["transcribe"]
    did_work = False
    batch_size = req.batch_size or default_batch_size()

    # Check if a matching final transcript already exists (skip everything)
    transcript_params = {
        "model": req.model_size,
        "diarize": req.diarize,
    }
    if req.language:
        transcript_params["language"] = req.language
    if not req.force and has_matching_version(p.base, "transcript", transcript_params):
        return False

    # Sub-step 1: Transcribe (WhisperX)
    seg_params = {"model": req.model_size}
    if req.language:
        seg_params["language"] = req.language
    new_segments = not has_matching_version(p.base, "segments", seg_params)
    if new_segments:
        did_work = True
        ep_progress(i, step_offset, sw, 0.0, "Transcribing...")
        transcribe_file(
            audio_path,
            model_size=req.model_size,
            language=req.language or None,
            batch_size=batch_size,
        )
        if cancelled():
            return did_work

    # Sub-step 2: Diarize (pyannote)
    new_diarization = False
    if not cancelled() and req.diarize:
        if not has_version(p.base, "diarization"):
            new_diarization = True
            did_work = True
            ep_progress(i, step_offset, sw, 0.4, "Diarizing...")
            diarize_file(
                audio_path,
                hf_token=req.hf_token,
                num_speakers=req.num_speakers,
            )
            if cancelled():
                return did_work

    # Sub-step 3: Assign speakers
    # Re-run if segments or diarization changed (inputs are newer than output)
    if not cancelled() and req.diarize:
        if (
            new_segments
            or new_diarization
            or not has_version(p.base, "diarized_segments")
        ):
            did_work = True
            ep_progress(i, step_offset, sw, 0.7, "Assigning speakers...")
            assign_speakers(audio_path)
            if cancelled():
                return did_work

    # Sub-step 4: Export final transcript
    if not cancelled():
        did_work = True
        ep_progress(i, step_offset, sw, 0.9, "Exporting transcript...")

        # When diarizing, export both a plain (non-diarized) and a diarized
        # transcript so downstream steps can pick whichever they need.
        variants = [False, True] if req.diarize else [False]
        for diarized_flag in variants:
            provenance = build_provenance(
                "transcript",
                model=req.model_size,
                params=transcribe_prov_params(
                    diarized_flag,
                    model=req.model_size,
                    language=req.language or None,
                    batch_size=batch_size,
                    num_speakers=req.num_speakers,
                    clean=req.clean,
                ),
            )
            export_transcript(
                audio_path,
                show=req.show_name,
                episode=stem,
                diarized=diarized_flag,
                clean=req.clean,
                provenance=provenance,
            )

    return did_work


def _batch_transcribe_from_subs(
    audio_path, stem, p, req, cancelled, ep_progress, i, step_offset
):
    """Import cached VTT subtitles as a transcript version. Returns True if work was done."""
    from podcodex.core._utils import vtt_to_segments
    from podcodex.core.pipeline_db import mark_step
    from podcodex.core.versions import save_version

    sw = _STEP_WEIGHTS["transcribe"]
    ep_progress(i, step_offset, sw, 0.0, "Importing subtitles...")

    # Find cached VTT file
    vtt_path = p.base.parent / f"{stem}.subtitles.{req.sub_lang}.vtt"
    if not vtt_path.exists():
        # Try downloading if not cached yet
        from podcodex.ingest.youtube import cache_youtube_subtitles, _pace_request

        ep_progress(i, step_offset, sw, 0.2, "Downloading subtitles...")
        # Extract video_id from feed cache
        video_id = _resolve_video_id(p.base.parent)
        if not video_id:
            logger.warning("No video ID found for {}, skipping subtitle import", stem)
            return False
        _pace_request()
        if not cache_youtube_subtitles(
            video_id, p.base.parent, stem, lang=req.sub_lang
        ):
            logger.warning("No subtitles available for {}", stem)
            return False

    if cancelled():
        return False

    ep_progress(i, step_offset, sw, 0.5, "Parsing subtitles...")
    vtt_text = vtt_path.read_text(encoding="utf-8")
    segments = vtt_to_segments(vtt_text)
    if not segments:
        logger.warning("VTT parsed to zero segments for {}", stem)
        return False

    ep_progress(i, step_offset, sw, 0.8, "Saving transcript...")
    provenance = build_provenance(
        "transcript",
        params=transcribe_prov_params(
            diarize=False,
            source="youtube-subtitles",
            language=req.sub_lang,
        ),
    )
    save_version(p.base, "transcript", segments, provenance)
    mark_step(
        p.show_dir, p.base.name, transcribed=True, provenance={"transcript": provenance}
    )
    return True


def _resolve_video_id(episode_dir: Path) -> str | None:
    """Extract YouTube video ID from episode metadata."""
    from podcodex.ingest.rss import load_episode_meta

    meta = load_episode_meta(episode_dir)
    return meta.guid if meta else None


def _batch_llm_step(
    audio_path, p, req, cancelled, ep_progress, i, step_offset, *, step
):
    """Run a correct or translate step. Returns True if work was done.

    Skips if a version already exists with matching LLM params.
    """
    from podcodex.core.versions import has_matching_version

    is_translate = step == "translate"
    step_name = normalize_lang(req.target_lang) if is_translate else "corrected"
    sw = _STEP_WEIGHTS[step]

    match_params = {
        "model": req.llm_model,
        "llm_mode": req.llm_mode,
        "llm_provider": req.llm_provider,
        "source_lang": req.source_lang,
    }
    if is_translate:
        match_params["target_lang"] = req.target_lang
    if not req.force and has_matching_version(p.base, step_name, match_params):
        return False

    # Load source segments
    if is_translate:
        from podcodex.api.routes._helpers import load_best_source

        try:
            segments = load_best_source(audio_path)
        except ValueError:
            return False
    else:
        from podcodex.core.transcribe import load_transcript

        segments = load_transcript(audio_path)
        if not segments:
            return False

    label = "Translating" if is_translate else "Correcting"
    ep_progress(i, step_offset, sw, 0.0, f"{label}...")

    llm_kwargs = dict(
        mode=req.llm_mode,
        context=req.context,
        source_lang=req.source_lang,
        model=req.llm_model,
        batch_minutes=req.llm_batch_minutes,
        provider=req.llm_provider,
        api_base_url=req.llm_api_base_url,
        api_key=req.llm_api_key,
        original_segments=segments,
        merge=False,
    )

    prov_params = llm_prov_params(
        req.llm_mode,
        req.llm_provider,
        source_lang=req.source_lang,
        batch_minutes=req.llm_batch_minutes,
    )

    if is_translate:
        from podcodex.core.translate import save_translation_raw, translate_segments

        llm_kwargs["target_lang"] = req.target_lang
        prov_params["target_lang"] = req.target_lang
        result = translate_segments(segments, **llm_kwargs)
        provenance = build_provenance(
            step_name, model=req.llm_model, audio_path=audio_path, params=prov_params
        )
        save_translation_raw(audio_path, result, req.target_lang, provenance=provenance)
    else:
        from podcodex.core.correct import correct_segments, save_corrected

        tc_kwargs = enrich_correct_kwargs(audio_path, None, req.source_lang)
        llm_kwargs.update(tc_kwargs)
        prov_params["engine"] = tc_kwargs["engine"]
        prov_params["source_lang"] = tc_kwargs["source_lang"]
        result = correct_segments(segments, **llm_kwargs)
        provenance = build_provenance(
            step_name, model=req.llm_model, audio_path=audio_path, params=prov_params
        )
        save_corrected(audio_path, result, provenance=provenance)

    return True


def _batch_index(audio_path, stem, p, req, cancelled, ep_progress, i, step_offset):
    """Run index step. Returns True if work was done."""
    sw = _STEP_WEIGHTS["index"]
    marker = p.base.parent / ".rag_indexed"

    if not req.force and marker.exists():
        return False

    ep_progress(i, step_offset, sw, 0.0, "Indexing...")

    from podcodex.api.routes._helpers import build_index_transcript
    from podcodex.rag.indexing import vectorize_batch
    from podcodex.rag.localstore import LocalStore

    # Let build_index_transcript resolve the best source (corrected > transcript)
    transcript = build_index_transcript(audio_path, req.show_name, stem)
    if not transcript.get("segments"):
        return False

    db_path = p.vectors_db
    local = LocalStore(db_path)
    try:
        upserted = vectorize_batch(
            transcript,
            req.show_name,
            stem,
            req.index_model_keys,
            req.index_chunkings,
            local,
        )
    finally:
        local.close()

    if upserted == 0:
        logger.warning("Index produced 0 chunks for {} — not marking as indexed", stem)
        return False

    marker.touch()

    from podcodex.core.pipeline_db import mark_step

    provenance = build_provenance(
        "indexed",
        model=(req.index_model_keys or ["bge-m3"])[0],
        audio_path=audio_path,
        params={
            "model_keys": req.index_model_keys,
            "chunkings": req.index_chunkings,
        },
    )
    mark_step(p.show_dir, p.base.name, indexed=True, provenance={"indexed": provenance})
    return True


def _run_batch(progress_cb, req: BatchRequest):
    """Sequential batch pipeline: loop episodes, run enabled steps."""
    from podcodex.core._utils import AudioPaths
    from podcodex.ingest.folder import invalidate_scan_cache

    from podcodex.api.tasks import task_manager

    cancel = getattr(progress_cb, "cancel_event", None)
    batch_info = task_manager.get_active(f"batch:{req.show_folder}")
    batch_task_id = batch_info.task_id if batch_info else "batch"
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
            has_audio = Path(audio_path).exists()
            p = AudioPaths.from_audio(audio_path)
            step_offset = 0.0

            if req.transcribe and not _cancelled():
                if req.transcribe_source == "subtitles":
                    if _batch_transcribe_from_subs(
                        audio_path,
                        stem,
                        p,
                        req,
                        _cancelled,
                        ep_progress,
                        i,
                        step_offset,
                    ):
                        ep_had_work = True
                elif not has_audio:
                    logger.debug("Skipping transcribe for {} (no audio file)", stem)
                elif _batch_transcribe(
                    audio_path,
                    stem,
                    p,
                    req,
                    _cancelled,
                    ep_progress,
                    i,
                    step_offset,
                ):
                    ep_had_work = True
                step_offset += _STEP_WEIGHTS["transcribe"]

            if req.correct and not _cancelled():
                if _batch_llm_step(
                    audio_path,
                    p,
                    req,
                    _cancelled,
                    ep_progress,
                    i,
                    step_offset,
                    step="correct",
                ):
                    ep_had_work = True
                step_offset += _STEP_WEIGHTS["correct"]

            if req.translate and req.target_lang and not _cancelled():
                if _batch_llm_step(
                    audio_path,
                    p,
                    req,
                    _cancelled,
                    ep_progress,
                    i,
                    step_offset,
                    step="translate",
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
