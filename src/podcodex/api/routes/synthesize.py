"""Synthesis pipeline routes — voice extraction, TTS generation, assembly."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from podcodex.api.routes._helpers import load_best_source
from podcodex.api.schemas import TaskResponse
from podcodex.api.tasks import task_manager
from podcodex.core._utils import AudioPaths

router = APIRouter()


# ── Status ───────────────────────────────────────────────


@router.get("/status")
async def synthesis_status(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Check which synthesis artifacts exist on disk."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    voice_dir = p.base.parent / "voice_samples"
    tts_dir = p.base.parent / "tts_segments"
    return {
        "voice_samples_extracted": voice_dir.is_dir() and any(voice_dir.glob("*.wav")),
        "tts_segments_generated": tts_dir.is_dir() and any(tts_dir.glob("*.wav")),
        "synthesized": p.synthesized.exists(),
    }


# ── Voice sample extraction ─────────────────────────────


class ExtractVoicesRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    min_duration: float | None = None
    max_duration: float | None = None
    top_k: int = 3


@router.post("/extract-voices", response_model=TaskResponse)
async def extract_voices(req: ExtractVoicesRequest) -> TaskResponse:
    """Extract voice samples for cloning as a background task."""

    def run_extract(progress_cb, req_data):
        from podcodex.core.synthesize import extract_voice_samples
        from podcodex.core.transcribe import load_transcript

        progress_cb(0.0, "Loading transcript...")
        segments = load_transcript(req_data.audio_path, output_dir=req_data.output_dir)
        if not segments:
            raise ValueError("No transcript found — transcribe first")

        progress_cb(0.1, "Extracting voice samples...")
        samples = extract_voice_samples(
            req_data.audio_path,
            segments,
            output_dir=req_data.output_dir,
            min_duration=req_data.min_duration,
            max_duration=req_data.max_duration,
            top_k=req_data.top_k,
        )

        total = sum(len(v) for v in samples.values())
        return {
            "speakers": len(samples),
            "total_samples": total,
        }

    try:
        info = task_manager.submit("extract_voices", req.audio_path, run_extract, req)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return TaskResponse(task_id=info.task_id)


# ── Voice samples (load from disk) ──────────────────────


@router.get("/voice-samples")
async def get_voice_samples(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict[str, list[dict]]:
    """Load extracted voice samples from disk."""
    from podcodex.core.synthesize import load_voice_samples
    from podcodex.core.transcribe import load_transcript

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    # Get speakers from transcript
    segments = load_transcript(audio_path, output_dir=output_dir) or []
    speakers = sorted(
        {s.get("speaker", "") for s in segments} - {"", "[BREAK]", "UNKNOWN", "UNK"}
    )

    samples = load_voice_samples(str(p.base.parent), speakers)

    # Convert Path objects to strings for JSON serialisation
    return {
        speaker: [
            {
                "file": str(entry["file"]),
                "duration": entry["duration"],
                "text": entry.get("text", ""),
            }
            for entry in entries
        ]
        for speaker, entries in samples.items()
    }


# ── TTS generation ───────────────────────────────────────


class GenerateRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    model_size: str = "1.7B"
    language: str = "English"
    source_lang: str | None = None
    max_chunk_duration: float = 20.0


@router.post("/generate", response_model=TaskResponse)
async def generate_tts(req: GenerateRequest) -> TaskResponse:
    """Generate TTS audio for all segments as a background task."""

    def run_generate(progress_cb, req_data):
        from podcodex.core._utils import free_vram
        from podcodex.core.synthesize import (
            build_clone_prompts,
            generate_segment,
            load_tts_model,
            load_voice_samples,
        )

        progress_cb(0.0, "Loading source segments...")
        # Try to load translation matching the target language
        p = AudioPaths.from_audio(req_data.audio_path, output_dir=req_data.output_dir)
        segments = None
        if req_data.source_lang:
            lang_file = p.translation_best(req_data.source_lang)
            if lang_file.exists():
                import json

                data = json.loads(lang_file.read_text(encoding="utf-8"))
                segments = data if isinstance(data, list) else data.get("segments")

        if not segments:
            try:
                segments = load_best_source(req_data.audio_path, req_data.output_dir)
            except ValueError:
                pass
        if not segments:
            raise ValueError("No source segments found")

        progress_cb(0.05, "Loading voice samples...")
        speakers = sorted(
            {s.get("speaker", "") for s in segments} - {"", "[BREAK]", "UNKNOWN", "UNK"}
        )
        voice_samples = load_voice_samples(str(p.base.parent), speakers)
        if not voice_samples:
            raise ValueError("No voice samples found — extract voices first")

        progress_cb(0.1, "Loading TTS model...")
        model = load_tts_model(model_size=req_data.model_size)
        clone_prompts = build_clone_prompts(model, voice_samples)

        segments_dir = p.tts_segments_dir
        generated = []
        total = len(segments)

        for i, seg in enumerate(segments):
            speaker = seg.get("speaker", "UNK")
            if speaker == "[BREAK]" or not seg.get("text", "").strip():
                continue

            frac = 0.1 + 0.85 * (i / total)
            progress_cb(frac, f"Segment {i + 1}/{total} ({speaker})")

            output_path = segments_dir / f"{i:04d}_{speaker}.wav"
            result = generate_segment(
                model,
                seg,
                clone_prompts,
                output_path,
                language=req_data.language,
                max_chunk_duration=req_data.max_chunk_duration,
            )
            if result:
                generated.append(result)

        progress_cb(0.98, "Releasing GPU memory...")
        free_vram(model)

        return {"count": len(generated), "skipped": total - len(generated)}

    try:
        info = task_manager.submit("generate_tts", req.audio_path, run_generate, req)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return TaskResponse(task_id=info.task_id)


# ── Generated segments (load from disk) ─────────────────


@router.get("/generated-segments")
async def get_generated_segments(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """Load generated TTS segments from disk."""
    from podcodex.core.synthesize import load_generated_segments

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    try:
        segments = load_best_source(audio_path, output_dir)
    except ValueError:
        raise HTTPException(404, "No source segments found")

    generated = load_generated_segments(str(p.base.parent), segments)
    # Convert Path objects for JSON
    return [
        {
            "speaker": seg.get("speaker", ""),
            "text": seg.get("text", ""),
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "audio_file": str(seg["audio_file"]),
            "duration": seg.get("end", 0) - seg.get("start", 0),
        }
        for seg in generated
    ]


# ── Assembly ─────────────────────────────────────────────


class AssembleRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    strategy: str = "original_timing"
    silence_duration: float = 0.5


@router.post("/assemble")
async def assemble(req: AssembleRequest) -> dict:
    """Assemble generated TTS segments into a final episode audio file."""
    from podcodex.core.synthesize import assemble_episode, load_generated_segments

    p = AudioPaths.from_audio(req.audio_path, output_dir=req.output_dir)

    try:
        segments = load_best_source(req.audio_path, req.output_dir)
    except ValueError:
        raise HTTPException(404, "No source segments found")

    generated = load_generated_segments(str(p.base.parent), segments)
    if not generated:
        raise HTTPException(404, "No generated TTS segments found")

    try:
        out_path = assemble_episode(
            generated,
            req.audio_path,
            output_dir=req.output_dir,
            strategy=req.strategy,
            silence_duration=req.silence_duration,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    import soundfile as sf

    info = sf.info(str(out_path))
    return {
        "path": str(out_path),
        "duration": info.duration,
    }
