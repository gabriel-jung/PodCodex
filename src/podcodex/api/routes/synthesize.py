"""Synthesis pipeline routes — voice extraction, TTS generation, assembly."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import load_best_source, submit_task
from podcodex.api.schemas import TaskResponse
from podcodex.core._utils import AudioPaths, SAMPLE_RATE

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

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        """Validate that top_k is at least 1."""
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v


@router.post("/extract-voices", response_model=TaskResponse)
async def extract_voices(req: ExtractVoicesRequest) -> TaskResponse:
    """Extract voice samples for cloning as a background task."""

    def run_extract(progress_cb, req_data):
        """Load transcript and extract speaker voice samples."""
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

    return submit_task("extract_voices", req.audio_path, run_extract, req)


class ExtractSelectedRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    selections: list[dict]  # [{speaker, start, end, text}, ...]


@router.post("/extract-selected")
async def extract_selected(req: ExtractSelectedRequest) -> dict:
    """Extract specific user-chosen segments as voice samples."""
    from podcodex.core.synthesize import extract_selected_samples

    if not req.selections:
        raise HTTPException(400, "No segments selected")

    samples = extract_selected_samples(
        req.audio_path,
        req.selections,
        output_dir=req.output_dir,
    )

    # Convert Path objects for JSON
    result = {}
    for speaker, entries in samples.items():
        result[speaker] = [
            {
                "file": str(e["file"]),
                "duration": e["duration"],
                "text": e.get("text", ""),
            }
            for e in entries
        ]

    total = sum(len(v) for v in result.values())
    return {
        "status": "ok",
        "speakers": len(result),
        "total_samples": total,
        "samples": result,
    }


# ── Upload custom voice sample ─────────────────────────


@router.post("/upload-sample")
async def upload_voice_sample(
    audio_path: str = Form(...),
    speaker: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    """Upload an external audio file as a voice sample for a speaker."""
    p = AudioPaths.from_audio(audio_path)
    samples_dir = p.voice_samples_dir

    # Find next available index for this speaker
    existing = sorted(samples_dir.glob(f"{speaker}_*.wav"))
    next_idx = len(existing)
    out_path = samples_dir / f"{speaker}_{next_idx:02d}.wav"

    # Save uploaded file to a temp location, then convert to 16kHz mono WAV
    import tempfile

    suffix = Path(file.filename or "upload.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_path,
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(
            400, f"Failed to convert audio: {exc.stderr.decode()[:200]}"
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    import soundfile as sf

    info = sf.info(str(out_path))
    return {
        "file": str(out_path),
        "duration": info.duration,
        "text": f"[uploaded: {file.filename}]",
        "speaker": speaker,
    }


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
    force: bool = False
    only_speakers: list[str] | None = None

    @field_validator("max_chunk_duration")
    @classmethod
    def max_chunk_duration_positive(cls, v: float) -> float:
        """Validate that max_chunk_duration is a positive number."""
        if v <= 0:
            raise ValueError("max_chunk_duration must be positive")
        return v


@router.post("/generate", response_model=TaskResponse)
async def generate_tts(req: GenerateRequest) -> TaskResponse:
    """Generate TTS audio for all segments as a background task.

    Supports incremental generation: previously generated segments are skipped
    if their text and voice sample haven't changed.  Use ``force=true`` to
    regenerate everything, or ``only_speakers`` to target specific speakers.
    """

    def run_generate(progress_cb, req_data):
        """Load source segments, run incremental TTS generation, and save a manifest."""
        from podcodex.core._utils import free_vram
        from podcodex.core.synthesize import (
            _sample_key,
            _text_hash,
            build_clone_prompts,
            generate_segment,
            load_manifest,
            load_tts_model,
            load_voice_samples,
            save_manifest,
            segment_is_current,
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

        segments_dir = p.tts_segments_dir
        force = req_data.force
        only_speakers = req_data.only_speakers

        # Load manifest for incremental generation
        manifest = (
            load_manifest(segments_dir)
            if not force
            else {"model": None, "language": None, "segments": {}}
        )

        # First pass: figure out which segments actually need generation
        to_generate: list[
            tuple[int, dict, Path, str]
        ] = []  # (index, seg, path, sample_name)
        generated = []
        reused = 0
        total = len(segments)

        for i, seg in enumerate(segments):
            speaker = seg.get("speaker", "UNK")
            text = seg.get("text", "").strip()
            if speaker == "[BREAK]" or not text:
                continue

            filename = f"{i:04d}_{speaker}.wav"
            output_path = segments_dir / filename
            sample_name = _sample_key(voice_samples, speaker)

            # Skip if only regenerating specific speakers
            if only_speakers and speaker not in only_speakers:
                if output_path.exists():
                    generated.append(
                        (
                            i,
                            {
                                **seg,
                                "audio_file": str(output_path),
                                "sample_rate": SAMPLE_RATE,
                            },
                        )
                    )
                continue

            # Check manifest — reuse if segment is still valid
            if (
                not force
                and output_path.exists()
                and segment_is_current(
                    manifest,
                    filename,
                    text,
                    speaker,
                    sample_name,
                    req_data.model_size,
                    req_data.language,
                )
            ):
                generated.append(
                    (
                        i,
                        {
                            **seg,
                            "audio_file": str(output_path),
                            "sample_rate": SAMPLE_RATE,
                        },
                    )
                )
                reused += 1
                continue

            to_generate.append((i, seg, output_path, sample_name))

        # If everything is cached, skip model loading entirely
        if not to_generate:
            manifest["model"] = req_data.model_size
            manifest["language"] = req_data.language
            save_manifest(segments_dir, manifest)
            progress_cb(1.0, "All segments up to date")
            # result_segs = [s for _, s in sorted(generated)]
            return {"count": 0, "reused": reused, "skipped": total - len(generated)}

        progress_cb(
            0.1, f"Loading TTS model ({len(to_generate)} segments to generate)..."
        )
        model = load_tts_model(model_size=req_data.model_size)
        clone_prompts = build_clone_prompts(model, voice_samples)

        # Second pass: generate only stale/missing segments
        for i, seg, output_path, sample_name in to_generate:
            speaker = seg.get("speaker", "UNK")
            text = seg.get("text", "").strip()
            filename = output_path.name

            frac = 0.1 + 0.85 * (i / total)
            progress_cb(frac, f"Segment {i + 1}/{total} ({speaker})")

            result = generate_segment(
                model,
                seg,
                clone_prompts,
                output_path,
                language=req_data.language,
                max_chunk_duration=req_data.max_chunk_duration,
            )
            if result:
                generated.append((i, result))
                manifest["segments"][filename] = {
                    "speaker": speaker,
                    "voice_sample": sample_name,
                    "text_hash": _text_hash(text),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

        # Save manifest
        manifest["model"] = req_data.model_size
        manifest["language"] = req_data.language
        save_manifest(segments_dir, manifest)

        progress_cb(0.98, "Releasing GPU memory...")
        free_vram(model)

        new_count = len(to_generate)
        return {"count": new_count, "reused": reused, "skipped": total - len(generated)}

    return submit_task("generate_tts", req.audio_path, run_generate, req)


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
    # Convert Path objects for JSON, include manifest metadata
    return [
        {
            "speaker": seg.get("speaker", ""),
            "text": seg.get("text", ""),
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "audio_file": str(seg["audio_file"]),
            "duration": seg.get("end", 0) - seg.get("start", 0),
            "voice_sample": seg.get("voice_sample", ""),
            "generated_at": seg.get("generated_at", ""),
        }
        for seg in generated
    ]


# ── Assembly ─────────────────────────────────────────────


class AssembleRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    strategy: str = "original_timing"
    silence_duration: float = 0.5

    @field_validator("silence_duration")
    @classmethod
    def silence_duration_non_negative(cls, v: float) -> float:
        """Validate that silence_duration is zero or positive."""
        if v < 0:
            raise ValueError("silence_duration must be non-negative")
        return v


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
