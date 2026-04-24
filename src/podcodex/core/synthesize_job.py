"""Subprocess entry points for the synthesize pipeline step.

Keeps TTS models (torch) out of the FastAPI process. Two entries:

* ``run_extract`` — extract voice samples from an existing transcript.
* ``run_generate`` — run incremental TTS generation with a manifest.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


def run_extract(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    audio_path: str,
    output_dir: str | None,
    min_duration: float | None,
    max_duration: float | None,
    top_k: int,
) -> dict[str, Any]:
    """Extract speaker voice samples from a transcript."""
    from podcodex.core.synthesize import extract_voice_samples
    from podcodex.core.transcribe import load_transcript

    progress_cb(0.0, "Loading transcript...")
    segments = load_transcript(audio_path, output_dir=output_dir)
    if not segments:
        raise ValueError("No transcript found — transcribe first")

    progress_cb(0.1, "Extracting voice samples...")
    samples = extract_voice_samples(
        audio_path,
        segments,
        output_dir=output_dir,
        min_duration=min_duration,
        max_duration=max_duration,
        top_k=top_k,
    )
    total = sum(len(v) for v in samples.values())
    return {"speakers": len(samples), "total_samples": total}


def run_generate(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    audio_path: str,
    output_dir: str | None,
    source_lang: str,
    source_version_id: str | None,
    model_size: str,
    language: str,
    max_chunk_duration: float,
    force: bool,
    only_speakers: list[str] | None,
    keep_segment_keys: list[str] | None,
) -> dict[str, Any]:
    """Incremental TTS generation guided by a manifest."""
    from podcodex.core._utils import (
        AudioPaths,
        SAMPLE_RATE,
        check_vram,
        free_vram,
        normalize_lang,
        seg_key,
    )
    from podcodex.core.constants import TTS_VRAM_MB
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
    from podcodex.core.versions import (
        list_all_versions,
        load_latest as _load_latest,
        load_version,
    )
    from podcodex.api.routes._helpers import load_best_source

    progress_cb(0.0, "Loading source segments...")
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    segments: list[dict] | None = None
    if source_version_id:
        # User pinned a specific version via the source picker. Look up its
        # step (transcript / corrected / <lang>) and load it directly so
        # re-running against a non-latest version stays reproducible.
        meta = next(
            (v for v in list_all_versions(p.base) if v.get("id") == source_version_id),
            None,
        )
        if meta and meta.get("step"):
            try:
                segments = load_version(p.base, meta["step"], source_version_id)
            except FileNotFoundError as exc:
                logger.warning(
                    "Pinned source version {} unreadable ({}), falling back",
                    source_version_id,
                    exc,
                )
                segments = None
        else:
            logger.warning(
                "Pinned source version {} not found in DB, falling back",
                source_version_id,
            )
    if segments is None and source_lang:
        segments = _load_latest(p.base, normalize_lang(source_lang))
    if not segments:
        try:
            segments = load_best_source(audio_path, output_dir)
        except ValueError:
            pass
    if not segments:
        raise ValueError("No source segments found")

    # UI scope filter — drop every segment the user unchecked in the source
    # picker. Uses the shared seg_key helper so keys agree with the frontend.
    if keep_segment_keys is not None:
        wanted = set(keep_segment_keys)
        segments = [seg for seg in segments if seg_key(seg) in wanted]
        if not segments:
            raise ValueError("Selection dropped every segment, nothing to synthesize")

    progress_cb(0.05, "Loading voice samples...")
    from podcodex.core._utils import real_speakers

    speakers = real_speakers(segments)
    voice_samples = load_voice_samples(str(p.base.parent), speakers)
    if not voice_samples:
        raise ValueError("No voice samples found — extract voices first")

    segments_dir = p.tts_segments_dir
    manifest = (
        load_manifest(segments_dir)
        if not force
        else {"model": None, "language": None, "segments": {}}
    )

    to_generate: list[tuple[int, dict, Path, str]] = []
    generated: list[tuple[int, dict]] = []
    reused = 0
    total = len(segments)

    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "UNK")
        text = seg.get("text", "").strip()
        if speaker == "[BREAK]" or not text:
            continue

        filename = f"{i:04d}_{speaker}.wav"
        out_path = segments_dir / filename
        sample_name = _sample_key(voice_samples, speaker)

        if only_speakers and speaker not in only_speakers:
            if out_path.exists():
                generated.append(
                    (
                        i,
                        {
                            **seg,
                            "audio_file": str(out_path),
                            "sample_rate": SAMPLE_RATE,
                        },
                    )
                )
            continue

        if (
            not force
            and out_path.exists()
            and segment_is_current(
                manifest, filename, text, speaker, sample_name, model_size, language
            )
        ):
            generated.append(
                (i, {**seg, "audio_file": str(out_path), "sample_rate": SAMPLE_RATE})
            )
            reused += 1
            continue

        to_generate.append((i, seg, out_path, sample_name))

    if not to_generate:
        manifest["model"] = model_size
        manifest["language"] = language
        save_manifest(segments_dir, manifest)
        progress_cb(1.0, "All segments up to date")
        return {"count": 0, "reused": reused, "skipped": total - len(generated)}

    progress_cb(0.1, f"Loading TTS model ({len(to_generate)} segments to generate)...")
    check_vram(f"TTS ({model_size})", TTS_VRAM_MB.get(model_size, 4000))
    model = load_tts_model(model_size=model_size)
    clone_prompts = build_clone_prompts(model, voice_samples)

    for i, seg, out_path, sample_name in to_generate:
        if cancelled():
            break
        speaker = seg.get("speaker", "UNK")
        text = seg.get("text", "").strip()
        filename = out_path.name

        frac = 0.1 + 0.85 * (i / total)
        progress_cb(frac, f"Segment {i + 1}/{total} ({speaker})")

        result = generate_segment(
            model,
            seg,
            clone_prompts,
            out_path,
            language=language,
            max_chunk_duration=max_chunk_duration,
        )
        if result:
            generated.append((i, result))
            manifest["segments"][filename] = {
                "speaker": speaker,
                "voice_sample": sample_name,
                "text_hash": _text_hash(text),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    manifest["model"] = model_size
    manifest["language"] = language
    save_manifest(segments_dir, manifest)

    progress_cb(0.98, "Releasing GPU memory...")
    del model
    free_vram()

    new_count = len(to_generate)
    return {"count": new_count, "reused": reused, "skipped": total - len(generated)}
