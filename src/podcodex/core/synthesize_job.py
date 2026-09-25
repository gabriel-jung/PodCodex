"""Subprocess entry points for the synthesize pipeline step.

Keeps TTS models (torch) out of the FastAPI process. One entry:

* ``run_generate`` — run incremental TTS generation with a manifest.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


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
        BREAK_SPEAKER,
        SAMPLE_RATE,
        check_vram,
        fill_narrator_speaker,
        free_vram,
        real_speakers,
        seg_key,
        tts_segment_filename,
    )
    from podcodex.core.constants import TTS_VRAM_MB
    from podcodex.core.synthesize import (
        _sample_key,
        build_clone_prompts,
        empty_manifest,
        generate_segment,
        load_manifest,
        load_tts_model,
        load_voice_samples,
        record_segment,
        save_manifest,
        segment_is_current,
    )
    from podcodex.core.source import load_synth_source

    progress_cb(0.0, "Loading source segments...")
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    source, speaker_map = load_synth_source(
        audio_path, output_dir, source_version_id, source_lang
    )
    segments = source.segments

    # UI scope filter: drop every segment the user unchecked in the source
    # picker. Uses the shared seg_key helper so keys agree with the frontend.
    # Run before the narrator fill so we don't allocate dicts for dropped segs.
    if keep_segment_keys is not None:
        wanted = set(keep_segment_keys)
        segments = [seg for seg in segments if seg_key(seg) in wanted]
        if not segments:
            raise ValueError("Selection dropped every segment, nothing to synthesize")

    # Legacy transcripts (e.g. YouTube subtitle imports pre-parser-default)
    # carry empty speaker labels. Map them to NARRATOR_SPEAKER so a single
    # uploaded voice sample under that key applies to every segment.
    segments = fill_narrator_speaker(segments)

    progress_cb(0.05, "Loading voice samples...")

    speakers = real_speakers(segments)
    voice_samples = load_voice_samples(
        str(p.base.parent), speakers, speaker_map=speaker_map
    )
    if not voice_samples:
        raise ValueError("No voice samples found. Extract voices first.")

    segments_dir = p.ensure_tts_segments_dir()
    if force:
        # Persist the reset now: the old entries on disk would otherwise keep
        # vouching for files this run is about to overwrite.
        manifest = empty_manifest()
        save_manifest(segments_dir, manifest)
    else:
        manifest = load_manifest(segments_dir)

    to_generate: list[tuple[int, dict, Path, str]] = []
    generated: list[tuple[int, dict]] = []
    reused = 0
    total = len(segments)

    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "UNK")
        text = seg.get("text", "").strip()
        if speaker == BREAK_SPEAKER or not text:
            continue

        filename = tts_segment_filename(seg)
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
                manifest, filename, text, sample_name, model_size, language
            )
        ):
            generated.append(
                (i, {**seg, "audio_file": str(out_path), "sample_rate": SAMPLE_RATE})
            )
            reused += 1
            continue

        to_generate.append((i, seg, out_path, sample_name))

    if not to_generate:
        progress_cb(1.0, "All segments up to date")
        return {"count": 0, "reused": reused, "skipped": total - len(generated)}

    progress_cb(0.1, f"Loading TTS model ({len(to_generate)} segments to generate)...")
    check_vram(f"TTS ({model_size})", TTS_VRAM_MB.get(model_size, 4000))
    model = load_tts_model(model_size=model_size)
    clone_prompts = build_clone_prompts(model, voice_samples)

    # The manifest is what lets finished segments be reused. Saved at most
    # every few seconds (a full rewrite per segment is quadratic in writes on
    # a long episode, and each one re-syncs a synced show folder) and once
    # more on the way out, crash included. The runner SIGTERMs a cancelled
    # child after a few seconds, so a kill loses at most the last interval.
    last_save = time.monotonic()
    try:
        for i, seg, out_path, sample_name in to_generate:
            if cancelled():
                break
            speaker = seg.get("speaker", "UNK")
            progress_cb(
                0.1 + 0.85 * (i / total), f"Segment {i + 1}/{total} ({speaker})"
            )
            result = generate_segment(
                model,
                seg,
                clone_prompts,
                out_path,
                language=language,
                max_chunk_duration=max_chunk_duration,
                cancelled=cancelled,
            )
            if not result:
                continue
            generated.append((i, result))
            record_segment(
                manifest,
                out_path.name,
                speaker=speaker,
                text=seg.get("text", "").strip(),
                voice_sample_name=sample_name,
                model_size=model_size,
                language=language,
            )
            if time.monotonic() - last_save >= _MANIFEST_SAVE_INTERVAL_S:
                save_manifest(segments_dir, manifest)
                last_save = time.monotonic()
    finally:
        save_manifest(segments_dir, manifest)

    # Skip cleanup on cancel: del model + gc can stall 30-60s; OS reaps faster.
    if not cancelled():
        progress_cb(0.98, "Releasing GPU memory...")
        del model
        free_vram()

    new_count = len(to_generate)
    return {"count": new_count, "reused": reused, "skipped": total - len(generated)}


_MANIFEST_SAVE_INTERVAL_S = 5.0
