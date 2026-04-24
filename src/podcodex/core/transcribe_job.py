"""Subprocess entry points for the transcribe pipeline step.

Keeps Whisper / pyannote / torch out of the FastAPI process. Invoked via
``podcodex.api.subprocess_runner``. Two entry functions:

* ``run`` — single-episode flow used by ``POST /transcribe/start``.
* ``run_for_batch`` — multi-episode flow that honors version-match skipping
  and cooperative cancel between sub-steps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def run(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    audio_path: str,
    output_dir: str | None,
    model_size: str,
    language: str,
    batch_size: int | None,
    force: bool,
    diarize: bool,
    hf_token: str | None,
    num_speakers: int | None,
    show: str,
    episode: str,
    clean: bool,
) -> dict[str, Any]:
    """Single-episode transcribe. Returns ``{count}``."""
    from podcodex.api.routes._helpers import build_provenance, transcribe_prov_params
    from podcodex.core._utils import default_batch_size
    from podcodex.core.transcribe import (
        assign_speakers,
        diarize_file,
        export_transcript,
        transcribe_file,
    )

    effective_batch = batch_size or default_batch_size()
    progress_cb(0.0, f"Transcribing audio (batch_size={effective_batch})...")
    transcribe_file(
        audio_path,
        model_size=model_size,
        language=language or None,
        batch_size=effective_batch,
        force=force,
        output_dir=output_dir,
    )

    if diarize:
        progress_cb(0.25, "Diarizing speakers...")
        diarize_file(
            audio_path,
            hf_token=hf_token,
            num_speakers=num_speakers,
            force=force,
            output_dir=output_dir,
        )
        progress_cb(0.5, "Assigning speakers...")
        assign_speakers(audio_path, force=force, output_dir=output_dir)

    progress_cb(0.75, "Exporting transcript...")
    provenance = build_provenance(
        "transcript",
        model=model_size,
        params=transcribe_prov_params(
            diarize,
            source="whisper",
            model=model_size,
            language=language or None,
            batch_size=batch_size,
            num_speakers=num_speakers,
            clean=clean,
        ),
    )
    segments = export_transcript(
        audio_path,
        output_dir=output_dir,
        show=show,
        episode=episode,
        diarized=diarize,
        clean=clean,
        provenance=provenance,
    )
    return {"count": len(segments)}


def run_for_batch(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    audio_path: str,
    stem: str,
    show_name: str,
    model_size: str,
    language: str,
    batch_size: int | None,
    diarize: bool,
    hf_token: str | None,
    num_speakers: int | None,
    clean: bool,
    force: bool,
) -> dict[str, Any]:
    """Batch-mode transcribe — version-match skipping + cancel checks.

    Returns ``{"did_work": bool}``. ``did_work`` is False when every
    sub-step was skipped (version already matches).
    """
    from podcodex.api.routes._helpers import build_provenance, transcribe_prov_params
    from podcodex.core._utils import AudioPaths, default_batch_size
    from podcodex.core.transcribe import (
        assign_speakers,
        diarize_file,
        export_transcript,
        transcribe_file,
    )
    from podcodex.core.versions import has_matching_version, has_version

    p = AudioPaths.from_audio(audio_path)
    did_work = False
    effective_batch = batch_size or default_batch_size()

    transcript_params: dict[str, Any] = {"model": model_size, "diarize": diarize}
    if language:
        transcript_params["language"] = language
    if not force and has_matching_version(p.base, "transcript", transcript_params):
        return {"did_work": False}

    seg_params: dict[str, Any] = {"model": model_size}
    if language:
        seg_params["language"] = language

    new_segments = force or not has_matching_version(p.base, "segments", seg_params)
    if new_segments:
        did_work = True
        progress_cb(0.0, "Transcribing...")
        transcribe_file(
            audio_path,
            model_size=model_size,
            language=language or None,
            batch_size=effective_batch,
            force=force,
        )
        if cancelled():
            return {"did_work": did_work}

    new_diarization = False
    if not cancelled() and diarize:
        if force or not has_version(p.base, "diarization"):
            new_diarization = True
            did_work = True
            progress_cb(0.4, "Diarizing...")
            diarize_file(
                audio_path,
                hf_token=hf_token,
                num_speakers=num_speakers,
                force=force,
            )
            if cancelled():
                return {"did_work": did_work}

    if not cancelled() and diarize:
        if (
            force
            or new_segments
            or new_diarization
            or not has_version(p.base, "diarized_segments")
        ):
            did_work = True
            progress_cb(0.7, "Assigning speakers...")
            assign_speakers(audio_path, force=force)
            if cancelled():
                return {"did_work": did_work}

    if not cancelled():
        progress_cb(0.9, "Exporting transcript...")
        variants = [False, True] if diarize else [False]
        for diarized_flag in variants:
            if cancelled():
                return {"did_work": did_work}
            provenance = build_provenance(
                "transcript",
                model=model_size,
                params=transcribe_prov_params(
                    diarized_flag,
                    model=model_size,
                    language=language or None,
                    batch_size=effective_batch,
                    num_speakers=num_speakers,
                    clean=clean,
                ),
            )
            export_transcript(
                audio_path,
                show=show_name,
                episode=stem,
                diarized=diarized_flag,
                clean=clean,
                provenance=provenance,
            )
        did_work = True

    return {"did_work": did_work}
