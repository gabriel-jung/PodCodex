"""
podcodex.core.transcribe — Transcription/diarization pipeline using WhisperX.

All functions are idempotent: if matching versions already exist, they are
reloaded without recomputing (unless force=True).

Versioned outputs (all tracked in pipeline.db)::

    transcript/
      {id}.json                        — final transcript segments
      segments/{id}.parquet            — raw WhisperX segments
      diarization/{id}.parquet         — pyannote speaker timeline
      diarized_segments/{id}.parquet   — segments with SPEAKER_XX assigned
    {stem}.speaker_map.json            — {SPEAKER_00: "Claude", ...} (filled by UI)
"""

import os
import warnings
from pathlib import Path

from loguru import logger

from podcodex.core._utils import (
    BREAK_SPEAKER,
    NARRATOR_SPEAKER,
    SAMPLE_RATE,
    UNKNOWN_SPEAKERS,
    AudioPaths,
    check_vram,
    free_vram,
    read_json,
    write_json,
)
from podcodex.core.pipeline_db import mark_step
from podcodex.core.versions import (
    get_latest_provenance,
    has_matching_version,
    has_version,
    load_latest,
    save_version,
)

# Suppress known harmless warnings from third-party libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")


def processing_status(
    audio_path: Path | str,
    output_dir: str | Path | None = None,
) -> dict[str, bool]:
    """Return the processing state of an audio file.

    Checks the version DB for each pipeline sub-step.

    Args:
        audio_path: Source audio file.
        output_dir: Output directory override (see ``AudioPaths.output_dir``
            for resolution rules).

    Returns:
        Dict with boolean flags: ``transcribed``, ``diarized``, ``assigned``,
        ``mapped``, ``exported``.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return {
        "transcribed": has_version(p.base, "segments"),
        "diarized": has_version(p.base, "diarization"),
        "assigned": has_version(p.base, "diarized_segments"),
        "mapped": p.speaker_map.exists(),
        "exported": has_version(p.base, "transcript"),
    }


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────


def get_device() -> tuple[str, str]:
    """Auto-detect best device and compute type.

    Note: MPS (Apple Silicon) is not yet supported by WhisperX, falls back to CPU.

    Returns:
        (device, compute_type) — e.g. ("cuda", "float16") or ("cpu", "int8")
    """
    import torch

    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


# ──────────────────────────────────────────────
# STEP 1 — Transcription
# ──────────────────────────────────────────────


def transcribe_file(
    audio_path: Path | str,
    model_size: str = "large-v3",
    language: str = "fr",
    batch_size: int = 4,
    compute_type: str | None = None,
    device: str | None = None,
    force: bool = False,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Transcribe an audio file with WhisperX + phonetic alignment.
    Saves segments.parquet + segments.meta.json in output_dir.

    Args:
        audio_path   : source audio file
        model_size   : Whisper model size (default "large-v3")
        language     : ISO language code (default "fr")
        batch_size   : transcription batch size (default 4)
        compute_type : "float16", "int8", etc. (auto-detected if None)
        device       : "cuda" or "cpu" (auto-detected if None)
        force        : re-run even if output files already exist
        output_dir   : output directory (see AudioPaths.output_dir for resolution rules)

    Returns:
        dict with keys 'segments', 'language', 'duration', 'num_segments'
    """
    import whisperx

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    match_params = {"model": model_size}
    if language:
        match_params["language"] = language
    if not force and has_matching_version(p.base, "segments", match_params):
        logger.info("[SKIP] Matching segments version already exists")
        return load_segments(audio_path, output_dir=output_dir)

    dev, ctype = get_device()
    device = device or dev
    compute_type = compute_type or ctype

    logger.info(f"Transcribing {p.audio_path.name} ({device}, {compute_type})")

    if device == "cuda":
        from podcodex.core.constants import WHISPER_VRAM_MB

        check_vram(f"whisper ({model_size})", WHISPER_VRAM_MB.get(model_size, 512))

    audio = whisperx.load_audio(str(p.audio_path))

    from podcodex.core.cache import get_hf_cache_dir

    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        language=language or None,
        download_root=str(get_hf_cache_dir()),
    )
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    del model
    free_vram()

    # Use detected language when none was specified
    detected_lang = result.get("language") or language
    model_a, metadata = whisperx.load_align_model(
        language_code=detected_lang, device=device
    )
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    del model_a, metadata
    free_vram()

    segments = result["segments"]
    duration = float(audio.shape[0]) / SAMPLE_RATE
    meta = {
        "language": detected_lang,
        "duration": duration,
        "num_segments": len(segments),
    }

    provenance = {
        "step": "segments",
        "type": "raw",
        "model": model_size,
        "params": {
            "language": detected_lang,
            "batch_size": batch_size,
            "duration": duration,
        },
    }
    save_version(p.base, "segments", segments, provenance)

    logger.success(f"Transcription done — {len(segments)} segments")
    return {"segments": segments, **meta}


def _load_versioned(
    audio_path: Path | str, step: str, output_dir: str | Path | None = None
) -> tuple[list, dict]:
    """Load latest versioned data and its provenance params.

    Returns:
        (data, params) tuple where data is the loaded segments/speakers list
        and params is the provenance params dict.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    data = load_latest(p.base, step)
    if data is None:
        raise FileNotFoundError(f"No {step} version found")
    prov = get_latest_provenance(p.base, step) or {}
    return data, prov.get("params", {})


def load_segments(audio_path: Path | str, output_dir: str | Path | None = None) -> dict:
    """Load raw WhisperX segments from the version DB.

    Args:
        audio_path: Source audio file.
        output_dir: Output directory override (see ``AudioPaths.output_dir``
            for resolution rules).

    Returns:
        Dict with keys ``segments``, ``language``, ``duration``,
        ``num_segments``.
    """
    segments, params = _load_versioned(audio_path, "segments", output_dir)
    return {
        "segments": segments,
        "language": params.get("language", ""),
        "duration": params.get("duration", 0.0),
        "num_segments": len(segments),
    }


# ──────────────────────────────────────────────
# STEP 2 — Diarization
# ──────────────────────────────────────────────


def diarize_file(
    audio_path: Path | str,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    num_speakers: int | None = None,
    device: str | None = None,
    force: bool = False,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Diarize an audio file using whisperx.DiarizationPipeline (pyannote).
    Saves diarization.parquet + diarization.meta.json in output_dir.

    Args:
        audio_path   : source audio file
        hf_token     : HuggingFace token for pyannote (reads HF_TOKEN env var if None)
        min_speakers : minimum expected speakers (optional)
        max_speakers : maximum expected speakers (optional)
        num_speakers : exact number of speakers, if known (optional)
        device       : "cuda" or "cpu" (auto-detected if None)
        force        : re-run even if output files already exist
        output_dir   : output directory (see AudioPaths.output_dir for resolution rules)

    Returns:
        dict with keys 'speakers' (list of {start, end, speaker} dicts), 'num_speakers'
    """
    import whisperx

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    if not force and has_version(p.base, "diarization"):
        logger.info("[SKIP] Diarization version already exists")
        return load_diarization(audio_path, output_dir=output_dir)

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN not found. Set the HF_TOKEN environment variable or pass hf_token=."
        )

    dev, _ = get_device()
    device = device or dev

    logger.info(f"Diarizing {p.audio_path.name}")

    if device == "cuda":
        from podcodex.core.constants import DIARIZATION_VRAM_MB

        check_vram("diarization", DIARIZATION_VRAM_MB)

    from podcodex.core.cache import get_hf_cache_dir
    from whisperx.diarize import DiarizationPipeline

    get_hf_cache_dir()  # ensure HF_HOME is set before pyannote downloads
    audio = whisperx.load_audio(str(p.audio_path))
    pipeline = DiarizationPipeline(token=token, device=device)
    diarize_segments = pipeline(
        audio,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    del pipeline
    free_vram()

    df = diarize_segments.reset_index(drop=True)
    if "segment" in df.columns:
        speakers = [
            {
                "start": row["segment"].start,
                "end": row["segment"].end,
                "speaker": row["speaker"],
            }
            for _, row in df.iterrows()
        ]
    else:
        speakers = df[["start", "end", "speaker"]].to_dict("records")

    unique = sorted({s["speaker"] for s in speakers})
    meta = {"num_speakers": len(unique), "speakers_found": unique}

    provenance = {
        "step": "diarization",
        "type": "raw",
        "params": {
            "num_speakers": num_speakers,
            "speakers_found": unique,
        },
    }
    save_version(p.base, "diarization", speakers, provenance)

    logger.success(f"Diarization done — {len(unique)} speakers")
    return {"speakers": speakers, **meta}


def load_diarization(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> dict:
    """Load diarization from the version DB.

    Args:
        audio_path: Source audio file.
        output_dir: Output directory override (see ``AudioPaths.output_dir``
            for resolution rules).

    Returns:
        Dict with keys ``speakers`` (list of {start, end, speaker} dicts)
        and ``num_speakers``.
    """
    speakers, params = _load_versioned(audio_path, "diarization", output_dir)
    return {
        "speakers": speakers,
        "num_speakers": params.get(
            "num_speakers", len({s.get("speaker") for s in speakers})
        ),
        "speakers_found": params.get(
            "speakers_found", sorted({s.get("speaker", "") for s in speakers})
        ),
    }


# ──────────────────────────────────────────────
# STEP 3 — Speaker assignment
# ──────────────────────────────────────────────


def assign_speakers(
    audio_path: Path | str,
    force: bool = False,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """
    Assign SPEAKER_XX labels to segments via whisperx.assign_word_speakers.
    Requires transcribe_file() and diarize_file() to have already run.
    Saves diarized_segments.parquet in output_dir.

    Args:
        audio_path : source audio file
        force      : re-run even if output file already exists
        output_dir : output directory (see AudioPaths.output_dir for resolution rules)

    Returns:
        List of segments with 'speaker' key.
    """
    import pandas as pd
    import whisperx

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    if not force and has_version(p.base, "diarized_segments"):
        logger.info("[SKIP] Diarized segments version already exists")
        return load_diarized_segments(audio_path, output_dir=output_dir)

    def _has_timestamps(d: dict) -> bool:
        """Return True if the dict has non-None ``start`` and ``end`` keys."""
        return d.get("start") is not None and d.get("end") is not None

    diarization = load_diarization(audio_path, output_dir=output_dir)
    df_diarize = pd.DataFrame(diarization["speakers"]).dropna(subset=["start", "end"])

    transcription = load_segments(audio_path, output_dir=output_dir)
    for s in transcription["segments"]:
        if "words" in s and s["words"] is not None:
            s["words"] = [w for w in s["words"] if _has_timestamps(w)]
    filtered = [s for s in transcription["segments"] if _has_timestamps(s)]

    result = whisperx.assign_word_speakers(df_diarize, {"segments": filtered})
    segments = result["segments"]

    provenance = {
        "step": "diarized_segments",
        "type": "raw",
    }
    save_version(p.base, "diarized_segments", segments, provenance)
    logger.success(f"Assignment done — {len(segments)} segments")
    return segments


def load_diarized_segments(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> list[dict]:
    """Load diarized segments from the version DB.

    Args:
        audio_path: Source audio file.
        output_dir: Output directory override (see ``AudioPaths.output_dir``
            for resolution rules).

    Returns:
        List of segment dicts with ``speaker`` key assigned.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    segments = load_latest(p.base, "diarized_segments")
    if segments is None:
        raise FileNotFoundError("No diarized_segments version found")
    return segments


# ──────────────────────────────────────────────
# STEP 4 — Speaker map
# ──────────────────────────────────────────────


def load_speaker_map(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> dict[str, str]:
    """Load SPEAKER_XX → name mapping.

    Args:
        audio_path: Source audio file.
        output_dir: Output directory override (see ``AudioPaths.output_dir``
            for resolution rules).

    Returns:
        Dict mapping diarization labels (e.g. ``"SPEAKER_00"``) to display
        names (e.g. ``"Alice"``).  Returns an empty dict if the speaker map
        file does not exist yet.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    if not p.speaker_map.exists():
        return {}
    return read_json(p.speaker_map)


def save_speaker_map(
    audio_path: Path | str,
    mapping: dict[str, str],
    output_dir: str | Path | None = None,
) -> None:
    """Save SPEAKER_XX → human name mapping.

    The mapping keys must match the speaker labels from diarization
    (e.g. ``"SPEAKER_00"``, ``"SPEAKER_01"``). Values are display names
    used in the exported transcript.

    Args:
        audio_path: Source audio file.
        mapping: Dict mapping diarization labels to human-readable names.
        output_dir: Output directory override (see ``AudioPaths.output_dir``
            for resolution rules).

    Example::

        save_speaker_map(audio, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    write_json(p.speaker_map, mapping)
    logger.info(f"Speaker map saved → {p.speaker_map.name}")


# ──────────────────────────────────────────────
# STEP 5 — Final export
# ──────────────────────────────────────────────


def export_transcript(
    audio_path: Path | str,
    output_dir: str | Path | None = None,
    show: str = "",
    episode: str = "",
    diarized: bool = True,
    clean: bool = False,
    provenance: dict | None = None,
) -> list[dict]:
    """
    Generate the final JSON transcript with resolved speaker names.

    When *diarized* is True (default), requires diarized_segments.parquet +
    speaker_map.json.  When False, uses raw WhisperX segments and assigns
    :data:`NARRATOR_SPEAKER` to every segment.

    Saves a new version via the version DB.

    The file format is:
        {"meta": {show, episode, diarized, speakers, duration, word_count},
         "segments": [...]}

    Args:
        audio_path : source audio file
        output_dir : directory relative to audio_path for outputs
        show       : podcast show name (stored in meta, defaults to "")
        episode    : episode name (stored in meta, defaults to "")
        diarized   : whether diarization was performed (default True)
        provenance : optional version metadata dict for archiving

    Returns:
        List of final segments [{start, end, speaker, text}]
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    if diarized:
        segments = load_diarized_segments(audio_path, output_dir=output_dir)
        mapping = load_speaker_map(audio_path, output_dir=output_dir)
    else:
        raw = load_segments(audio_path, output_dir=output_dir)
        segments = raw["segments"]
        mapping = {}

    resolved = [
        {
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "speaker": (
                mapping.get(seg.get("speaker") or "", seg.get("speaker") or "")
                or "UNKNOWN"
                if diarized
                else NARRATOR_SPEAKER
            ),
            "text": str(seg.get("text", "")).strip(),
        }
        for seg in segments
    ]

    def _build_meta(segs):
        return {
            "show": show,
            "episode": episode,
            "diarized": diarized,
            "speakers": sorted({s["speaker"] for s in segs}),
            "duration": round(max((s["end"] for s in segs), default=0.0), 3),
            "word_count": sum(len(s["text"].split()) for s in segs),
        }

    def _make_prov(
        meta: dict, *, ptype: str = "raw", extra_params: dict | None = None
    ) -> dict:
        base_params = (provenance or {}).get("params", {})
        return {
            **(provenance or {"step": "transcript"}),
            "type": ptype,
            "params": {**base_params, **(extra_params or {}), "meta": meta},
        }

    # Always save raw transcript
    raw_prov = _make_prov(_build_meta(resolved))
    save_version(p.base, "transcript", resolved, raw_prov)
    logger.success(f"Export done — {len(resolved)} segments")

    # If clean, also save a filtered version
    if clean:
        resolved = clean_transcript(resolved, remove_unknown_speakers=diarized)
        save_version(
            p.base,
            "transcript",
            resolved,
            _make_prov(
                _build_meta(resolved), ptype="edited", extra_params={"clean": True}
            ),
        )
        logger.success(f"Clean export — {len(resolved)} segments (filtered)")

    mark_step(
        p.show_dir, p.base.name, transcribed=True, provenance={"transcript": raw_prov}
    )
    return resolved


def _load_transcript_file(path: Path) -> dict:
    """Load a transcript JSON file, normalizing old (list) and new (dict) formats."""
    data = read_json(path)
    if isinstance(data, dict):
        return data
    return {"meta": {}, "segments": data}


def load_transcript_full(
    audio_path: Path | str | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Load the final transcript with metadata.

    Tries the version DB first, then falls back to legacy files.

    Args:
        audio_path: Source audio file.
        output_dir: Output directory override (see ``AudioPaths.output_dir``
            for resolution rules).

    Returns:
        Dict with keys ``meta`` (show, episode, speakers, duration,
        word_count) and ``segments`` (list of segment dicts).  If loaded
        from an old list-format file, ``meta`` will be an empty dict.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    # Try version DB
    segments = load_latest(p.base, "transcript")
    if segments is not None:
        prov = get_latest_provenance(p.base, "transcript") or {}
        meta = prov.get("params", {}).get("meta", {})
        return {"meta": meta, "segments": segments}
    # Legacy fallback
    return _load_transcript_file(p.transcript_best)


def load_transcript(
    audio_path: Path | str | None = None,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Load the final transcript segments as a plain list.

    Convenience wrapper around :func:`load_transcript_full` that returns
    only the segment list, discarding metadata.

    Args:
        audio_path: Source audio file.
        output_dir: Output directory override (see ``AudioPaths.output_dir``
            for resolution rules).

    Returns:
        List of segment dicts (each with ``speaker``, ``start``, ``end``,
        ``text``).
    """
    return load_transcript_full(audio_path, output_dir=output_dir)["segments"]


# ──────────────────────────────────────────────
# Segment analysis
# ──────────────────────────────────────────────


# Segments assigned to this name are excluded by clean_transcript().
REMOVE_SPEAKERS = {"[remove]"}

# Speech density thresholds (chars/s) for flagging hallucinations.
MIN_DENSITY = 2.0
MAX_DENSITY = 75.0


def segment_speech_density(seg: dict) -> float | None:
    """Return chars/second for a segment, or None if duration is too short.

    Args:
        seg: Segment dict with ``text``, ``start``, and ``end`` keys.

    Returns:
        Speech density as characters per second, or ``None`` when segment
        duration is below 0.5 s (too short for a meaningful measurement).
    """
    text = str(seg.get("text", "")).strip()
    dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
    if dur < 0.5:
        return None
    return len(text) / dur


def is_segment_flagged(seg: dict, diarized: bool = True) -> bool:
    """Return True if a segment is suspicious and should be reviewed or removed.

    A segment is flagged when:

    - speaker is missing or an unresolved placeholder (UNKNOWN, UNK, ...)
    - speaker is a reserved remove marker ([remove])
    - speech density is abnormal (< 2 or > 75 chars/s), indicating music, noise,
      or a Whisper hallucination artifact

    When *diarized* is False, speaker-based checks are skipped (only density
    is used) since all segments share a generic narrator label.

    Args:
        seg: Segment dict with ``speaker``, ``text``, ``start``, ``end``.
        diarized: Whether diarization was performed.  When False, only
            density-based flagging is applied.

    Returns:
        True if the segment should be flagged for review.
    """
    speaker = seg.get("speaker", "")
    if speaker == BREAK_SPEAKER:
        return False
    if diarized:
        if not speaker or speaker in UNKNOWN_SPEAKERS:
            return True
        if speaker in REMOVE_SPEAKERS:
            return True
    density = segment_speech_density(seg)
    return density is not None and (density < MIN_DENSITY or density > MAX_DENSITY)


def clean_transcript(
    segments: list[dict],
    *,
    remove_unknown_speakers: bool = True,
    remove_abnormal_density: bool = True,
) -> list[dict]:
    """Remove flagged segments from a transcript.

    Args:
        segments                : list of segment dicts (from load_transcript)
        remove_unknown_speakers : drop segments with missing/unresolved speaker
        remove_abnormal_density : drop segments outside MIN_DENSITY..MAX_DENSITY chars/s

    Returns:
        Filtered list of segments.
    """
    result = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        if speaker == BREAK_SPEAKER:
            result.append(seg)
            continue
        if speaker in REMOVE_SPEAKERS:
            continue
        if remove_unknown_speakers and (not speaker or speaker in UNKNOWN_SPEAKERS):
            continue
        if remove_abnormal_density:
            density = segment_speech_density(seg)
            if density is not None and (density < MIN_DENSITY or density > MAX_DENSITY):
                continue
        result.append(seg)
    logger.debug(f"clean_transcript: {len(segments)} → {len(result)} segments")
    return result


def save_transcript(
    audio_path: Path | str,
    segments: list[dict],
    output_dir: str | Path | None = None,
    provenance: dict | None = None,
) -> str:
    """Save transcript segments (version DB + pipeline DB). Returns the version id."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    version_id = save_version(p.base, "transcript", segments, provenance)
    logger.info(f"Transcript saved → {p.base.name} ({len(segments)} segments)")
    prov_update = {"transcript": provenance} if provenance else {}
    mark_step(p.show_dir, p.base.name, transcribed=True, provenance=prov_update)
    return version_id
