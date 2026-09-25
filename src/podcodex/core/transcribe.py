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
    speaker_map/{id}.json              — {SPEAKER_00: "Claude", ...} (linked to diarization)
"""

import functools
import os
import warnings
from pathlib import Path

from loguru import logger

from podcodex.core._utils import (
    BREAK_SPEAKER,
    NARRATOR_SPEAKER,
    REMOVE_SPEAKER,
    SAMPLE_RATE,
    UNKNOWN_SPEAKERS,
    AudioPaths,
    check_vram,
    free_vram,
)
from podcodex.core.constants import DEFAULT_WHISPER_MODEL, DIARIZATION_MODEL
from podcodex.core.pipeline_db import mark_step
from podcodex.core.versions import (
    get_latest_provenance,
    has_matching_version,
    load_latest,
    load_latest_speaker_map,
    load_latest_with_meta,
    save_speaker_map_version,
    save_version,
)

# Suppress known harmless warnings from third-party libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")


# ──────────────────────────────────────────────
# STEP 1 — Transcription
# ──────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _decode(path: str, mtime_ns: int, size: int):
    import whisperx

    return whisperx.load_audio(path)


def release_decoded_audio() -> None:
    """Drop the cached waveform (about 230 MB per hour of audio).

    Called once nothing later in the run needs it: after diarization, or
    after transcription when no diarization follows.
    """
    _decode.cache_clear()


def _decoded_audio(path: Path):
    """The episode decoded by ffmpeg, once per run.

    Transcribe and diarize both need the full waveform and run back to back
    in the same worker; each used to spawn ffmpeg and hold its own copy. One
    entry: a worker handles one episode, and the key (path, mtime, size)
    drops it the moment the file changes.
    """
    st = Path(path).stat()
    return _decode(str(path), st.st_mtime_ns, st.st_size)


def segments_match_params(model_size: str, language: str | None) -> dict:
    """Params a ``segments`` version must carry to be reused for a run."""
    params: dict = {"model": model_size}
    if language:
        params["language"] = language
    return params


def diarization_match_params(num_speakers: int | None) -> dict:
    """Params a ``diarization`` version must carry to be reused for a run."""
    return {"num_speakers": num_speakers}


def transcribe_file(
    audio_path: Path | str,
    model_size: str = DEFAULT_WHISPER_MODEL,
    language: str = "en",
    batch_size: int = 4,
    force: bool = False,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Transcribe an audio file with WhisperX + phonetic alignment.
    Saves a new ``segments`` version (transcript/segments/<id>.parquet, indexed in pipeline.db).

    Args:
        audio_path   : source audio file
        model_size   : Whisper model size (default DEFAULT_WHISPER_MODEL)
        language     : ISO language code (default "en")
        batch_size   : transcription batch size (default 4)
        force        : re-run even if the current segments already match
        output_dir   : output directory (see AudioPaths.output_dir for resolution rules)

    Returns:
        dict with keys 'segments', 'language', 'duration', 'num_segments'
    """
    import whisperx

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    match_params = segments_match_params(model_size, language)
    if not force and has_matching_version(
        p.base, "segments", match_params, current_only=True
    ):
        logger.info("[SKIP] Current segments version already matches")
        return load_segments(audio_path, output_dir=output_dir)

    from podcodex.core.device import resolve_device

    device, compute_type = resolve_device()

    logger.info(f"Transcribing {p.audio_path.name} ({device}, {compute_type})")

    if device == "cuda":
        from podcodex.core.constants import WHISPER_VRAM_MB

        check_vram(f"whisper ({model_size})", WHISPER_VRAM_MB.get(model_size, 512))

    audio = _decoded_audio(p.audio_path)

    from podcodex.core._hf_logging import timed_load
    from podcodex.core.cache import get_hf_cache_dir

    with timed_load(f"WhisperX {model_size} on {device}"):
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
    with timed_load(f"WhisperX align model ({detected_lang}) on {device}"):
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
    num_speakers: int | None = None,
    force: bool = False,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Diarize an audio file using whisperx.DiarizationPipeline (pyannote).
    Saves a new ``diarization`` version (transcript/diarization/<id>.parquet).

    Args:
        audio_path   : source audio file
        hf_token     : HuggingFace token for pyannote (reads HF_TOKEN env var if None)
        num_speakers : exact number of speakers, if known (optional)
        force        : re-run even if the current diarization already matches
        output_dir   : output directory (see AudioPaths.output_dir for resolution rules)

    Returns:
        dict with keys 'speakers' (list of {start, end, speaker} dicts), 'num_speakers'
    """

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    if not force and has_matching_version(
        p.base, "diarization", diarization_match_params(num_speakers), current_only=True
    ):
        logger.info(
            "[SKIP] Current diarization already has num_speakers={}",
            num_speakers,
        )
        return load_diarization(audio_path, output_dir=output_dir)

    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        raise ValueError(
            "HF_TOKEN not found. Set the HF_TOKEN environment variable or pass hf_token=."
        )
    if not token.startswith("hf_"):
        # pyannote silently drops a token without this prefix and the load
        # then fails as a gated-repo error, which points at the wrong fix.
        raise ValueError(
            "HF_TOKEN does not look like a Hugging Face access token (they "
            "start with 'hf_'). Create one at https://huggingface.co/settings/tokens."
        )

    from podcodex.core.device import resolve_device

    device, _ = resolve_device()

    logger.info(f"Diarizing {p.audio_path.name}")

    if device == "cuda":
        from podcodex.core.constants import DIARIZATION_VRAM_MB

        check_vram("diarization", DIARIZATION_VRAM_MB)

    from podcodex.core._hf_logging import timed_load
    from podcodex.core.cache import get_hf_hub_dir
    from whisperx.diarize import DiarizationPipeline

    audio = _decoded_audio(p.audio_path)
    with timed_load(f"pyannote DiarizationPipeline on {device}"):
        # Explicit model and cache: the model is the one provenance records,
        # and the cache holds even in a process whose huggingface_hub was
        # imported before the env vars were set.
        try:
            pipeline = DiarizationPipeline(
                model_name=DIARIZATION_MODEL,
                token=token,
                device=device,
                cache_dir=str(get_hf_hub_dir()),
            )
        except Exception as exc:
            raise _diarization_load_error(exc) from exc
    diarize_segments = pipeline(audio, num_speakers=num_speakers)
    del audio
    release_decoded_audio()

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
        "model": DIARIZATION_MODEL,
        "params": {
            "num_speakers": num_speakers,
            "speakers_found": unique,
        },
    }
    save_version(p.base, "diarization", speakers, provenance)

    logger.success(f"Diarization done — {len(unique)} speakers")
    return {"speakers": speakers, **meta}


def _diarization_load_error(exc: Exception) -> Exception:
    """A message that says what to do, for the Hub failures users hit.

    huggingface_hub's own errors are accurate but read as stack-trace noise
    in a task toast; everything else is returned unchanged.
    """
    try:
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            LocalEntryNotFoundError,
        )
    except ImportError:
        return exc
    url = f"https://huggingface.co/{DIARIZATION_MODEL}"
    if isinstance(exc, GatedRepoError):
        return RuntimeError(
            f"Diarization model access not granted: accept its conditions at {url} "
            "with the account that owns HF_TOKEN."
        )
    if isinstance(exc, LocalEntryNotFoundError):
        return RuntimeError(
            "Diarization model is not cached and Hugging Face could not be "
            "reached. Check the connection (or PODCODEX_HF_OFFLINE)."
        )
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401:
            return RuntimeError("HF_TOKEN was rejected by Hugging Face (401).")
        if status == 429:
            return RuntimeError("Hugging Face rate limit hit; retry in a few minutes.")
    return exc


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
    Saves a new ``diarized_segments`` version (transcript/diarized_segments/<id>.parquet).

    Args:
        audio_path : source audio file
        force      : re-run even if output file already exists
        output_dir : output directory (see AudioPaths.output_dir for resolution rules)

    Returns:
        List of segments with 'speaker' key.
    """
    import pandas as pd
    import whisperx

    from podcodex.core.versions import (
        diarized_segments_input_hash,
        diarized_segments_is_fresh,
    )

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    if not force and diarized_segments_is_fresh(p.base):
        logger.info(
            "[SKIP] Diarized segments already match current segments + diarization"
        )
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
        "model": "whisperx.assign_word_speakers",
        "input_hash": diarized_segments_input_hash(p.base),
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
    """Load SPEAKER_XX → name mapping from the version DB.

    Returns the speaker map whose ``input_hash`` matches the current
    speaker-label source (``diarized_segments`` for the diarized flow,
    ``segments`` for subtitle-imported transcripts). Returns ``{}`` when
    no matching map exists, so a stale mapping is never silently applied
    after re-diarization or re-import.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return load_latest_speaker_map(p.base)


def save_speaker_map(
    audio_path: Path | str,
    mapping: dict[str, str],
    output_dir: str | Path | None = None,
) -> None:
    """Save SPEAKER_XX → human name mapping as a versioned entry.

    *mapping* is composed onto the current map (see
    ``versions.save_speaker_map_version``), so a later rename never undoes an
    earlier one. The map is linked to its label source via ``input_hash``
    (the latest ``diarized_segments`` content_hash, or ``segments`` for
    subtitle-only transcripts); maps for older diarizations/imports are
    preserved so re-running upstream steps never destroys curated mappings.

    Example::

        save_speaker_map(audio, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    save_speaker_map_version(p.base, mapping)
    logger.info(f"Speaker map saved ({len(mapping)} entries) for {p.base.name}")


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

    When *diarized* is True (default), requires a ``diarized_segments``
    version and a speaker map.  When False, uses raw WhisperX segments and assigns
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

    def _resolve_speaker(raw) -> str:
        # NaN from parquet is a float that is truthy — coerce to "" first.
        if raw is None or (isinstance(raw, float) and raw != raw):
            raw = ""
        raw = str(raw)
        return mapping.get(raw, raw) or "UNKNOWN"

    resolved = [
        {
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "speaker": (
                _resolve_speaker(seg.get("speaker")) if diarized else NARRATOR_SPEAKER
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
    logger.success(
        f"Export done ({'diarized' if diarized else 'raw'}) — {len(resolved)} segments"
    )

    # If clean, also save a filtered version. It is the newer one, so it is
    # what every reader loads, and the status provenance must describe it.
    status_prov = raw_prov
    if clean:
        resolved = clean_transcript(resolved, remove_unknown_speakers=diarized)
        status_prov = _make_prov(
            _build_meta(resolved), ptype="validated", extra_params={"clean": True}
        )
        save_version(p.base, "transcript", resolved, status_prov)
        logger.success(f"Clean export, {len(resolved)} segments (filtered)")

    mark_step(
        p.show_dir,
        p.base.name,
        transcribed=True,
        provenance={"transcript": status_prov},
    )
    return resolved


def load_transcript_full(
    audio_path: Path | str | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Load the latest transcript version with metadata.

    Args:
        audio_path: Source audio file.
        output_dir: Output directory override (see ``AudioPaths.output_dir``
            for resolution rules).

    Returns:
        Dict with keys ``meta`` (show, episode, speakers, duration,
        word_count) and ``segments`` (list of segment dicts), or
        ``{"meta": {}, "segments": []}`` if no transcript version exists.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    found = load_latest_with_meta(p.base, "transcript")
    if found is None:
        return {"meta": {}, "segments": []}
    version, segments = found
    return {"meta": (version.get("params") or {}).get("meta", {}), "segments": segments}


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
REMOVE_SPEAKERS = {REMOVE_SPEAKER}

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
