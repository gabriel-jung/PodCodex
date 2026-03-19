"""
podcodex.core.transcribe — Transcription/diarization pipeline using WhisperX.

All functions are idempotent: if output files already exist, they are reloaded
without recomputing (unless force=True).

Files produced alongside the MP3:
    {stem}.segments.parquet          — raw whisperx segments
    {stem}.segments.meta.json        — language, duration, etc.
    {stem}.diarization.parquet       — raw speaker timeline
    {stem}.diarization.meta.json     — num speakers, etc.
    {stem}.diarized_segments.parquet — segments with SPEAKER_XX assigned
    {stem}.speaker_map.json          — {SPEAKER_00: "Claude", ...}  (filled by UI)
    {stem}.transcript.raw.json       — pipeline export (unvalidated)
    {stem}.transcript.json           — validated/edited transcript
"""

import json
import os
import subprocess
import warnings
from pathlib import Path

from loguru import logger

from podcodex.core._utils import (
    UNKNOWN_SPEAKERS,
    AudioPaths,
    free_vram,
    merge_consecutive_segments,
    read_json,
    read_parquet,
    write_json,
    write_parquet,
)

# Suppress known harmless warnings from third-party libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")


def processing_status(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> dict[str, bool]:
    """Return the processing state of an audio file."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return {
        "transcribed": p.segments.exists() and p.segments_meta.exists(),
        "diarized": p.diarization.exists() and p.diarization_meta.exists(),
        "assigned": p.diarized_segments.exists(),
        "mapped": p.speaker_map.exists(),
        "exported": p.transcript.exists() or p.transcript_raw.exists(),
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

    if not force and p.segments.exists() and p.segments_meta.exists():
        logger.info(f"[SKIP] Transcription already exists: {p.segments.name}")
        return load_segments(audio_path, output_dir=output_dir)

    dev, ctype = get_device()
    device = device or dev
    compute_type = compute_type or ctype

    logger.info(f"Transcribing {p.audio_path.name} ({device}, {compute_type})")

    audio = whisperx.load_audio(str(p.audio_path))

    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    free_vram(model)

    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    free_vram(model_a)

    segments = result["segments"]
    duration = float(audio.shape[0]) / 16000
    meta = {"language": language, "duration": duration, "num_segments": len(segments)}

    write_parquet(p.segments, segments)
    write_json(p.segments_meta, meta)

    logger.success(f"Transcription done — {len(segments)} segments → {p.segments.name}")
    return {"segments": segments, **meta}


def load_segments(audio_path: Path | str, output_dir: str | Path | None = None) -> dict:
    """Load raw WhisperX segments from parquet + meta.json.

    Returns:
        dict with keys 'segments', 'language', 'duration', 'num_segments'
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    segments = read_parquet(p.segments)
    meta = read_json(p.segments_meta)
    return {"segments": segments, **meta}


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

    if not force and p.diarization.exists() and p.diarization_meta.exists():
        logger.info(f"[SKIP] Diarization already exists: {p.diarization.name}")
        return load_diarization(audio_path, output_dir=output_dir)

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN not found. Set the HF_TOKEN environment variable or pass hf_token=."
        )

    dev, _ = get_device()
    device = device or dev

    logger.info(f"Diarizing {p.audio_path.name}")

    from whisperx.diarize import DiarizationPipeline

    audio = whisperx.load_audio(str(p.audio_path))
    pipeline = DiarizationPipeline(token=token, device=device)
    diarize_segments = pipeline(
        audio,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

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

    write_parquet(p.diarization, speakers)
    write_json(p.diarization_meta, meta)

    logger.success(f"Diarization done — {len(unique)} speakers → {p.diarization.name}")
    return {"speakers": speakers, **meta}


def load_diarization(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> dict:
    """Load diarization from parquet + meta.json."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    speakers = read_parquet(p.diarization)
    meta = read_json(p.diarization_meta)
    return {"speakers": speakers, **meta}


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

    if not force and p.diarized_segments.exists():
        logger.info(f"[SKIP] Assignment already exists: {p.diarized_segments.name}")
        return load_diarized_segments(audio_path, output_dir=output_dir)

    def _has_timestamps(d: dict) -> bool:
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

    write_parquet(p.diarized_segments, segments)
    logger.success(
        f"Assignment done — {len(segments)} segments → {p.diarized_segments.name}"
    )
    return segments


def load_diarized_segments(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> list[dict]:
    """Load diarized segments from parquet."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return read_parquet(p.diarized_segments)


# ──────────────────────────────────────────────
# STEP 4 — Speaker map
# ──────────────────────────────────────────────


def load_speaker_map(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> dict[str, str]:
    """Load SPEAKER_XX → name mapping. Returns {} if file does not exist."""
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
    (e.g. "SPEAKER_00", "SPEAKER_01"). Values are display names used
    in the exported transcript.

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
    max_gap: float = 10.0,
) -> list[dict]:
    """
    Generate the final JSON transcript with resolved speaker names.
    Requires diarized_segments.parquet + speaker_map.json.
    Saves transcript.raw.json in output_dir (pipeline output, unvalidated).

    The file format is:
        {"meta": {show, episode, speakers, duration, word_count}, "segments": [...]}

    Args:
        audio_path : source audio file
        output_dir : directory relative to audio_path for outputs
        show       : podcast show name (stored in meta, defaults to "")
        episode    : episode name (stored in meta, defaults to "")
        max_gap    : maximum silence gap (seconds) to merge across (default 10s);
                     0 disables merging

    Returns:
        List of final segments [{start, end, speaker, text}]
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    segments = load_diarized_segments(audio_path, output_dir=output_dir)
    mapping = load_speaker_map(audio_path, output_dir=output_dir)

    resolved = [
        {
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "speaker": mapping.get(seg.get("speaker") or "", seg.get("speaker") or "")
            or "UNKNOWN",
            "text": str(seg.get("text", "")).strip(),
        }
        for seg in segments
    ]
    export = merge_consecutive_segments(resolved, max_gap=max_gap)

    meta = {
        "show": show,
        "episode": episode,
        "speakers": sorted({seg["speaker"] for seg in export}),
        "duration": round(max((seg["end"] for seg in export), default=0.0), 3),
        "word_count": sum(len(seg["text"].split()) for seg in export),
    }

    out_data = {"meta": meta, "segments": export}

    write_json(p.transcript_raw, out_data)
    logger.success(
        f"Export done — {len(resolved)} → {len(export)} segments (merged) → {p.transcript_raw.name}"
    )
    return export


def _load_transcript_file(path: Path) -> dict:
    """Load a transcript JSON file, handling both old (list) and new (dict) formats.

    Returns {"meta": {...}, "segments": [...]}
    """
    data = read_json(path)
    if isinstance(data, dict):
        return data
    return {"meta": {}, "segments": data}


def load_transcript_full(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> dict:
    """Load the final transcript with metadata.

    Prefers the validated transcript.json; falls back to transcript.raw.json.

    Returns:
        {"meta": {show, episode, speakers, duration, word_count}, "segments": [...]}
        If the file is in old list format, meta will be an empty dict.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return _load_transcript_file(p.transcript_best)


def load_transcript(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> list[dict]:
    """Load the final transcript segments as a plain list.

    Prefers the validated transcript.json; falls back to transcript.raw.json.
    """
    return load_transcript_full(audio_path, output_dir=output_dir)["segments"]


def load_transcript_raw(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> list[dict]:
    """Load segments specifically from transcript.raw.json (pipeline output)."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return _load_transcript_file(p.transcript_raw)["segments"]


def load_transcript_validated(
    audio_path: Path | str, output_dir: str | Path | None = None
) -> list[dict]:
    """Load segments specifically from transcript.json (user-validated)."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return _load_transcript_file(p.transcript)["segments"]


# ──────────────────────────────────────────────
# Segment analysis
# ──────────────────────────────────────────────


# Segments assigned to this name are excluded by clean_transcript().
REMOVE_SPEAKERS = {"[remove]"}


def segment_speech_density(seg: dict) -> float | None:
    """Return chars/second for a segment, or None if duration is too short to measure."""
    text = str(seg.get("text", "")).strip()
    dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
    if dur < 0.5:
        return None
    return len(text) / dur


def is_segment_flagged(seg: dict) -> bool:
    """Return True if a segment is suspicious and should be reviewed or removed.

    A segment is flagged when:
    - speaker is missing or an unresolved placeholder (UNKNOWN, UNK, …)
    - speaker is a reserved remove marker ([remove])
    - speech density is abnormally low (< 2 chars/s), indicating music, noise,
      or a Whisper hallucination artifact
    """
    speaker = seg.get("speaker", "")
    if speaker == "[BREAK]":
        return False
    if not speaker or speaker in UNKNOWN_SPEAKERS:
        return True
    if speaker in REMOVE_SPEAKERS:
        return True
    density = segment_speech_density(seg)
    return density is not None and density < 2.0


def clean_transcript(
    segments: list[dict],
    *,
    remove_unknown_speakers: bool = True,
    remove_low_density: bool = True,
) -> list[dict]:
    """Remove flagged segments from a transcript.

    Args:
        segments               : list of segment dicts (from load_transcript)
        remove_unknown_speakers: drop segments with missing/unresolved speaker
        remove_low_density     : drop segments with < 2 chars/s (hallucinations)

    Returns:
        Filtered list of segments.
    """
    result = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        if speaker == "[BREAK]":
            result.append(seg)
            continue
        if speaker in REMOVE_SPEAKERS:
            continue
        if remove_unknown_speakers and (not speaker or speaker in UNKNOWN_SPEAKERS):
            continue
        if remove_low_density:
            density = segment_speech_density(seg)
            if density is not None and density < 2.0:
                continue
        result.append(seg)
    logger.debug(f"clean_transcript: {len(segments)} → {len(result)} segments")
    return result


def save_transcript(
    audio_path: Path | str,
    segments: list[dict],
    output_dir: str | Path | None = None,
    max_gap: float = 10.0,
) -> None:
    """Save an edited segment list back to transcript.json, preserving metadata.

    Consecutive segments from the same speaker are merged when the gap between
    them is ≤ max_gap seconds.  Pass max_gap=0 to disable merging entirely.

    Args:
        audio_path : source audio file (used to locate transcript.json)
        segments   : updated segment list
        output_dir : same output_dir used when the transcript was created
        max_gap    : maximum silence gap (seconds) to merge across (default 10s);
                     0 disables merging
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    full = load_transcript_full(audio_path, output_dir=output_dir)
    merged = merge_consecutive_segments(segments, max_gap=max_gap)
    full["segments"] = merged
    write_json(p.transcript, full)
    logger.info(
        f"Transcript saved → {p.transcript.name} ({len(segments)} → {len(merged)} segments)"
    )


def audio_duration(path: Path | str) -> float | None:
    """Return audio duration in seconds, trying soundfile then ffprobe.

    Returns None if neither method is available.
    """
    path = Path(path)
    try:
        import soundfile as sf

        return sf.info(str(path)).duration
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ],
            stderr=subprocess.DEVNULL,
        )
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return None


def trim_audio(
    audio_path: Path | str,
    start: float,
    end: float,
    output_dir: str | Path | None = None,
) -> Path:
    """Cut audio_path to [start, end] and save as a new file.

    The trimmed file is placed in a sibling directory named
    ``{stem}_trim_{start}_{end}/`` next to output_dir, keeping the original
    filename so downstream pipeline outputs have clean names.

    Returns the path to the trimmed audio file (skips if it already exists).
    """
    audio_path = Path(audio_path)
    out = AudioPaths.output_dir(audio_path, output_dir)

    def _mmss(s: float) -> str:
        return f"{int(s) // 60}m{int(s) % 60:02d}s"

    # Place trim dir next to output_dir, not inside it
    parent = out.parent
    trim_dir = parent / f"{audio_path.stem}_trim_{_mmss(start)}_{_mmss(end)}"
    trim_dir.mkdir(parents=True, exist_ok=True)
    dest = trim_dir / audio_path.name
    if dest.exists():
        return dest
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            str(start),
            "-to",
            str(end),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    logger.info(f"Trimmed audio → {dest.name} ({_mmss(start)} → {_mmss(end)})")
    return dest
