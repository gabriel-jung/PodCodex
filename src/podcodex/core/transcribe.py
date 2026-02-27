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
    {stem}.transcript.json           — final export with resolved speaker names
"""

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self

import pandas as pd
import torch
import warnings
from loguru import logger

# Suppress known harmless warnings from third-party libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────


@dataclass
class _EpisodePaths:
    """All derived file paths for a given audio episode."""

    base: Path  # audio_path.parent / output_dir / stem — no extension

    @classmethod
    def from_audio(cls, audio_path: Path, output_dir: str | Path = "") -> Self:
        if output_dir:
            output_dir = Path(output_dir)
            root = (
                output_dir
                if output_dir.is_absolute()
                else audio_path.parent / output_dir
            )
        else:
            root = audio_path.parent
        base = root / audio_path.stem
        base.parent.mkdir(parents=True, exist_ok=True)
        return cls(base=base)

    @property
    def segments(self) -> Path:
        return self.base.with_suffix(".segments.parquet")

    @property
    def segments_meta(self) -> Path:
        return self.base.with_suffix(".segments.meta.json")

    @property
    def diarization(self) -> Path:
        return self.base.with_suffix(".diarization.parquet")

    @property
    def diarization_meta(self) -> Path:
        return self.base.with_suffix(".diarization.meta.json")

    @property
    def diarized_segments(self) -> Path:
        return self.base.with_suffix(".diarized_segments.parquet")

    @property
    def speaker_map(self) -> Path:
        return self.base.with_suffix(".speaker_map.json")

    @property
    def transcript(self) -> Path:
        return self.base.with_suffix(".transcript.json")


def processing_status(audio_path: Path, output_dir: str | Path = "") -> dict[str, bool]:
    """Return the processing state of an audio file."""
    p = _EpisodePaths.from_audio(Path(audio_path), output_dir=output_dir)
    return {
        "transcribed": p.segments.exists() and p.segments_meta.exists(),
        "diarized": p.diarization.exists() and p.diarization_meta.exists(),
        "assigned": p.diarized_segments.exists(),
        "mapped": p.speaker_map.exists(),
        "exported": p.transcript.exists(),
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
    compute_type: Optional[str] = None,
    device: Optional[str] = None,
    force: bool = False,
    output_dir: str | Path = "",
) -> dict:
    """
    Transcribe an audio file with WhisperX + phonetic alignment.
    Saves segments.parquet + segments.meta.json in output_dir (relative to audio).

    Returns:
        dict with keys 'segments', 'language', 'duration', 'num_segments'
    """
    import whisperx

    audio_path = Path(audio_path)
    p = _EpisodePaths.from_audio(audio_path, output_dir=output_dir)

    if not force and p.segments.exists() and p.segments_meta.exists():
        logger.info(f"[SKIP] Transcription already exists: {p.segments.name}")
        return load_transcription(audio_path, output_dir=output_dir)

    dev, ctype = get_device()
    device = device or dev
    compute_type = compute_type or ctype

    logger.info(f"Transcribing {audio_path.name} ({device}, {compute_type})")

    audio = whisperx.load_audio(str(audio_path))

    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    _free_vram(model)

    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    _free_vram(model_a)

    segments = result["segments"]
    duration = float(audio.shape[0]) / 16000
    meta = {"language": language, "duration": duration, "num_segments": len(segments)}

    pd.DataFrame(segments).to_parquet(p.segments, index=False)
    p.segments_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.success(f"Transcription done — {len(segments)} segments → {p.segments.name}")
    return {"segments": segments, **meta}


def load_transcription(audio_path: Path | str, output_dir: str | Path = "") -> dict:
    """Load transcription from parquet + meta.json."""
    p = _EpisodePaths.from_audio(Path(audio_path), output_dir=output_dir)
    segments = pd.read_parquet(p.segments).to_dict("records")
    meta = json.loads(p.segments_meta.read_text(encoding="utf-8"))
    return {"segments": segments, **meta}


# ──────────────────────────────────────────────
# STEP 2 — Diarization
# ──────────────────────────────────────────────


def diarize_file(
    audio_path: Path | str,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
    device: Optional[str] = None,
    force: bool = False,
    output_dir: str | Path = "",
) -> dict:
    """
    Diarize an audio file using whisperx.DiarizationPipeline (pyannote).
    Saves diarization.parquet + diarization.meta.json in output_dir (relative to audio).

    hf_token is read from HF_TOKEN env variable if not provided.

    Returns:
        dict with keys 'speakers' (list of {start, end, speaker} dicts), 'num_speakers'
    """
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    audio_path = Path(audio_path)
    p = _EpisodePaths.from_audio(audio_path, output_dir=output_dir)

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

    logger.info(f"Diarizing {audio_path.name}")

    audio = whisperx.load_audio(str(audio_path))
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

    pd.DataFrame(speakers).to_parquet(p.diarization, index=False)
    p.diarization_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.success(f"Diarization done — {len(unique)} speakers → {p.diarization.name}")
    return {"speakers": speakers, **meta}


def load_diarization(audio_path: Path | str, output_dir: str | Path = "") -> dict:
    """Load diarization from parquet + meta.json."""
    p = _EpisodePaths.from_audio(Path(audio_path), output_dir=output_dir)
    speakers = pd.read_parquet(p.diarization).to_dict("records")
    meta = json.loads(p.diarization_meta.read_text(encoding="utf-8"))
    return {"speakers": speakers, **meta}


# ──────────────────────────────────────────────
# STEP 3 — Speaker assignment
# ──────────────────────────────────────────────


def assign_speakers_to_file(
    audio_path: Path | str,
    force: bool = False,
    output_dir: str | Path = "",
) -> list[dict]:
    """
    Assign SPEAKER_XX labels to segments via whisperx.assign_word_speakers.
    Requires transcribe_file() and diarize_file() to have already run.
    Saves diarized_segments.parquet in output_dir (relative to audio).

    Returns:
        List of segments with 'speaker' key.
    """
    import whisperx

    audio_path = Path(audio_path)
    p = _EpisodePaths.from_audio(audio_path, output_dir=output_dir)

    if not force and p.diarized_segments.exists():
        logger.info(f"[SKIP] Assignment already exists: {p.diarized_segments.name}")
        return load_diarized_segments(audio_path, output_dir=output_dir)

    transcription = load_transcription(audio_path, output_dir=output_dir)
    diarization = load_diarization(audio_path, output_dir=output_dir)

    df_diarize = pd.DataFrame(diarization["speakers"])
    result = whisperx.assign_word_speakers(
        df_diarize, {"segments": transcription["segments"]}
    )
    segments = result["segments"]

    pd.DataFrame(segments).to_parquet(p.diarized_segments, index=False)
    logger.success(
        f"Assignment done — {len(segments)} segments → {p.diarized_segments.name}"
    )
    return segments


def load_diarized_segments(
    audio_path: Path | str, output_dir: str | Path = ""
) -> list[dict]:
    """Load diarized segments from parquet."""
    p = _EpisodePaths.from_audio(Path(audio_path), output_dir=output_dir)
    return pd.read_parquet(p.diarized_segments).to_dict("records")


# ──────────────────────────────────────────────
# STEP 4 — Speaker map (filled by UI)
# ──────────────────────────────────────────────


def load_speaker_map(
    audio_path: Path | str, output_dir: str | Path = ""
) -> dict[str, str]:
    """Load SPEAKER_XX → name mapping. Returns {} if file does not exist."""
    p = _EpisodePaths.from_audio(Path(audio_path), output_dir=output_dir)
    if not p.speaker_map.exists():
        return {}
    return json.loads(p.speaker_map.read_text(encoding="utf-8"))


def save_speaker_map(
    audio_path: Path | str, mapping: dict[str, str], output_dir: str | Path = ""
) -> None:
    """Save SPEAKER_XX → name mapping."""
    p = _EpisodePaths.from_audio(Path(audio_path), output_dir=output_dir)
    p.speaker_map.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Speaker map saved → {p.speaker_map.name}")


# ──────────────────────────────────────────────
# STEP 5 — Final export
# ──────────────────────────────────────────────


def simplify_transcript(segments: list[dict]) -> list[dict]:
    """
    Merge consecutive segments from the same speaker into single entries.
    Called automatically by export_transcript().

    Returns:
        List of simplified segments [{speaker, start, end, text}]
    """
    result = []
    for seg in segments:
        speaker = seg.get("speaker_name") or seg.get("speaker", "UNKNOWN")
        entry = {
            "speaker": speaker,
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "text": str(seg.get("text", "")).strip(),
        }
        if result and result[-1]["speaker"] == entry["speaker"]:
            result[-1]["end"] = entry["end"]
            result[-1]["text"] += " " + entry["text"]
        else:
            result.append(entry)
    return result


def export_transcript(
    audio_path: Path | str, output_dir: str | Path = ""
) -> list[dict]:
    """
    Generate the final JSON transcript with resolved speaker names.
    Requires diarized_segments.parquet + speaker_map.json.
    Saves transcript.json in output_dir (relative to audio).

    Returns:
        List of final segments [{start, end, speaker_id, speaker_name, text}]
    """
    audio_path = Path(audio_path)
    p = _EpisodePaths.from_audio(audio_path, output_dir=output_dir)

    segments = load_diarized_segments(audio_path, output_dir=output_dir)
    mapping = load_speaker_map(audio_path, output_dir=output_dir)

    resolved = [
        {
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "speaker": mapping.get(
                seg.get("speaker", ""), seg.get("speaker", "UNKNOWN")
            ),
            "text": str(seg.get("text", "")).strip(),
        }
        for seg in segments
    ]
    export = simplify_transcript(resolved)

    p.transcript.write_text(
        json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.success(f"Export done — {len(export)} segments → {p.transcript.name}")
    return export


def load_transcript(audio_path: Path | str, output_dir: str | Path = "") -> list[dict]:
    """Load the final transcript."""
    p = _EpisodePaths.from_audio(Path(audio_path), output_dir=output_dir)
    return json.loads(p.transcript.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────


def _free_vram(model) -> None:
    """Release VRAM after model use."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcript_to_text(segments: list[dict]) -> str:
    """Format transcript segments as plain readable text for quick review.

    Works with both load_transcript() (resolved names) and
    load_diarized_segments() (raw SPEAKER_XX labels).
    """
    lines = []
    for seg in segments:
        speaker = seg.get(
            "speaker_name", seg.get("speaker_id", seg.get("speaker", "UNKNOWN"))
        )
        start = seg["start"]
        end = seg["end"]
        text = str(seg.get("text", "")).strip()
        lines.append(f"[{start:.3f}s - {end:.3f}s] {speaker}\n{text}")
    return "\n\n".join(lines)
