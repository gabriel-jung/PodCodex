"""
podcodex.ingest.folder — Scan a show folder for audio files and report per-episode status.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from podcodex.core._utils import INTERNAL_SUFFIXES as _INTERNAL_SUFFIXES

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


@dataclass
class EpisodeInfo:
    path: Path  # full path to audio file
    stem: str  # filename without extension
    output_dir: Path  # show_folder / stem  (where all processing outputs live)
    # transcription pipeline steps
    segments_ready: bool  # whisperx segments done
    diarized: bool  # diarization done
    assigned: bool  # speaker assignment done
    mapped: bool  # speaker_map.json exists (ready to export)
    transcribed: bool  # transcript exported (raw or validated)
    polished: bool  # polished file exists (raw or validated)
    indexed: bool  # (output_dir / ".rag_indexed").exists()
    translations: list[str]  # all language names (raw or validated)
    # raw/validated status per step
    raw_transcript: bool  # .transcript.raw.json exists, validated doesn't
    validated_transcript: bool  # .transcript.json exists
    raw_polished: bool  # .polished.raw.json exists, validated doesn't
    validated_polished: bool  # .polished.json exists
    raw_translations: list[str]  # langs with raw only (no validated)
    validated_translations: list[str]  # langs with validated file


def scan_folder(show_folder: Path) -> list[EpisodeInfo]:
    """Return a sorted list of EpisodeInfo for every audio file in show_folder.

    Uses a single directory listing per episode output folder to avoid
    hundreds of individual stat/mkdir calls.
    """
    show_folder = Path(show_folder)
    episodes = []
    for audio in sorted(show_folder.iterdir()):
        if audio.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        stem = audio.stem
        output_dir = show_folder / stem

        # Single listing — all subsequent checks are set lookups, no extra I/O.
        if output_dir.is_dir():
            existing = {f.name for f in output_dir.iterdir()}
        else:
            existing = set()

        # ── Pipeline steps ──
        segments_ready = (
            f"{stem}.segments.parquet" in existing
            and f"{stem}.segments.meta.json" in existing
        )
        diarized = (
            f"{stem}.diarization.parquet" in existing
            and f"{stem}.diarization.meta.json" in existing
        )
        assigned = f"{stem}.diarized_segments.parquet" in existing
        mapped = f"{stem}.speaker_map.json" in existing

        transcript_raw_exists = f"{stem}.transcript.raw.json" in existing
        transcript_validated_exists = f"{stem}.transcript.json" in existing
        transcribed = transcript_raw_exists or transcript_validated_exists

        polished_raw_exists = f"{stem}.polished.raw.json" in existing
        polished_validated_exists = f"{stem}.polished.json" in existing
        polished = polished_raw_exists or polished_validated_exists

        indexed = ".rag_indexed" in existing

        # ── Translations: derive from filenames ──
        langs_validated: set[str] = set()
        langs_raw: set[str] = set()
        for fname in existing:
            if not fname.startswith(f"{stem}.") or not fname.endswith(".json"):
                continue
            # suffix = everything between "{stem}." and ".json"
            suffix = fname[len(stem) + 1 : -5]
            if suffix in _INTERNAL_SUFFIXES:
                continue
            if suffix.endswith(".raw"):
                base = suffix[:-4]
                if base not in _INTERNAL_SUFFIXES:
                    langs_raw.add(base)
            else:
                langs_validated.add(suffix)

        all_langs = sorted(langs_validated | langs_raw)
        raw_translations = sorted(langs_raw - langs_validated)
        validated_translations = sorted(langs_validated)

        episodes.append(
            EpisodeInfo(
                path=audio,
                stem=stem,
                output_dir=output_dir,
                segments_ready=segments_ready,
                diarized=diarized,
                assigned=assigned,
                mapped=mapped,
                transcribed=transcribed,
                polished=polished,
                indexed=indexed,
                translations=all_langs,
                raw_transcript=transcript_raw_exists
                and not transcript_validated_exists,
                validated_transcript=transcript_validated_exists,
                raw_polished=polished_raw_exists and not polished_validated_exists,
                validated_polished=polished_validated_exists,
                raw_translations=raw_translations,
                validated_translations=validated_translations,
            )
        )
    return episodes


def find_audio(show_folder: str | Path, episode: str) -> Path | None:
    """Locate the audio file for a given episode stem in a show folder.

    Tries each known audio extension in order. Returns None if not found.
    """
    if not show_folder or not episode:
        return None
    folder = Path(show_folder)
    if not folder.is_dir():
        return None
    for ext in AUDIO_EXTENSIONS:
        candidate = folder / f"{episode}{ext}"
        if candidate.exists():
            return candidate
    return None
