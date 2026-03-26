"""
podcodex.ingest.folder — Scan a show folder for episodes and report per-episode status.

Episodes are discovered from three sources (in priority order):
    1. Audio files in the show folder (mp3, wav, m4a, ogg, flac)
    2. Subdirectories that contain transcript files (transcript-only episodes)
    3. Subdirectories with ``.episode_meta.json`` (metadata-only, e.g. from RSS)

Audio-sourced entries take priority when both exist for the same stem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from podcodex.core._utils import INTERNAL_SUFFIXES as _INTERNAL_SUFFIXES
from podcodex.ingest.rss import EPISODE_META_FILE

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

_TRANSCRIPT_MARKERS = frozenset(
    {
        "transcript.json",
        "transcript.raw.json",
        "nodiar.transcript.json",
        "nodiar.transcript.raw.json",
    }
)


@dataclass
class EpisodeInfo:
    """Status snapshot for a single episode in a show folder."""

    audio_path: Path | None  # None for transcript-only or metadata-only episodes
    stem: str  # filesystem identifier (directory / audio filename without extension)
    output_dir: Path  # show_folder / stem  (where all processing outputs live)
    title: str = ""  # display title from RSS metadata (falls back to stem)
    # transcription pipeline steps
    segments_ready: bool = False
    diarized: bool = False
    assigned: bool = False
    mapped: bool = False  # speaker_map.json exists
    transcribed: bool = False  # transcript exported (raw or validated)
    polished: bool = False
    indexed: bool = False
    synthesized: bool = False
    translations: list[str] = field(default_factory=list)
    # raw/validated status per step
    raw_transcript: bool = False
    validated_transcript: bool = False
    raw_polished: bool = False
    validated_polished: bool = False
    raw_translations: list[str] = field(default_factory=list)
    validated_translations: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path | None:
        """Back-compat alias for ``audio_path``."""
        return self.audio_path


def _episode_status(stem: str, existing: set[str]) -> dict:
    """Derive pipeline status flags from the set of filenames in an output dir."""
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

    transcript_raw = f"{stem}.transcript.raw.json" in existing
    transcript_val = f"{stem}.transcript.json" in existing
    nodiar_transcript_raw = f"{stem}.nodiar.transcript.raw.json" in existing
    nodiar_transcript_val = f"{stem}.nodiar.transcript.json" in existing
    transcribed = (
        transcript_raw
        or transcript_val
        or nodiar_transcript_raw
        or nodiar_transcript_val
    )

    polished_raw = f"{stem}.polished.raw.json" in existing
    polished_val = f"{stem}.polished.json" in existing
    nodiar_polished_raw = f"{stem}.nodiar.polished.raw.json" in existing
    nodiar_polished_val = f"{stem}.nodiar.polished.json" in existing
    polished = (
        polished_raw or polished_val or nodiar_polished_raw or nodiar_polished_val
    )

    indexed = ".rag_indexed" in existing
    synthesized = f"{stem}.synthesized.wav" in existing

    # Translations: derive from filenames
    langs_validated: set[str] = set()
    langs_raw: set[str] = set()
    for fname in existing:
        if not fname.startswith(f"{stem}.") or not fname.endswith(".json"):
            continue
        suffix = fname[len(stem) + 1 : -5]
        if suffix in _INTERNAL_SUFFIXES:
            continue
        if suffix.endswith(".raw"):
            base = suffix[:-4]
            if base not in _INTERNAL_SUFFIXES:
                langs_raw.add(base.removeprefix("nodiar."))
        else:
            langs_validated.add(suffix.removeprefix("nodiar."))

    return {
        "segments_ready": segments_ready,
        "diarized": diarized,
        "assigned": assigned,
        "mapped": mapped,
        "transcribed": transcribed,
        "polished": polished,
        "indexed": indexed,
        "synthesized": synthesized,
        "translations": sorted(langs_validated | langs_raw),
        "raw_transcript": (
            (transcript_raw and not transcript_val)
            or (nodiar_transcript_raw and not nodiar_transcript_val)
        ),
        "validated_transcript": transcript_val or nodiar_transcript_val,
        "raw_polished": (
            (polished_raw and not polished_val)
            or (nodiar_polished_raw and not nodiar_polished_val)
        ),
        "validated_polished": polished_val or nodiar_polished_val,
        "raw_translations": sorted(langs_raw - langs_validated),
        "validated_translations": sorted(langs_validated),
    }


def _list_dir(d: Path) -> set[str]:
    """Return the set of filenames in *d*, or empty set if it doesn't exist."""
    if d.is_dir():
        return {f.name for f in d.iterdir()}
    return set()


def _load_title(output_dir: Path) -> str:
    """Read the display title from episode metadata if it exists."""
    meta_path = output_dir / EPISODE_META_FILE
    if not meta_path.exists():
        return ""
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data.get("title", "")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Corrupt episode metadata, skipping: {meta_path} ({exc})")
        return ""


def _make_episode(
    stem: str,
    output_dir: Path,
    existing: set[str],
    audio_path: Path | None = None,
) -> EpisodeInfo:
    """Build an EpisodeInfo from a stem and the files in its output dir."""
    return EpisodeInfo(
        audio_path=audio_path,
        stem=stem,
        output_dir=output_dir,
        title=_load_title(output_dir),
        **_episode_status(stem, existing),
    )


def scan_folder(show_folder: Path) -> list[EpisodeInfo]:
    """Return a sorted list of EpisodeInfo for every episode in *show_folder*."""
    show_folder = Path(show_folder)
    episodes: dict[str, EpisodeInfo] = {}

    # Pass 1: audio files
    for item in show_folder.iterdir():
        if item.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        stem = item.stem
        output_dir = show_folder / stem
        episodes[stem] = _make_episode(
            stem, output_dir, _list_dir(output_dir), audio_path=item
        )

    # Pass 2 & 3: subdirectories (transcript-only or metadata-only)
    for item in show_folder.iterdir():
        if not item.is_dir() or item.name in episodes:
            continue
        stem = item.name
        existing = _list_dir(item)
        has_transcript = any(f"{stem}.{m}" in existing for m in _TRANSCRIPT_MARKERS)
        has_meta = EPISODE_META_FILE in existing
        if has_transcript or has_meta:
            episodes[stem] = _make_episode(stem, item, existing)

    return sorted(episodes.values(), key=lambda ep: ep.stem)


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
