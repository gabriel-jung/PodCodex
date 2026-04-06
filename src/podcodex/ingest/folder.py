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
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from podcodex.core.constants import AUDIO_EXTENSIONS
from podcodex.ingest.rss import EPISODE_META_FILE

# ── Scan cache ──────────────────────────────────
_scan_cache: dict[str, tuple[float, list["EpisodeInfo"]]] = {}
_CACHE_TTL = 10.0  # seconds

_TRANSCRIPT_MARKERS = frozenset(
    {
        "transcript.json",
        "transcript.raw.json",
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

    @property
    def path(self) -> Path | None:
        """Back-compat alias for ``audio_path``."""
        return self.audio_path


def _episode_status(stem: str, existing: set[str]) -> dict:
    """Derive pipeline status flags from the set of filenames in an output dir.

    Only detects artifacts that are still written to disk (transcription
    intermediates, transcript files, synthesis outputs, RAG marker).
    Polish/translation status comes from the version DB via ``mark_step``.
    """
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
    transcribed = transcript_raw or transcript_val

    indexed = ".rag_indexed" in existing
    synthesized = f"{stem}.synthesized.wav" in existing

    return {
        "segments_ready": segments_ready,
        "diarized": diarized,
        "assigned": assigned,
        "mapped": mapped,
        "transcribed": transcribed,
        "polished": False,
        "indexed": indexed,
        "synthesized": synthesized,
        "translations": [],
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
    """Return a sorted list of EpisodeInfo for every episode in *show_folder*.

    Results are cached for ``_CACHE_TTL`` seconds.  Call
    ``invalidate_scan_cache(show_folder)`` after mutations.
    """
    show_folder = Path(show_folder)
    key = str(show_folder)
    now = time.monotonic()

    cached = _scan_cache.get(key)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    result = _scan_folder_uncached(show_folder)
    _scan_cache[key] = (now, result)
    return result


def invalidate_scan_cache(show_folder: Path | str | None = None) -> None:
    """Drop cached scan results.  Pass ``None`` to clear everything."""
    if show_folder is None:
        _scan_cache.clear()
    else:
        _scan_cache.pop(str(show_folder), None)


def _scan_folder_uncached(show_folder: Path) -> list[EpisodeInfo]:
    """Batch-scan a show folder in two OS calls instead of O(n)."""
    episodes: dict[str, EpisodeInfo] = {}

    # Single os.scandir for the top-level folder
    audio_files: dict[str, Path] = {}
    subdirs: list[str] = []
    with os.scandir(show_folder) as it:
        for entry in it:
            if entry.is_file(follow_symlinks=False):
                name = entry.name
                dot = name.rfind(".")
                if dot > 0 and name[dot:].lower() in AUDIO_EXTENSIONS:
                    audio_files[name[:dot]] = show_folder / name
            elif entry.is_dir(follow_symlinks=False):
                subdirs.append(entry.name)

    # Batch-collect filenames for all subdirectories in one pass each
    subdir_files: dict[str, set[str]] = {}
    for name in subdirs:
        subdir_path = show_folder / name
        try:
            with os.scandir(subdir_path) as sub_it:
                subdir_files[name] = {e.name for e in sub_it}
        except OSError:
            subdir_files[name] = set()

    # Build episodes from audio files
    for stem, audio_path in audio_files.items():
        existing = subdir_files.get(stem, set())
        output_dir = show_folder / stem
        episodes[stem] = _make_episode(
            stem, output_dir, existing, audio_path=audio_path
        )

    # Transcript-only or metadata-only subdirectories
    for name in subdirs:
        if name in episodes:
            continue
        existing = subdir_files[name]
        has_transcript = any(f"{name}.{m}" in existing for m in _TRANSCRIPT_MARKERS)
        has_meta = EPISODE_META_FILE in existing
        if has_transcript or has_meta:
            episodes[name] = _make_episode(name, show_folder / name, existing)

    return sorted(episodes.values(), key=lambda ep: ep.stem)
