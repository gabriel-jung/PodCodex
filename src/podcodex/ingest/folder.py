"""
podcodex.ingest.folder — Scan a show folder for audio files and report per-episode status.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from podcodex.core.transcribe import processing_status

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


@dataclass
class EpisodeInfo:
    path: Path  # full path to audio file
    stem: str  # filename without extension
    output_dir: Path  # show_folder / stem  (where all processing outputs live)
    transcribed: bool  # status["exported"] — transcript.json exists
    polished: bool  # polished.json exists
    indexed: bool  # (output_dir / ".rag_indexed").exists()


def scan_folder(show_folder: Path) -> list[EpisodeInfo]:
    """Return a sorted list of EpisodeInfo for every audio file in show_folder."""
    from podcodex.core.polish import polished_exists

    show_folder = Path(show_folder)
    episodes = []
    for audio in sorted(show_folder.iterdir()):
        if audio.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        output_dir = show_folder / audio.stem
        status = processing_status(audio, output_dir=output_dir)
        episodes.append(
            EpisodeInfo(
                path=audio,
                stem=audio.stem,
                output_dir=output_dir,
                transcribed=status["exported"],
                polished=polished_exists(audio, output_dir=output_dir),
                indexed=(output_dir / ".rag_indexed").exists(),
            )
        )
    return episodes
