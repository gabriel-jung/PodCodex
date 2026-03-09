"""
podcodex.core._paths — Shared path resolution for episode output directories.
"""

from pathlib import Path


def episode_output_dir(audio_path: Path, output_dir: str | Path | None = None) -> Path:
    """Resolve the output directory for files related to a given episode.

    Args:
        audio_path : source audio file
        output_dir :
            None (default) — per-episode subfolder next to the audio:
                             {audio.parent}/{audio.stem}/   (matches UI behaviour)
            ""             — flat: files land directly next to the audio file
            relative path  — resolved relative to audio_path.parent
            absolute path  — used as-is

    Returns:
        Resolved output directory Path (not yet created).
    """
    if output_dir is None:
        return audio_path.parent / audio_path.stem
    p = Path(output_dir)
    if not str(output_dir):  # empty string
        return audio_path.parent
    return p if p.is_absolute() else audio_path.parent / p
