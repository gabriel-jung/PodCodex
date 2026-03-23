"""
podcodex.ingest.importer — Import external transcripts into PodCodex format.

Accepts a transcript JSON file (standard ``{meta, segments}`` format or a
bare segments list) and writes it into the show folder structure so it can
be vectorized with ``podcodex vectorize``.

For transcripts without timestamps, positional pseudo-timestamps are
computed as character-offset percentages (0–100) of the full transcript.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from podcodex.ingest.rss import RSSEpisode


def _has_real_timestamps(segments: list[dict]) -> bool:
    """Return True if any segment has a non-zero start or end value."""
    return any(
        seg.get("start", 0.0) != 0.0 or seg.get("end", 0.0) != 0.0 for seg in segments
    )


def _compute_positions(segments: list[dict]) -> None:
    """Add positional pseudo-timestamps (0–100) to segments that lack real ones.

    Modifies segments in place. Only runs when no segment has a real timestamp.
    """
    if _has_real_timestamps(segments):
        return

    total_chars = sum(len(seg.get("text", "")) for seg in segments)
    if total_chars == 0:
        return

    cursor = 0
    for seg in segments:
        text_len = len(seg.get("text", ""))
        seg["start"] = round(cursor / total_chars * 100, 1)
        cursor += text_len
        seg["end"] = round(cursor / total_chars * 100, 1)


def import_transcript(
    transcript_path: Path,
    show_folder: Path,
    episode_stem: str,
    show_name: str,
    *,
    rss_episode: RSSEpisode | None = None,
) -> Path:
    """Import a transcript JSON file into the show folder structure.

    Args:
        transcript_path: Path to the source transcript JSON.
        show_folder: Root folder for the show.
        episode_stem: Filesystem-safe episode identifier (used as subdir name).
        show_name: Canonical show name for metadata.
        rss_episode: Optional RSS metadata to merge into the transcript meta.

    Returns:
        Path to the written transcript file.
    """
    raw = json.loads(Path(transcript_path).read_text(encoding="utf-8"))

    # Accept both {meta, segments} and bare list formats
    if isinstance(raw, list):
        transcript = {"meta": {}, "segments": raw}
    elif isinstance(raw, dict):
        transcript = raw
        transcript.setdefault("meta", {})
        transcript.setdefault("segments", [])
    else:
        raise ValueError(f"Unexpected transcript format: {type(raw)}")

    meta = transcript["meta"]
    meta["show"] = show_name
    meta["episode"] = episode_stem
    meta["source"] = "imported"

    segments = transcript["segments"]
    meta["timed"] = _has_real_timestamps(segments)

    if not meta["timed"] and segments:
        _compute_positions(segments)

    # Merge RSS metadata if available
    if rss_episode:
        from podcodex.ingest.rss import save_episode_meta

        meta["rss_title"] = rss_episode.title
        meta["rss_pub_date"] = rss_episode.pub_date
        if rss_episode.description:
            meta["rss_description"] = rss_episode.description
        if rss_episode.duration:
            meta["rss_duration"] = rss_episode.duration
        if rss_episode.episode_number is not None:
            meta["episode_number"] = rss_episode.episode_number
        if rss_episode.season_number is not None:
            meta["season_number"] = rss_episode.season_number

    # Write to show_folder / episode_stem / episode_stem.transcript.json
    output_dir = Path(show_folder) / episode_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if rss_episode:
        save_episode_meta(output_dir, rss_episode)

    dest = output_dir / f"{episode_stem}.transcript.json"
    dest.write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.success(f"Imported transcript → {dest}")
    return dest
