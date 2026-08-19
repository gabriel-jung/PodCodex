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
    transcribed: bool = False  # transcript exported (raw or validated)
    corrected: bool = False
    indexed: bool = False
    synthesized: bool = False
    has_subtitles: bool = False
    translations: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path | None:
        """Back-compat alias for ``audio_path``."""
        return self.audio_path


def _step_has_versions(output_dir: Path | None, step: str, ext: str) -> bool:
    """Return True if ``output_dir/step/`` exists and holds any ``*{ext}`` file.

    Used to derive ``transcribed`` / ``synthesized`` flags from the version
    storage layout (``{ep_dir}/{step}/{id}.{json|wav}``). Mirrors the layout
    owned by ``core/versions.py:version_path``; if the layout changes there,
    update this resolver too.
    """
    if output_dir is None:
        return False
    return next((output_dir / step).glob(f"*{ext}"), None) is not None


def _episode_status(
    stem: str, existing: set[str], output_dir: Path | None = None
) -> dict:
    """Derive pipeline status flags from the set of filenames in an output dir.

    ``indexed`` is intentionally False here — authoritative truth comes from
    the LanceDB index, set by ``scan_folder`` after this function returns.
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

    transcribed = _step_has_versions(output_dir, "transcript", ".json")
    synthesized = _step_has_versions(output_dir, "synthesize", ".wav")
    has_subtitles = any(f.endswith(".vtt") for f in existing)

    return {
        "segments_ready": segments_ready,
        "diarized": diarized,
        "assigned": assigned,
        "transcribed": transcribed,
        "corrected": False,
        "indexed": False,
        "synthesized": synthesized,
        "has_subtitles": has_subtitles,
        "translations": [],
    }


# Indexed-stem sets by show name, guarded by the collections' dataset
# versions. Listing the episodes means scanning every chunk row (tens of ms on
# a large show) and the episode list route asks for it on every request,
# including the 5s status poll.
_INDEXED_STEMS_CACHE: dict[str, tuple[tuple[tuple[str, int], ...], set[str]]] = {}


def _versions_fingerprint(store, cols) -> tuple[tuple[str, int], ...]:
    """Cache key for a set of collections: sorted (name, dataset version) pairs.

    The coherence contract of `_INDEXED_STEMS_CACHE`: every writer of the
    cache must build its key here, or the scan path and the warm path stop
    agreeing and the cache degrades to stale sets or perpetual rescans.
    """
    return tuple(sorted((c, store.collection_version(c)) for c in cols))


def lance_indexed_stems(show_folder: Path) -> set[str]:
    """Return the set of episode stems that LanceDB has chunks for, for this show.

    Authoritative source for ``indexed`` status. Returns empty set if the
    index is unavailable or the show has no collections (treat as
    not-indexed rather than blocking the scan).

    Cached against the collections' dataset versions, which bump on every
    write from any process, so a stale set can't outlive an index run.
    """
    try:
        from podcodex.ingest.show import load_show_meta
        from podcodex.rag.index_store import get_index_store
    except Exception:
        return set()

    meta = load_show_meta(show_folder)
    show_name = (meta.name if meta else None) or show_folder.name
    try:
        store = get_index_store()
        # get_all_collection_info is cached against index_mtime, so this
        # avoids a LanceDB meta-table scan per request; list_collections
        # would rescan every call.
        cols = [
            name
            for name, info in store.get_all_collection_info().items()
            if info.get("show") == show_name
        ]
        versions = _versions_fingerprint(store, cols)
        cached = _INDEXED_STEMS_CACHE.get(show_name)
        if cached is not None and cached[0] == versions:
            return set(cached[1])
        indexed: set[str] = set()
        for col in cols:
            indexed.update(store.list_episodes(col))
        _INDEXED_STEMS_CACHE[show_name] = (versions, indexed)
        return set(indexed)
    except Exception as exc:
        logger.warning("lance indexed-set lookup failed for {!r}: {!r}", show_name, exc)
        return set()


def note_episode_indexed(show_name: str, stem: str) -> None:
    """Incrementally add *stem* to the cached indexed set after an index write.

    Every LanceDB write bumps the dataset version, so during an index batch
    each status poll would find the fingerprint stale and rebuild the set
    with a full chunk scan — once per indexed episode. The server process
    handling the index task knows exactly which episode just landed, so it
    refreshes the fingerprint and adds the stem in place. Collection names
    come from the cached fingerprint itself (metadata version reads only,
    no meta-table scan); an episode indexed into a brand-new collection
    self-heals through the normal rescan on the next request's mismatch.
    Cross-process writers (bot rsync) also take the rescan path; a cold
    cache just waits for the next request's scan.
    """
    cached = _INDEXED_STEMS_CACHE.get(show_name)
    if cached is None:
        return
    try:
        from podcodex.rag.index_store import get_index_store

        versions = _versions_fingerprint(
            get_index_store(), [name for name, _ in cached[0]]
        )
    except Exception as exc:
        # Can't trust the fingerprint: drop the entry so the next request
        # rebuilds from a real scan instead of serving a stale set.
        _INDEXED_STEMS_CACHE.pop(show_name, None)
        logger.warning("indexed-set refresh failed for {!r}: {!r}", show_name, exc)
        return
    stems = set(cached[1])
    stems.add(stem)
    _INDEXED_STEMS_CACHE[show_name] = (versions, stems)


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
        **_episode_status(stem, existing, output_dir),
    )


def scan_folder(
    show_folder: Path, indexed_stems: set[str] | None = None
) -> list[EpisodeInfo]:
    """Return a sorted list of EpisodeInfo for every episode in *show_folder*.

    Results are cached for ``_CACHE_TTL`` seconds. Call
    ``invalidate_scan_cache(show_folder)`` after mutations.

    Args:
        indexed_stems: Pre-computed set from :func:`lance_indexed_stems`.
            Pass it when the caller already queried LanceDB to avoid a
            second round-trip.
    """
    show_folder = Path(show_folder)
    key = str(show_folder)
    now = time.monotonic()

    cached = _scan_cache.get(key)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    result = _scan_folder_uncached(show_folder, indexed_stems)
    _scan_cache[key] = (now, result)
    return result


def invalidate_scan_cache(show_folder: Path | str | None = None) -> None:
    """Drop cached scan results.  Pass ``None`` to clear everything."""
    if show_folder is None:
        _scan_cache.clear()
    else:
        _scan_cache.pop(str(show_folder), None)


def _scan_folder_uncached(
    show_folder: Path, indexed_stems: set[str] | None = None
) -> list[EpisodeInfo]:
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
        has_transcript = "transcript" in existing
        has_meta = EPISODE_META_FILE in existing
        if has_transcript or has_meta:
            episodes[name] = _make_episode(name, show_folder / name, existing)

    if indexed_stems is None:
        indexed_stems = lance_indexed_stems(show_folder)
    for ep in episodes.values():
        ep.indexed = ep.stem in indexed_stems

    return sorted(episodes.values(), key=lambda ep: ep.stem)
