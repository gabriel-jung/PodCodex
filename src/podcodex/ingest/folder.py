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

from podcodex.core.source import audio_stem
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
    transcribed: bool = False  # transcript exported (raw or validated)
    corrected: bool = False
    indexed: bool = False
    synthesized: bool = False
    has_subtitles: bool = False
    translations: list[str] = field(default_factory=list)


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


def _episode_status(existing: set[str], output_dir: Path | None = None) -> dict:
    """Derive pipeline status flags from the set of filenames in an output dir.

    A seed for ``pipeline.db``, not the status itself: ``corrected`` and
    ``translations`` are left for ``reconcile_show_status``, which fills them
    from the version files right after seeding. ``indexed`` is intentionally
    False here; authoritative truth comes from the LanceDB index, set by
    ``scan_folder`` after this function returns.
    """
    transcribed = _step_has_versions(output_dir, "transcript", ".json")
    synthesized = _step_has_versions(output_dir, "synthesize", ".wav")
    has_subtitles = any(f.endswith(".vtt") for f in existing)

    return {
        "transcribed": transcribed,
        "corrected": False,
        "indexed": False,
        "synthesized": synthesized,
        "has_subtitles": has_subtitles,
        "translations": [],
    }


# This install's index state by show folder (resolved path): the indexed
# stems, and the source records once a reader asked for them (None until
# then), guarded by the collections' creation stamps and dataset versions.
# Not by display name: two shows may share one, and a stem indexed in either
# used to show up as indexed in both. Listing the episodes means scanning
# every chunk row (tens of ms on a large show) and the episode list route
# asks for it on every request, including the 5s status poll.
_Fingerprint = tuple[tuple[str, str, int], ...]
_INDEXED_STEMS_CACHE: dict[
    str, tuple[_Fingerprint, set[str], dict[str, list[dict]] | None]
] = {}


def _stems_cache_key(show_folder: Path | str) -> str:
    return str(Path(show_folder).resolve())


def _versions_fingerprint(store, cols) -> _Fingerprint:
    """Cache key for a set of collections: sorted (name, created_at, dataset
    version) triples.

    The version alone collides when a collection is dropped and recreated
    under its name: the new table counts up from 1 again and passes through
    the old number. The creation stamp in ``_collections`` tells the two
    apart.

    The coherence contract of `_INDEXED_STEMS_CACHE`: every writer of the
    cache must build its key here, or the scan path and the warm path stop
    agreeing and the cache degrades to stale sets or perpetual rescans.
    """
    info = store.get_all_collection_info()
    return tuple(
        sorted(
            (c, (info.get(c) or {}).get("created_at", ""), store.collection_version(c))
            for c in cols
        )
    )


def _collect_source_records(store, cols) -> dict[str, list[dict]]:
    """Source records of the collections this install can re-index (the
    batch's rule, ``is_known_collection``): a mark it could never clear
    would stay on forever."""
    from podcodex.rag.defaults import is_known_collection

    info = store.get_all_collection_info()
    records: dict[str, list[dict]] = {}
    for col in cols:
        if not is_known_collection(info.get(col) or {}):
            continue
        for stem, record in store.episode_source_records(col).items():
            records.setdefault(stem, []).append(record)
    return records


def _lance_index_state(
    show_folder: Path, *, with_records: bool
) -> tuple[set[str], dict[str, list[dict]] | None] | None:
    """``(indexed stems, source records or None)`` from the shared cache.

    The records are read only when asked for, and then kept with the stems
    under the same fingerprint. None when the index could not be read.
    """
    try:
        from podcodex.ingest.show import load_show_meta
        from podcodex.rag.index_store import get_index_store
    except Exception:
        return None

    meta = load_show_meta(show_folder)
    show_name = (meta.name if meta else None) or show_folder.name
    try:
        store = get_index_store()
        # get_all_collection_info is cached against index_mtime, so this
        # avoids a LanceDB meta-table scan per request; list_collections
        # would rescan every call.
        # meta is already loaded above, so the id costs nothing here; resolving
        # it by name would re-stat every registered show folder per request.
        cols = store.collections_for_show(
            meta.id if meta else "", show_label=show_name, strict=True
        )
        versions = _versions_fingerprint(store, cols)
        key = _stems_cache_key(show_folder)
        read = cached = _INDEXED_STEMS_CACHE.get(key)
        if cached is None or cached[0] != versions:
            indexed: set[str] = set()
            for col in cols:
                indexed.update(store.list_episodes(col))
            cached = (versions, indexed, None)
        if with_records and cached[2] is None:
            cached = (versions, cached[1], _collect_source_records(store, cols))
        # Stored only when this call built something, and only over the entry
        # it read: an index write's note_episode_indexed may have stored a
        # newer one meanwhile, and putting the older back would cost the next
        # poll the full rescan the warm path exists to avoid.
        if cached is not read and _INDEXED_STEMS_CACHE.get(key) is read:
            _INDEXED_STEMS_CACHE[key] = cached
        records = cached[2] if with_records else None
        return set(cached[1]), (
            {k: list(v) for k, v in records.items()} if records is not None else None
        )
    except Exception as exc:
        logger.warning("lance index lookup failed for {!r}: {!r}", show_name, exc)
        return None


def lance_indexed_stems(show_folder: Path) -> set[str] | None:
    """Return the set of episode stems that LanceDB has chunks for, for this show.

    Authoritative source for ``indexed`` status. An empty set means the show
    has no indexed episode; None means the index could not be read (rag not
    installed, locked table, LanceDB error), so callers must not write
    "not indexed" back as if it were known.

    Cached against the collections' dataset versions, which bump on every
    write from any process, so a stale set can't outlive an index run.
    """
    state = _lance_index_state(show_folder, with_records=False)
    return state[0] if state else None


def lance_source_records(show_folder: Path) -> dict[str, list[dict]] | None:
    """``{stem: [{hash, ref, step}, ...]}`` recorded in this install's index.

    One record per collection the episode is indexed in (they differ when one
    collection was re-indexed after an edit and another was not). Episodes
    indexed before records existed are absent. None when the index could not
    be read, like :func:`lance_indexed_stems`, whose cache entry holds them.
    """
    state = _lance_index_state(show_folder, with_records=True)
    return state[1] if state else None


def note_episode_indexed(show_folder: Path | str, stem: str) -> None:
    """Incrementally add *stem* to the cached index state after an index write.

    Every LanceDB write bumps the dataset version, so during an index batch
    each status poll would find the fingerprint stale and rebuild the set
    with a full chunk scan, once per indexed episode. The server process
    handling the index task knows exactly which episode just landed, so it
    refreshes the fingerprint, adds the stem in place and re-reads that
    episode's source records when they are cached. Collection names come
    from the cached fingerprint itself (metadata version reads only, no
    meta-table scan); an episode indexed into a brand-new collection
    self-heals through the normal rescan on the next request's mismatch.
    Cross-process writers (bot rsync) also take the rescan path; a cold
    cache just waits for the next request's scan.
    """
    key = _stems_cache_key(show_folder)
    cached = _INDEXED_STEMS_CACHE.get(key)
    if cached is None:
        return
    old_versions, old_stems, old_records = cached
    try:
        from podcodex.rag.index_store import get_index_store

        store = get_index_store()
        # Versions only, creation stamps carried over: re-reading them would
        # scan the meta table on every indexed episode.
        versions = tuple(
            (name, created, store.collection_version(name))
            for name, created, _ in old_versions
        )
    except Exception as exc:
        # Can't trust the fingerprint: drop the entry so the next request
        # rebuilds from a real scan instead of serving a stale set.
        _INDEXED_STEMS_CACHE.pop(key, None)
        logger.warning("indexed-set refresh failed for {}: {!r}", key, exc)
        return
    if any(new[2] < old[2] for new, old in zip(versions, old_versions)):
        # A table's version only grows: a lower one means the collection was
        # dropped (0) or recreated since the set was built, and the old set
        # holds the dropped table's episodes, so adding to it would keep them.
        _INDEXED_STEMS_CACHE.pop(key, None)
        return
    records = None
    if old_records is not None:
        try:
            fresh = [
                record
                for name, _created, _version in versions
                if (record := store.episode_source_record(name, stem))
            ]
        except Exception as exc:
            # The stems are still right; the records are re-read on demand.
            logger.warning("source-record refresh failed for {}: {!r}", key, exc)
        else:
            records = dict(old_records)
            if fresh:
                records[stem] = fresh
            else:
                records.pop(stem, None)
    _INDEXED_STEMS_CACHE[key] = (versions, old_stems | {stem}, records)


def forget_indexed_stems(show_folder: Path | str) -> None:
    """Drop the cached index state of *show_folder*.

    For a write that swaps a show's collections wholesale (a bundle import
    under REPLACE), where the imported ``_collections`` row can carry the
    same creation stamp as the one it replaces.
    """
    _INDEXED_STEMS_CACHE.pop(_stems_cache_key(show_folder), None)


def dir_holds_episode(names: set[str] | frozenset[str]) -> bool:
    """Scanner rule for when a bare directory counts as an episode: it holds
    transcript content or episode metadata. Single definition, shared with
    the pipeline-DB heal pass in ``api/routes/shows.py``; if the rule changes
    here, healing follows automatically."""
    return "transcript" in names or EPISODE_META_FILE in names


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
        **_episode_status(existing, output_dir),
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
            if entry.is_dir(follow_symlinks=False):
                subdirs.append(entry.name)
                continue
            # core.source.audio_stem: the shared rule, symlinked files included.
            # With several files for one stem, the first by name wins, as in
            # show_audio_files.
            stem = audio_stem(entry.name)
            if stem is not None and entry.is_file():
                path = show_folder / entry.name
                if stem not in audio_files or path < audio_files[stem]:
                    audio_files[stem] = path

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
        if dir_holds_episode(existing):
            episodes[name] = _make_episode(name, show_folder / name, existing)

    if indexed_stems is None:
        indexed_stems = lance_indexed_stems(show_folder) or set()
    for ep in episodes.values():
        ep.indexed = ep.stem in indexed_stems

    return sorted(episodes.values(), key=lambda ep: ep.stem)
