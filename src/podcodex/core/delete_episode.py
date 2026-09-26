"""Whole-episode deletion: chunks, output dir, audio copy, pipeline_db row.

Counterpart to the per-step ``delete_version`` facility in ``versions.py``.
That one removes a single version and demotes a flag; this one removes the
episode itself from all four stores that know about it.

Called by the episode delete route and ``core/source``. It orchestrates
``rag`` and ``ingest`` from inside ``core``, through function-level imports.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from podcodex.core._utils import bad_path_component
from podcodex.core.source import episode_audio_files


@dataclass
class DeleteReport:
    """Per-store outcome of one episode delete.

    ``collections`` is how many of the show's collections actually held the
    episode; informational, for the log line and the response receipt.
    """

    collections: int = 0
    output_dir_removed: bool = False
    audio_removed: bool = False
    db_row_removed: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def files_clean(self) -> bool:
        """True when every step succeeded and nothing of the episode survives."""
        return not self.warnings


def _delete_chunks(show_dir: Path, stem: str, report: DeleteReport) -> None:
    """Remove the episode from every collection belonging to this show.

    The fan-out itself lives in ``IndexStore.delete_episode_everywhere``;
    this wrapper only resolves the show name and turns a raised error into a
    report warning, which is what aborts the delete before it touches disk.

    No cache invalidation follows: each delete commits a new LanceDB dataset
    version, which is exactly the key ``_versions_fingerprint`` builds, so
    ``lance_indexed_stems`` rebuilds itself on the next request.
    """
    from podcodex.ingest.show import show_display, show_id

    # show_display is the same derivation ``lance_indexed_stems`` uses to find
    # this show's collections by name; the id comes from this folder's own
    # show.toml. Resolving the id from the name instead picked whichever
    # registered show carries that label, so with two shows of one name the
    # other show's chunks went.
    show_name = show_display(show_dir)
    try:
        # Local: lancedb / pyarrow must stay off the API boot path.
        from podcodex.rag.index_store import get_index_store
    except ImportError:
        # No `rag` extra: this install never indexed anything, so there are
        # no chunks to remove, and refusing would make every delete fail.
        logger.info("delete_episode: rag extra not installed, no chunks to remove")
        return
    try:
        touched = get_index_store().delete_episode_everywhere(
            show_name, stem, show_id=show_id(show_dir)
        )
    except Exception as exc:
        report.warnings.append(f"Could not remove chunks from the index: {exc}")
        logger.opt(exception=True).warning("delete_episode: chunk delete failed")
        return
    report.collections = len(touched)


def delete_episode(show_dir: Path, stem: str) -> DeleteReport:
    """Delete every trace of one episode from a show.

    Order is deliberate, and so is where it stops:

    1. LanceDB chunks. **If this fails, nothing else is touched.** Visibility
       is derived from disk (``episode_status.load_status_context`` bootstraps its status
       map from a filesystem scan whenever the DB has no rows), so an episode
       whose files are gone cannot be listed no matter what the DB says.
       Deleting the files first would therefore turn a chunk failure into
       chunks that still answer RAG searches for an episode the UI can no
       longer show, and cannot be retried from. Failing here instead leaves
       the episode completely intact and the retry clean.
    2. The output dir ``{show}/{stem}/``: transcripts, versions, parquet,
       ``llm_failures.json``, voice samples, synthesized audio. Before the DB
       row, because ``versions.py`` and ``llm_failures.py`` key off it.
    3. The audio copy at the show root. This is PodCodex's own copy, made by
       ``import_local_file``; the file the user imported *from* is a path the
       app never recorded and never touches.
    4. The ``pipeline_db`` row, only if 2 and 3 both succeeded. While files
       remain, the row (or the heal pass that rebuilds it from those files)
       is what keeps the episode on screen so the user can retry.

    Steps 2 and 3 are attempted independently of each other. Every step is
    idempotent, so re-running the delete is the documented recovery from a
    partial failure.

    A fifth store exists for feed-backed shows: ``.feed_cache.json``. It is
    pruned only for entries already flagged ``removed`` (see
    ``_stale_feed_guids``), and only after a fully clean delete.
    """
    if bad_path_component(stem):
        raise ValueError(f"Invalid episode stem: {stem!r}")
    if not show_dir.is_dir():
        raise FileNotFoundError(f"Show folder not found: {show_dir}")

    report = DeleteReport()
    ep_dir = show_dir / stem
    # Defence in depth: the consequence of being wrong here is an rmtree on
    # the wrong directory. bad_path_component already refuses separators,
    # traversal and Windows drive prefixes; requiring the parent to be the
    # show folder also covers whatever a future join quirk lets through, and
    # the degenerate ep_dir == show_dir case.
    if ep_dir.parent.resolve() != show_dir.resolve():
        raise ValueError(f"Episode {stem!r} does not resolve inside {show_dir}")

    # Resolved up front: the mapping depends on what is still on disk.
    stale_guids = _stale_feed_guids(show_dir, stem)

    _delete_chunks(show_dir, stem, report)
    if report.warnings:
        # Stop before touching disk. See the ordering note above: removing the
        # files now would strand the surviving chunks behind an episode that
        # can no longer be listed, let alone deleted again.
        report.warnings.append(
            "Stopped before deleting anything on disk. The fan-out has no "
            "rollback, so some collections may already be clear; running the "
            "delete again is safe and finishes the job."
        )
        logger.warning(
            "delete_episode: aborted {!r} in {}: index not clean", stem, show_dir
        )
        return report

    try:
        shutil.rmtree(ep_dir)
        report.output_dir_removed = True
    except FileNotFoundError:
        report.output_dir_removed = True
    except OSError as exc:
        report.warnings.append(f"Could not remove {ep_dir}: {exc}")
        logger.opt(exception=True).warning(
            "delete_episode: rmtree failed for {}", ep_dir
        )

    report.audio_removed = True
    for audio in episode_audio_files(show_dir, stem):
        try:
            audio.unlink()
        except OSError as exc:
            report.audio_removed = False
            report.warnings.append(f"Could not remove {audio.name}: {exc}")
            logger.opt(exception=True).warning(
                "delete_episode: unlink failed for {}", audio
            )

    # Before the row delete, not after: the heal pass in ``episode_status.load_status_context``
    # rebuilds rows from ``scan_folder``, whose results are cached per show.
    from podcodex.ingest.folder import invalidate_scan_cache

    invalidate_scan_cache(show_dir)

    if report.output_dir_removed and report.audio_removed:
        try:
            report.db_row_removed = drop_db_row(show_dir, stem)
        except Exception as exc:
            # Not swallowed: without a warning the route would report a clean
            # delete while the episode stays listed with stale flags.
            report.warnings.append(f"Could not update the episode database: {exc}")
            logger.opt(exception=True).warning(
                "delete_episode: could not drop the row for {!r}", stem
            )
    else:
        report.warnings.append(
            "Left this episode listed because part of it is still on disk. "
            "Close anything using its files and delete it again."
        )

    if report.files_clean:
        _prune_feed_cache(show_dir, stale_guids)

    logger.info(
        "Deleted episode {!r} from {}: collections={} dir={} audio={} row={} warnings={}",
        stem,
        show_dir,
        report.collections,
        report.output_dir_removed,
        report.audio_removed,
        report.db_row_removed,
        len(report.warnings),
    )
    return report


def _stale_feed_guids(show_dir: Path, stem: str) -> list[str]:
    """Guids of cached feed entries for ``stem`` that are gone from the feed.

    Must be computed *before* the files are deleted: ``episode_stem`` reuses an
    existing on-disk stem for a guid whose title has since changed, so once the
    episode's directory and audio are gone the same entry can resolve to a
    different (slugified) stem and stop matching.

    Only ``removed=True`` entries qualify. An episode still in the live feed is
    re-added by the next refresh no matter what the cache says, which is the
    documented behavior and what the confirm dialog promises; pruning it here
    would only make it blink out and come back.
    """
    from podcodex.core.source import list_show_stems
    from podcodex.ingest.rss import episode_stem, load_feed_cache

    cached = load_feed_cache(show_dir)
    if not cached:
        return []
    existing = list_show_stems(show_dir)
    return [
        ep.guid
        for ep in cached
        if ep.removed and episode_stem(ep, show_dir, existing_stems=existing) == stem
    ]


def _prune_feed_cache(show_dir: Path, guids: list[str]) -> None:
    """Drop ``guids`` from the show's feed cache. Best effort.

    Without this a ``removed=True`` episode survives its own deletion:
    ``merge_with_cache`` re-emits every cached guid on each refresh, so
    ``/unified`` keeps listing an episode whose four stores are all empty and
    which the user has no remaining way to clear.
    """
    if not guids:
        return
    from podcodex.ingest.rss import load_feed_cache, save_feed_cache

    try:
        cached = load_feed_cache(show_dir)
        if not cached:
            return
        drop = set(guids)
        remaining = [ep for ep in cached if ep.guid not in drop]
        if len(remaining) != len(cached):
            save_feed_cache(show_dir, remaining)
    except Exception:
        # The episode itself is already gone; a surviving cache row is a
        # cosmetic leftover, not a reason to fail the delete.
        logger.opt(exception=True).warning(
            "delete_episode: could not prune the feed cache in {}", show_dir
        )


def drop_db_row(show_dir: Path, stem: str) -> bool:
    """Remove the episode's ``pipeline_db`` row. Returns True if one went.

    Raises on a DB error rather than reporting False, so the two callers can
    differ: the full delete turns a failure into a report warning, while the
    audio route's orphan cleanup is best-effort and logs it.
    """
    from podcodex.core.pipeline_db import get_pipeline_db

    return get_pipeline_db(show_dir).delete_episode(stem)


def episode_has_leftovers(show_dir: Path, stem: str) -> bool:
    """True when anything besides the audio file still represents this episode.

    Used by ``DELETE /api/audio/file`` to decide whether unlinking the audio
    left a ghost: a status row with no audio, no output dir and no versions,
    which ``/unified`` would keep listing forever.
    """
    # dir_holds_episode, not a bare is_dir(): the scanner's rule is what
    # decides whether a directory can rebuild a row, so a leftover holding
    # only, say, voice_samples/ would otherwise pin a row nothing can heal.
    import os

    from podcodex.ingest.folder import dir_holds_episode

    try:
        if dir_holds_episode(set(os.listdir(show_dir / stem))):
            return True
    except OSError:
        pass  # no directory, or unreadable: fall through to the version check

    from podcodex.core.pipeline_db import get_pipeline_db

    try:
        return bool(get_pipeline_db(show_dir).list_all_versions(stem))
    except Exception:
        # Can't tell, so assume something worth keeping the row for.
        logger.opt(exception=True).warning(
            "delete_episode: leftover check failed for {!r}", stem
        )
        return True
