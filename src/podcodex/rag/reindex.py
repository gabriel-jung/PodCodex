"""podcodex-reindex — rebuild a show's LanceDB index from its transcripts.

LanceDB tables are treated as derived state: they can be rebuilt from the
per-episode transcripts + versions on disk with no data loss. This command is
the recovery path after index corruption, schema changes, or when
re-embedding with a different model.

The rebuild is in place: each episode is re-embedded over its old rows, and
episodes the scan no longer produces are removed at the end, so a run that is
killed or fails part way leaves the rest of the index searchable and a rerun
picks up from there. ``--fresh`` drops the collections first instead, for an
index whose tables themselves are broken.

Usage::

    podcodex-reindex <show-folder> [--model bge-m3] [--chunker semantic]
    podcodex-reindex <show-folder> --all-models        # every model+chunker
    podcodex-reindex <show-folder> --fresh             # drop, then rebuild
    podcodex-reindex <show-folder> --list              # show current collections
    podcodex-reindex <show-folder> --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from podcodex.cli.resolve import resolve_show_folder
from podcodex.ingest.folder import scan_folder
from podcodex.rag.defaults import (
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING,
    DEFAULT_MODEL,
    MODELS,
)
from podcodex.rag.index_store import get_index_store


def _reindex_show(
    folder: Path,
    show_name: str,
    model_keys: list[str],
    chunkers: list[str],
    dry_run: bool,
    fresh: bool = False,
) -> bool:
    """Rebuild every (model × chunker) collection for a show, in place.

    Every episode with a transcript is re-embedded with ``overwrite``, which
    replaces that episode's rows in one write. Rows of episodes the scan did
    not re-index are removed afterwards, and only when no episode failed, so
    a failure never costs more than what it failed to rebuild. ``fresh``
    drops the collections up front instead (the old behaviour).

    Returns:
        True when every episode with a transcript was re-indexed.
    """
    from podcodex.core.device import device_str
    from podcodex.core.source import NoTranscriptError, build_index_transcript
    from podcodex.ingest.show_registry import show_id_for_folder
    from podcodex.rag.indexing import IndexingError, vectorize_batch

    # From the folder being reindexed, never by display name: with two shows
    # of one name, the name picked the oldest one, whose tables were dropped
    # and then rebuilt from this folder's episodes. Minted when writing, so
    # the rebuilt collections carry the id.
    sid = show_id_for_folder(folder, mint=not dry_run)
    if dry_run and not _index_exists():
        # Opening a store creates the index and stamps this machine as its
        # owner; a dry run must not, so there is nothing to drop or prune.
        logger.info("[dry-run] no index yet: every episode would be indexed fresh")
        store, targets = None, []
    else:
        store = get_index_store()
        # Strict: a show with no id owns only id-less rows. Loose, this
        # matched a same-named show that has an id, and dropped or pruned
        # its table.
        targets = [
            col
            for model in model_keys
            for chunker in chunkers
            if (
                col := store.resolve_collection(
                    sid, model, chunker, show_label=show_name, strict=True
                )
            )
        ]
    if fresh:
        for col in targets:
            if dry_run:
                logger.info(f"[dry-run] would drop collection: {col}")
                continue
            try:
                store.delete_collection(col)
                logger.info(f"Dropped {col}")
            except Exception as exc:
                # Refused on a replica, or already gone. Either way the
                # rebuild below writes over whatever is still there.
                logger.warning(f"Could not drop {col}: {exc}")

    # Re-vectorize every episode with a transcript. Audio-less episodes
    # (YouTube subtitle imports, flat extraction) index through output_dir
    # exactly as the app does; skipping them emptied a subtitle-driven show
    # on every rebuild while still reporting success.
    episodes = scan_folder(folder)
    if not episodes:
        # Nothing is pruned: a folder that resolves to nothing is far more
        # likely a wrong path than a show whose every episode was deleted.
        logger.warning(f"No episodes found in {folder}; index left as it was")
        return True

    device = "cpu" if dry_run else device_str()
    indexed: set[str] = set()
    reindexed = 0
    skipped = 0
    failed = 0
    for ep in episodes:
        audio = getattr(ep, "audio_path", None)
        try:
            transcript = build_index_transcript(
                str(audio) if audio else None,
                show_name,
                ep.stem,
                output_dir=None if audio else str(ep.output_dir),
            )
        except NoTranscriptError:
            # Never transcribed: nothing to index, and its old rows (if any)
            # are stale, so the prune below removes them.
            logger.warning(f"[skip] {ep.stem}: no transcript")
            skipped += 1
            continue
        except Exception as exc:
            # Unreadable, not absent: a failure, which keeps the episode's
            # current rows (no prune) and fails the run.
            logger.error(f"[failed] {ep.stem}: could not read its transcript: {exc}")
            failed += 1
            continue

        if dry_run:
            logger.info(
                f"[dry-run] would re-index {ep.stem} "
                f"(chunks={len(transcript.get('segments') or [])} segments)"
            )
            reindexed += 1
            indexed.add(ep.stem)
            continue

        try:
            n = vectorize_batch(
                transcript,
                show_name,
                ep.stem,
                model_keys,
                chunkers,
                store,
                show_id=sid,
                overwrite=True,
                device=device,
            )
        except IndexingError as exc:
            logger.error(f"[failed] {ep.stem}: {exc}")
            failed += 1
            continue
        if n == 0:
            logger.warning(f"[skip] {ep.stem}: transcript produced no chunks")
            skipped += 1
            continue
        logger.success(f"{ep.stem}: +{n} chunks")
        reindexed += 1
        indexed.add(ep.stem)

    # Rows the scan did not rebuild: deleted episodes, and episodes whose
    # transcript is gone or now yields nothing. Kept after any failure, so
    # the run stays resumable and never ends emptier than its failures.
    if not fresh and not failed:
        for col in targets:
            stale = [e for e in store.list_episodes(col) if e not in indexed]
            for stem in stale:
                if dry_run:
                    logger.info(f"[dry-run] would remove {stem} from {col}")
                else:
                    store.delete_episode(col, stem)
                    logger.info(f"Removed {stem} from {col}")

    if store is not None and not dry_run:
        # A full rebuild is the biggest producer of superseded Lance
        # fragments; reclaim them before the index is shipped anywhere.
        store.compact()

    summary = f"{reindexed} re-indexed, {skipped} skipped, {failed} failed"
    if failed:
        logger.error(f"Finished with failures: {summary}")
        return False
    logger.info(f"Done: {summary}")
    return True


def _index_exists() -> bool:
    from podcodex.rag.index_store import _dir_has_data, _resolve_default_index_path

    path, _reason = _resolve_default_index_path()
    return _dir_has_data(Path(path))


def _list_collections(folder: Path, show_name: str) -> None:
    """Print collections currently in LanceDB that belong to this show."""
    from podcodex.ingest.show_registry import show_id_for_folder

    store = get_index_store()
    # Resolved, not reconstructed from the name: a collection's table name is
    # internal to the store, and after a rename it no longer resembles the
    # show's current name at all.
    names = store.collections_for_show(
        show_id_for_folder(folder), show_label=show_name, strict=True
    )
    if not names:
        print(f"(no collections for {show_name!r})")
        return
    for name in names:
        print(
            f"  {name:<60} "
            f"chunks={store.count_rows(name):>6} "
            f"episodes={store.episode_count(name):>4}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="podcodex-reindex",
        description="Rebuild a show's LanceDB index from its transcripts.",
    )
    ap.add_argument("show", help="Show folder path or registered show name.")
    # `choices`, like podcodex-bot: a mistyped model reached the indexing
    # loop and died there as a bare KeyError, and a mistyped chunker was
    # never rejected at all — the dispatcher fell through to semantic and
    # the run reported success while building a collection keyed on the typo
    # that no other surface can resolve.
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODELS),
        help=f"Embedding model (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--chunker",
        default=DEFAULT_CHUNKING,
        choices=list(CHUNKING_STRATEGIES),
        help=f"Chunking strategy (default: {DEFAULT_CHUNKING}).",
    )
    ap.add_argument(
        "--all-models",
        action="store_true",
        help="Rebuild every model × chunker combination present in MODELS.",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Drop the collections before rebuilding (for broken tables). "
        "A killed or failed --fresh run leaves the show partly indexed.",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Show current collections for this show and exit.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing to LanceDB.",
    )
    args = ap.parse_args()

    folder, name = resolve_show_folder(args.show)

    if args.list:
        _list_collections(folder, name)
        return

    # Like every entry point that loads models: cache env vars and the ML
    # patches first. After --list, which needs neither (torch is seconds).
    from podcodex.bootstrap import bootstrap_for_dev

    bootstrap_for_dev()

    if args.all_models:
        model_keys = list(MODELS.keys())
        chunkers = ["semantic", "speaker"]
    else:
        model_keys = [args.model]
        chunkers = [args.chunker]

    logger.info(
        f"Reindexing {name!r} ({folder}) — "
        f"models={model_keys}, chunkers={chunkers}"
        + (" [dry-run]" if args.dry_run else "")
    )
    if not _reindex_show(
        folder, name, model_keys, chunkers, dry_run=args.dry_run, fresh=args.fresh
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
