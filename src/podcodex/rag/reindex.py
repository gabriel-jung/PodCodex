"""podcodex-reindex — rebuild a show's LanceDB index from its transcripts.

LanceDB tables are treated as derived state: they can be wiped and rebuilt
from the per-episode transcripts + versions on disk with no data loss.
This command is the recovery path after index corruption, schema changes,
or when re-embedding with a different model.

Usage::

    podcodex-reindex <show-folder> [--model bge-m3] [--chunker semantic]
    podcodex-reindex <show-folder> --all-models        # every model+chunker
    podcodex-reindex <show-folder> --list              # show current collections
    podcodex-reindex <show-folder> --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from podcodex.cli.resolve import resolve_show_folder
from podcodex.ingest.folder import scan_folder
from podcodex.rag.defaults import DEFAULT_CHUNKING, DEFAULT_MODEL
from podcodex.rag.index_store import get_index_store
from podcodex.rag.store import collection_name


def _reindex_show(
    folder: Path,
    show_name: str,
    model_keys: list[str],
    chunkers: list[str],
    dry_run: bool,
) -> None:
    """Drop then rebuild every (model × chunker) collection for a show."""
    from podcodex.api.routes._helpers import build_index_transcript
    from podcodex.rag.indexing import vectorize_batch

    store = get_index_store()

    # 1. Drop every target collection so stale rows can't linger.
    for model in model_keys:
        for chunker in chunkers:
            col = collection_name(show_name, model, chunker)
            if dry_run:
                logger.info(f"[dry-run] would drop collection: {col}")
            else:
                try:
                    store.delete_collection(col)
                    logger.info(f"Dropped {col}")
                except Exception as exc:
                    logger.debug(f"Nothing to drop for {col}: {exc}")

    # 2. Re-vectorize every episode that has audio + a transcript.
    episodes = scan_folder(folder)
    if not episodes:
        logger.warning(f"No episodes found in {folder}")
        return

    reindexed = 0
    skipped = 0
    for ep in episodes:
        if not getattr(ep, "audio_path", None):
            continue
        try:
            transcript = build_index_transcript(
                str(ep.audio_path), show_name, ep.audio_path.stem
            )
        except Exception as exc:
            logger.warning(f"[skip] {ep.audio_path.stem}: {exc}")
            skipped += 1
            continue

        if dry_run:
            logger.info(
                f"[dry-run] would re-index {ep.audio_path.stem} "
                f"(chunks={len(transcript.get('segments') or [])} segments)"
            )
            reindexed += 1
            continue

        n = vectorize_batch(
            transcript,
            show_name,
            ep.audio_path.stem,
            model_keys,
            chunkers,
            store,
            overwrite=True,
        )
        logger.success(f"{ep.audio_path.stem}: +{n} chunks")
        reindexed += 1

    logger.info(f"Done: {reindexed} re-indexed, {skipped} skipped")


def _list_collections(show_name: str) -> None:
    """Print collections currently in LanceDB that belong to this show."""
    store = get_index_store()
    prefix = f"{show_name.lower().replace(' ', '_')}__"
    info = store.get_all_collection_info()
    matches = [row for row in info if row["name"].startswith(prefix)]
    if not matches:
        print(f"(no collections for {show_name!r})")
        return
    for row in matches:
        print(
            f"  {row['name']:<60} "
            f"chunks={row.get('chunk_count', 0):>6} "
            f"episodes={row.get('episode_count', 0):>4}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="podcodex-reindex",
        description="Rebuild a show's LanceDB index from its transcripts.",
    )
    ap.add_argument("show", help="Show folder path or registered show name.")
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--chunker",
        default=DEFAULT_CHUNKING,
        help=f"Chunking strategy (default: {DEFAULT_CHUNKING}).",
    )
    ap.add_argument(
        "--all-models",
        action="store_true",
        help="Rebuild every model × chunker combination present in MODELS.",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Show current collections for this show and exit.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without touching LanceDB.",
    )
    args = ap.parse_args()

    folder, name = resolve_show_folder(args.show)

    if args.list:
        _list_collections(name)
        return

    if args.all_models:
        from podcodex.rag.defaults import MODELS

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
    _reindex_show(folder, name, model_keys, chunkers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
