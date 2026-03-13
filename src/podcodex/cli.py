"""
podcodex.cli — Command-line interface.

    podcodex vectorize <transcript.json> --show <name>
                       [--model bge-m3] [--chunking semantic|speaker]
                       [--chunk-size N] [--threshold F]
                       [--episode <name>] [--overwrite]
    podcodex sync      --show <name> [--episode <name>] [--overwrite]
                       [--qdrant-url URL] [--db PATH]
    podcodex query     <QUERY> --show <name> [--top-k N] [--alpha F]
    podcodex list      [--show <name>]
    podcodex delete    <collection>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from loguru import logger

from podcodex.rag.chunker import semantic_chunks, speaker_chunks
from podcodex.rag.embedder import get_embedder
from podcodex.rag.defaults import (
    ALPHA,
    CHUNK_SIZE,
    CHUNK_THRESHOLD,
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING,
    DEFAULT_MODEL,
    MODELS,
    TOP_K,
)
from podcodex.rag.localstore import LocalStore
from podcodex.rag.retriever import Retriever
from podcodex.rag.store import QdrantStore, collection_name


# ──────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────


def _resolve_source(transcript_path: Path, source: str) -> Path:
    """Return the file to vectorize based on --source.

    The episode stem is inferred from the output dir name (= transcript parent).
    Falls back to transcript_path if the requested file does not exist.
    """
    episode_stem = transcript_path.parent.name
    parent = transcript_path.parent

    if source == "transcript":
        return transcript_path
    elif source == "polished":
        p = parent / f"{episode_stem}.polished.json"
        if not p.exists():
            logger.warning(f"Polished file not found: {p} — falling back to transcript")
            return transcript_path
        return p
    else:
        # Treat as a language name (e.g. "english", "French")
        lang_norm = source.lower().strip().replace(" ", "_")
        p = parent / f"{episode_stem}.{lang_norm}.json"
        if not p.exists():
            logger.warning(
                f"Translation '{lang_norm}' not found: {p} — falling back to transcript"
            )
            return transcript_path
        return p


def _chunk_transcript(
    transcript: dict,
    chunking: str,
    chunk_size: int = CHUNK_SIZE,
    threshold: float = CHUNK_THRESHOLD,
) -> list[dict]:
    """Chunk a transcript using the given strategy. Returns empty list on failure."""
    if chunking == "speaker":
        return speaker_chunks(transcript)
    else:
        return semantic_chunks(transcript, chunk_size=chunk_size, threshold=threshold)


def vectorize_episode(
    transcript: dict,
    show: str,
    episode: str,
    model_key: str,
    chunking: str,
    local: "LocalStore",
    store: "QdrantStore",
    *,
    chunks: list[dict] | None = None,
    chunk_size: int = CHUNK_SIZE,
    threshold: float = CHUNK_THRESHOLD,
    overwrite: bool = False,
) -> tuple[list[dict], int]:
    """
    Vectorize a single (model, chunker) combination.

    Args:
        transcript : parsed transcript dict (with meta.show / meta.episode set)
        show, episode : identifiers
        model_key, chunking : which model and chunker to use
        local : LocalStore instance (SQLite)
        store : QdrantStore instance (Qdrant)
        chunks : pre-computed chunks for this chunking strategy (avoids re-chunking)
        chunk_size, threshold : semantic chunking params
        overwrite : delete and recreate if already indexed

    Returns:
        (chunks, n_embedded) — the chunks list (for reuse across models)
        and the count of chunks upserted (0 if skipped/cached).

    Raises:
        ValueError: if no chunks could be produced.
    """
    col = collection_name(show, model_key, chunking)
    dim = MODELS[model_key].dim
    local.ensure_collection(col, show=show, model=model_key, chunker=chunking, dim=dim)

    # ── Cached in LocalStore → load and push ──
    if local.episode_is_indexed(col, episode) and not overwrite:
        logger.info(f"[SKIP] '{episode}' already in local store ({col})")
        cached = local.load_chunks(col, episode)
        payload = [{k: v for k, v in c.items() if k != "embedding"} for c in cached]
        embeddings = np.stack([c["embedding"] for c in cached])
        store.create_collection(col, model=model_key, overwrite=False)
        store.upsert(col, payload, embeddings)
        return chunks or payload, 0

    if overwrite and local.episode_is_indexed(col, episode):
        local.delete_episode(col, episode)

    # ── Chunk (reuse if already computed for this strategy) ──
    if chunks is None:
        chunks = _chunk_transcript(transcript, chunking, chunk_size, threshold)
    if not chunks:
        raise ValueError(f"No chunks produced for strategy '{chunking}'")

    # ── Embed ──
    embedder = get_embedder(model_key)
    embeddings = embedder.encode_passages(chunks)
    local.save_chunks(col, episode, chunks, embeddings)

    # ── Push to Qdrant ──
    store.create_collection(col, model=model_key, overwrite=overwrite)
    store.upsert(col, chunks, embeddings)

    logger.success(f"Vectorized {len(chunks)} chunks into '{col}'")
    return chunks, len(chunks)


def vectorize_batch(
    transcript: dict,
    show: str,
    episode: str,
    model_keys: list[str],
    chunkings: list[str],
    local: "LocalStore",
    store: "QdrantStore",
    *,
    chunk_size: int = CHUNK_SIZE,
    threshold: float = CHUNK_THRESHOLD,
    overwrite: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> int:
    """
    Vectorize all (model, chunker) combinations for an episode.

    Chunks once per strategy, then embeds with each model.

    Args:
        on_progress : callback(step, total, label) for UI progress updates.

    Returns:
        Total number of chunks upserted across all combinations.
    """
    total = len(model_keys) * len(chunkings)
    step = 0
    total_upserted = 0

    for chunking in chunkings:
        chunks_for_strategy: list[dict] | None = None

        for model_key in model_keys:
            label = f"{MODELS[model_key].label} / {chunking}"
            if on_progress:
                on_progress(step, total, label)

            try:
                chunks_for_strategy, n = vectorize_episode(
                    transcript,
                    show,
                    episode,
                    model_key,
                    chunking,
                    local,
                    store,
                    chunks=chunks_for_strategy,
                    chunk_size=chunk_size,
                    threshold=threshold,
                    overwrite=overwrite,
                )
                total_upserted += n
            except ValueError as e:
                logger.warning(str(e))
                step += len(model_keys) - model_keys.index(model_key)
                break
            except Exception:
                logger.exception(f"Failed for {label}")

            step += 1

    return total_upserted


def cmd_vectorize(args: argparse.Namespace) -> None:
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        logger.error(f"Transcript file not found: {transcript_path}")
        sys.exit(1)

    source_path = _resolve_source(transcript_path, args.source)
    logger.info(f"Source: {source_path.name}")

    data = json.loads(source_path.read_text(encoding="utf-8"))
    transcript = data if isinstance(data, dict) else {"meta": {}, "segments": data}

    meta = transcript.get("meta", {})
    episode = args.episode or meta.get("episode") or transcript_path.stem
    show = args.show

    transcript.setdefault("meta", {})
    transcript["meta"].setdefault("show", show)
    transcript["meta"].setdefault("episode", episode)

    logger.info(f"Vectorizing '{episode}' for show '{show}' with model '{args.model}'")

    db_path = transcript_path.parent / "vectors.db"
    local = LocalStore(db_path=db_path)
    qdrant_url = getattr(args, "qdrant_url", None)
    store = QdrantStore(url=qdrant_url)

    try:
        chunks, n = vectorize_episode(
            transcript,
            show,
            episode,
            args.model,
            args.chunking,
            local,
            store,
            chunk_size=args.chunk_size,
            threshold=args.threshold,
            overwrite=args.overwrite,
        )
    except ValueError as e:
        logger.warning(str(e))
        return

    marker = transcript_path.parent / ".rag_indexed"
    marker.touch()


def cmd_sync(args: argparse.Namespace) -> None:
    """Push all episodes from LocalStore into Qdrant (no re-embedding)."""
    db_path = getattr(args, "db", None)
    local = LocalStore(db_path=db_path)
    qdrant_url = getattr(args, "qdrant_url", None)
    store = QdrantStore(url=qdrant_url)

    show_filter = args.show or ""
    episode_filter = getattr(args, "episode", None)

    if episode_filter and not show_filter:
        logger.error("--episode requires --show")
        sys.exit(1)

    collections = local.list_collections(show=show_filter)
    if not collections:
        logger.warning("No collections found in local store.")
        return

    total_chunks = 0
    for col in collections:
        info = local.get_collection_info(col)
        if info is None:
            continue

        store.create_collection(col, model=info["model"], overwrite=args.overwrite)

        episodes = local.list_episodes(col)
        if episode_filter:
            episodes = [ep for ep in episodes if ep == episode_filter]

        for ep in episodes:
            cached = local.load_chunks(col, ep)
            if not cached:
                continue
            chunks = [{k: v for k, v in c.items() if k != "embedding"} for c in cached]
            embeddings = np.stack([c["embedding"] for c in cached])
            store.upsert(col, chunks, embeddings)
            total_chunks += len(chunks)
            logger.info(f"Synced '{ep}' → '{col}' ({len(chunks)} chunks)")

    logger.success(f"Sync complete — {total_chunks} chunks pushed to Qdrant")


def cmd_query(args: argparse.Namespace) -> None:
    col = collection_name(args.show, args.model, args.chunking)
    retriever = Retriever(model=args.model)
    results = retriever.retrieve(args.query, col, top_k=args.top_k, alpha=args.alpha)

    if not results:
        print("No results.")
        return

    for res in results:
        score = res.get("score", 0.0)
        speaker = res.get("dominant_speaker", res.get("speaker", "?"))
        start = res.get("start", 0.0)
        end = res.get("end", 0.0)
        episode = res.get("episode", "")
        text = res.get("text", "")
        print(f"[{score:.3f}] [{start:.1f}s–{end:.1f}s] {speaker} ({episode})")
        print(f"  {text}")
        print()


def cmd_list(args: argparse.Namespace) -> None:
    store = QdrantStore()
    collections = store.list_collections(show=args.show or "")
    if not collections:
        print("No collections found.")
    else:
        for name in collections:
            print(name)


def cmd_delete(args: argparse.Namespace) -> None:
    store = QdrantStore()
    store.delete_collection(args.collection)
    print(f"Deleted collection '{args.collection}'")


# ──────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcodex",
        description="AI tools for podcast production.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # vectorize
    p_vec = sub.add_parser("vectorize", help="Chunk, embed and store a transcript.")
    p_vec.add_argument(
        "transcript",
        metavar="transcript.json",
        help="Path to the transcript JSON file.",
    )
    p_vec.add_argument("--show", required=True, help="Show name.")
    p_vec.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Embedding model (default: {DEFAULT_MODEL}).",
    )
    p_vec.add_argument(
        "--chunking",
        default=DEFAULT_CHUNKING,
        choices=list(CHUNKING_STRATEGIES.keys()),
        help=f"Chunking strategy (default: {DEFAULT_CHUNKING}).",
    )
    p_vec.add_argument(
        "--episode",
        default=None,
        help="Episode name (default: from transcript meta or filename stem).",
    )
    p_vec.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        dest="chunk_size",
        help=f"Max tokens per semantic chunk (default: {CHUNK_SIZE}).",
    )
    p_vec.add_argument(
        "--threshold",
        type=float,
        default=CHUNK_THRESHOLD,
        help=f"Semantic similarity threshold for chunk splitting (default: {CHUNK_THRESHOLD}).",
    )
    p_vec.add_argument(
        "--source",
        default="polished",
        metavar="SOURCE",
        help=(
            "Which file to vectorize: 'transcript' (raw), 'polished' (default), "
            "or a language name such as 'english' for a translation file."
        ),
    )
    p_vec.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the collection if it already exists.",
    )

    # sync
    p_sync = sub.add_parser(
        "sync", help="Push episodes from local SQLite store into Qdrant."
    )
    p_sync.add_argument("--show", default=None, help="Filter by show name.")
    p_sync.add_argument(
        "--episode", default=None, help="Sync a single episode (requires --show)."
    )
    p_sync.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate Qdrant collections before pushing.",
    )
    p_sync.add_argument(
        "--qdrant-url",
        default=None,
        dest="qdrant_url",
        help="Qdrant server URL (default: QDRANT_URL env var or localhost:6333).",
    )
    p_sync.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help="Path to local SQLite database (default: PODCODEX_DB env var or global default).",
    )

    # query
    p_query = sub.add_parser("query", help="Search a vectorized show.")
    p_query.add_argument("query", metavar="QUERY", help="Query string.")
    p_query.add_argument("--show", required=True, help="Show name.")
    p_query.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Embedding model (default: {DEFAULT_MODEL}).",
    )
    p_query.add_argument(
        "--chunking",
        default=DEFAULT_CHUNKING,
        choices=list(CHUNKING_STRATEGIES.keys()),
        help=f"Chunking strategy (default: {DEFAULT_CHUNKING}).",
    )
    p_query.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        dest="top_k",
        help=f"Number of results to return (default: {TOP_K}).",
    )
    p_query.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help=f"Blend between BM25 (0.0) and dense vector (1.0). Default: {ALPHA}.",
    )

    # list
    p_list = sub.add_parser("list", help="List vector collections.")
    p_list.add_argument("--show", default=None, help="Filter by show name.")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a vector collection.")
    p_delete.add_argument("collection", help="Collection name to delete.")

    return parser


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "vectorize":
        cmd_vectorize(args)
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)
