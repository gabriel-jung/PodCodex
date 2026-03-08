"""
podcodex.cli — Command-line interface.

    podcodex vectorize <transcript.json> --show <name>
                       [--chunking semantic|speaker] [--chunk-size N] [--threshold F]
                       [--episode <name>] [--overwrite]
    podcodex query     <QUERY> --show <name> [--top-k N] [--alpha F]
    podcodex list      [--show <name>]
    podcodex delete    <collection>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from podcodex.rag.defaults import (
    ALPHA,
    CHUNK_SIZE,
    CHUNK_THRESHOLD,
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING,
    TOP_K,
)


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


def cmd_vectorize(args: argparse.Namespace) -> None:
    from podcodex.rag.chunker import semantic_chunks
    from podcodex.rag.embedder import BGEEmbedder
    from podcodex.rag.store import QdrantStore, _normalize_show

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

    # Inject episode/show into meta so chunker picks them up
    transcript.setdefault("meta", {})
    transcript["meta"].setdefault("show", show)
    transcript["meta"].setdefault("episode", episode)

    logger.info(f"Vectorizing '{episode}' for show '{show}'")

    if args.chunking == "speaker":
        from podcodex.rag.chunker import speaker_chunks

        chunks = speaker_chunks(transcript)
    else:
        chunks = semantic_chunks(
            transcript,
            chunk_size=args.chunk_size,
            threshold=args.threshold,
        )
    if not chunks:
        logger.warning("No chunks produced — nothing to vectorize.")
        return

    embedder = BGEEmbedder()
    embeddings = embedder.encode_passages(chunks)

    store = QdrantStore()
    col = _normalize_show(show)
    store.create_collection(col, overwrite=args.overwrite)
    store.upsert(col, chunks, embeddings)

    marker = transcript_path.parent / ".rag_indexed"
    marker.touch()

    logger.success(f"Vectorized {len(chunks)} chunks into '{col}'")


def cmd_query(args: argparse.Namespace) -> None:
    from podcodex.rag.retriever import Retriever
    from podcodex.rag.store import _normalize_show

    col = _normalize_show(args.show)
    retriever = Retriever()
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
    from podcodex.rag.store import QdrantStore

    store = QdrantStore()
    collections = store.list_collections(show=args.show or "")
    if not collections:
        print("No collections found.")
    else:
        for name in collections:
            print(name)


def cmd_delete(args: argparse.Namespace) -> None:
    from podcodex.rag.store import QdrantStore

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

    # query
    p_query = sub.add_parser("query", help="Search a vectorized show.")
    p_query.add_argument("query", metavar="QUERY", help="Query string.")
    p_query.add_argument("--show", required=True, help="Show name.")
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
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)
