"""
podcodex.cli — Command-line interface.

    podcodex vectorize <transcript.json> --show <name> --strategy <strategy>
                       [--episode <name>] [--overwrite]
    podcodex query     <QUERY> --show <name> --strategy <strategy> [--top-k N]
    podcodex list      [--show <name>]
    podcodex delete    <collection>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

_SPEAKER_STRATEGIES = {"pplx_context", "bge_speaker"}
_SEMANTIC_STRATEGIES = {"e5_semantic", "bge_semantic"}


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _load_transcript(path: Path) -> dict:
    """Load a transcript JSON file into the standard {meta, segments} dict."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    return {"meta": {}, "segments": data}


def _chunk(transcript: dict, strategy: str) -> list[dict]:
    if strategy in _SPEAKER_STRATEGIES:
        from podcodex.rag.chunker import speaker_chunks

        return speaker_chunks(transcript)
    else:
        from podcodex.rag.chunker import semantic_chunks

        return semantic_chunks(transcript)


def _embed(chunks: list[dict], strategy: str):
    if strategy == "pplx_context":
        from podcodex.rag.embedder import PplxEmbedder

        return PplxEmbedder().encode_passages(chunks)
    elif strategy == "e5_semantic":
        from podcodex.rag.embedder import E5Embedder

        return E5Embedder().encode_passages(chunks)
    else:  # bge_speaker, bge_semantic
        from podcodex.rag.embedder import BGEEmbedder

        return BGEEmbedder().encode_passages(chunks)


# ──────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────


def cmd_vectorize(args: argparse.Namespace) -> None:
    from podcodex.rag.store import QdrantStore, collection_name

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        logger.error(f"Transcript file not found: {transcript_path}")
        sys.exit(1)

    transcript = _load_transcript(transcript_path)
    meta = transcript.get("meta", {})
    episode = args.episode or meta.get("episode") or transcript_path.stem
    show = args.show

    logger.info(f"Vectorizing '{episode}' for show '{show}' (strategy={args.strategy})")

    chunks = _chunk(transcript, args.strategy)
    if not chunks:
        logger.warning("No chunks produced — nothing to vectorize.")
        return

    embeddings = _embed(chunks, args.strategy)

    store = QdrantStore()
    col_name = collection_name(show, args.strategy)
    store.create_collection(col_name, args.strategy, overwrite=args.overwrite)
    store.upsert(col_name, chunks, embeddings)

    marker = transcript_path.parent / ".rag_indexed"
    marker.touch()

    logger.success(f"Vectorized {len(chunks)} chunks into '{col_name}'")


def cmd_query(args: argparse.Namespace) -> None:
    from podcodex.rag.retriever import Retriever
    from podcodex.rag.store import collection_name

    col = collection_name(args.show, args.strategy)
    retriever = Retriever(args.strategy)
    results = retriever.retrieve(args.query, col, top_k=args.top_k)

    if not results:
        print("No results.")
        return

    for res in results:
        score = res.get("score", 0.0)
        speaker = res.get("speaker", "?")
        start = res.get("start", 0.0)
        end = res.get("end", 0.0)
        text = res.get("text", "")
        print(f"[{score:.3f}] [{start:.1f}s–{end:.1f}s] {speaker}")
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
        "--strategy",
        required=True,
        choices=("pplx_context", "e5_semantic", "bge_speaker", "bge_semantic"),
        help="Embedding strategy.",
    )
    p_vec.add_argument(
        "--episode",
        default=None,
        help="Episode name (default: from transcript meta or filename stem).",
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
        "--strategy",
        required=True,
        choices=("pplx_context", "e5_semantic", "bge_speaker", "bge_semantic"),
        help="Strategy used when vectorizing.",
    )
    p_query.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of results to return (default: 5).",
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
