"""
podcodex.cli — Command-line interface.

    podcodex init      <show_folder> --name <name> [--rss <url>] [--language <lang>]
    podcodex rss       <show_folder> [--rss <url>] [--episode <title>] [--download]
    podcodex import    <transcript.json> <show_folder> [--episode <stem>] [--show <name>]
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

    Automatically detects nodiar files: if the transcript_path has a ``.nodiar.``
    prefix, polished and translation lookups also use the nodiar prefix.

    ``auto`` priority chain:
        1. {stem}.polished.json (polished validated)
        2. {stem}.polished.raw.json (polished raw)
        3. transcript_path (fallback)
    """
    episode_stem = transcript_path.parent.name
    parent = transcript_path.parent
    # Detect nodiar from the transcript filename itself
    nodiar_prefix = "nodiar." if ".nodiar." in transcript_path.name else ""

    if source == "auto":
        for suffix in (
            f"{episode_stem}.{nodiar_prefix}polished.json",
            f"{episode_stem}.{nodiar_prefix}polished.raw.json",
        ):
            p = parent / suffix
            if p.exists():
                return p
        return transcript_path
    elif source == "transcript":
        return transcript_path
    elif source == "polished":
        p = parent / f"{episode_stem}.{nodiar_prefix}polished.json"
        if not p.exists():
            logger.warning(f"Polished file not found: {p} — falling back to transcript")
            return transcript_path
        return p
    else:
        # Treat as a language name (e.g. "english", "French")
        lang_norm = source.lower().strip().replace(" ", "_")
        p = parent / f"{episode_stem}.{nodiar_prefix}{lang_norm}.json"
        if not p.exists():
            logger.warning(
                f"Translation '{lang_norm}' not found: {p} — falling back to transcript"
            )
            return transcript_path
        return p


def _source_label(source_path: Path, transcript_path: Path) -> str:
    """Derive a human-readable source label from the resolved path.

    Examples: 'polished', 'english', 'transcript'.

    Uses the transcript stem (directory name) to reliably extract the
    qualifier even when the episode stem itself contains dots.
    """
    if source_path == transcript_path:
        return "transcript"
    # Strip the episode stem prefix to isolate the qualifier.
    # e.g. stem="ep.01", name="ep.01.polished.raw.json"
    #       → remainder="polished.raw.json" → label="polished"
    episode_stem = transcript_path.parent.name
    remainder = source_path.name
    if remainder.startswith(episode_stem + "."):
        remainder = remainder[len(episode_stem) + 1 :]
    # remainder is now e.g. "polished.json" or "nodiar.polished.raw.json" or "english.json"
    # Strip "nodiar." prefix so the label reflects the actual source type
    if remainder.startswith("nodiar."):
        remainder = remainder[len("nodiar.") :]
    label = remainder.split(".")[0]
    return label or "transcript"


def _chunk_transcript(
    transcript: dict,
    chunking: str,
    chunk_size: int = CHUNK_SIZE,
    threshold: float = CHUNK_THRESHOLD,
) -> list[dict]:
    """Chunk a transcript using the given strategy. Returns empty list on failure."""
    if chunking == "speaker":
        if not transcript.get("meta", {}).get("diarized", True):
            logger.warning(
                "Speaker chunking is not useful without diarization — "
                "falling back to semantic."
            )
            return semantic_chunks(
                transcript, chunk_size=chunk_size, threshold=threshold
            )
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
    *,
    chunks: list[dict] | None = None,
    chunk_size: int = CHUNK_SIZE,
    threshold: float = CHUNK_THRESHOLD,
    overwrite: bool = False,
    device: str = "cpu",
) -> tuple[list[dict], int]:
    """
    Vectorize a single (model, chunker) combination into LocalStore.

    Qdrant is not touched — use ``podcodex sync`` to push later.

    Args:
        transcript : parsed transcript dict (with meta.show / meta.episode set)
        show, episode : identifiers
        model_key, chunking : which model and chunker to use
        local : LocalStore instance (SQLite)
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

    # ── Already in LocalStore? ──
    if local.episode_is_indexed(col, episode) and not overwrite:
        # Stale source detection: auto-upgrade if a better source is now available
        new_source = transcript.get("meta", {}).get("source", "")
        stored = local.load_chunks_no_embeddings(col, episode)
        stored_source = stored[0].get("source", "") if stored else ""

        if stored_source and new_source and stored_source != new_source:
            logger.info(
                f"[UPGRADE] '{episode}' source changed: "
                f"{stored_source} → {new_source} ({col})"
            )
            local.delete_episode(col, episode)
            # Fall through to re-chunk + re-embed below
        else:
            local_count = local.episode_chunk_count(col, episode)
            logger.info(f"[SKIP] '{episode}' cached ({col}, {local_count} chunks)")
            return chunks or stored, 0

    if overwrite and local.episode_is_indexed(col, episode):
        local.delete_episode(col, episode)

    # ── Chunk (reuse if already computed for this strategy) ──
    if chunks is None:
        chunks = _chunk_transcript(transcript, chunking, chunk_size, threshold)
    if not chunks:
        raise ValueError(f"No chunks produced for strategy '{chunking}'")

    # ── Embed & save to LocalStore ──
    embedder = get_embedder(model_key, device=device)
    embeddings = embedder.encode_passages(chunks)
    local.save_chunks(col, episode, chunks, embeddings)

    logger.success(f"Vectorized {len(chunks)} chunks into '{col}'")
    return chunks, len(chunks)


def vectorize_batch(
    transcript: dict,
    show: str,
    episode: str,
    model_keys: list[str],
    chunkings: list[str],
    local: "LocalStore",
    *,
    chunk_size: int = CHUNK_SIZE,
    threshold: float = CHUNK_THRESHOLD,
    overwrite: bool = False,
    device: str = "cpu",
    on_progress: Callable[[int, int, str], None] | None = None,
) -> int:
    """
    Vectorize all (model, chunker) combinations into LocalStore.

    Chunks once per strategy, then embeds with each model.
    Qdrant is not touched — use ``podcodex sync`` to push later.

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
                    chunks=chunks_for_strategy,
                    chunk_size=chunk_size,
                    threshold=threshold,
                    overwrite=overwrite,
                    device=device,
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
    transcript["meta"]["source"] = _source_label(source_path, transcript_path)

    logger.info(f"Vectorizing '{episode}' for show '{show}' with model '{args.model}'")

    # Show-level DB: episode dir is transcript_path.parent, show dir is one up
    db_path = transcript_path.parent.parent / "vectors.db"
    local = LocalStore(db_path=db_path)

    try:
        chunks, n = vectorize_episode(
            transcript,
            show,
            episode,
            args.model,
            args.chunking,
            local,
            chunk_size=args.chunk_size,
            threshold=args.threshold,
            overwrite=args.overwrite,
            device=args.device,
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

            local_count = len(cached)

            if not args.overwrite:
                qdrant_count = store.episode_point_count(col, ep)
                if local_count == qdrant_count:
                    logger.info(f"[SKIP] '{ep}' → '{col}' ({local_count} chunks)")
                    continue
                if qdrant_count > 0:
                    store.delete_episode_points(col, ep)

            chunks = [{k: v for k, v in c.items() if k != "embedding"} for c in cached]
            embeddings = np.stack([c["embedding"] for c in cached])
            store.upsert(col, chunks, embeddings)
            total_chunks += len(chunks)
            logger.info(f"Synced '{ep}' → '{col}' ({len(chunks)} chunks)")

    logger.success(f"Sync complete — {total_chunks} chunks pushed to Qdrant")


def cmd_query(args: argparse.Namespace) -> None:
    col = collection_name(args.show, args.model, args.chunking)
    retriever = Retriever(model=args.model)
    source_filter = getattr(args, "source_filter", None)
    results = retriever.retrieve(
        args.query, col, top_k=args.top_k, alpha=args.alpha, source=source_filter
    )

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


def cmd_init(args: argparse.Namespace) -> None:
    from podcodex.ingest.show import ShowMeta, load_show_meta, save_show_meta

    folder = Path(args.show_folder)
    existing = load_show_meta(folder)
    if existing and not args.overwrite:
        logger.info(
            f"show.toml already exists for '{existing.name}' — use --overwrite to replace"
        )
        return

    meta = ShowMeta(
        name=args.name,
        rss_url=args.rss or "",
        language=args.language or "",
        speakers=[s.strip() for s in args.speakers.split(",")] if args.speakers else [],
    )
    path = save_show_meta(folder, meta)
    logger.success(f"Created {path}")


def cmd_rss(args: argparse.Namespace) -> None:
    from podcodex.ingest.rss import (
        download_audio,
        fetch_feed,
        save_feed_cache,
        episode_stem as rss_episode_stem,
    )
    from podcodex.ingest.show import load_show_meta

    folder = Path(args.show_folder)
    meta = load_show_meta(folder)

    rss_url = args.rss or (meta.rss_url if meta else "")
    if not rss_url:
        logger.error("No RSS URL — pass --rss or set rss_url in show.toml")
        sys.exit(1)

    logger.info(f"Fetching {rss_url}")
    episodes = fetch_feed(rss_url)
    if not episodes:
        logger.warning("No episodes found in feed.")
        return

    save_feed_cache(folder, episodes)

    # Filter to single episode if requested
    if args.episode:
        q = args.episode.lower()
        episodes = [
            ep for ep in episodes if q in ep.title.lower() or q in ep.guid.lower()
        ]
        if not episodes:
            logger.error(f"No episode matching '{args.episode}'")
            sys.exit(1)

    new_count = 0
    existing_count = 0
    downloaded_count = 0

    for ep in episodes:
        stem = rss_episode_stem(ep)
        output_dir = folder / stem

        if output_dir.exists():
            existing_count += 1
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            ep_meta = {
                "guid": ep.guid,
                "title": ep.title,
                "pub_date": ep.pub_date,
                "description": ep.description,
                "audio_url": ep.audio_url,
                "duration": ep.duration,
            }
            (output_dir / ".episode_meta.json").write_text(
                json.dumps(ep_meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            new_count += 1

        has_audio = "🎵" if ep.audio_url else "  "
        print(f"  {has_audio} {stem}  ←  {ep.title}")

        if args.download and ep.audio_url:
            result = download_audio(ep, folder)
            if result:
                downloaded_count += 1

    logger.success(
        f"{new_count} new, {existing_count} existing"
        + (f", {downloaded_count} downloaded" if args.download else "")
    )


def cmd_import(args: argparse.Namespace) -> None:
    from podcodex.ingest.importer import import_transcript
    from podcodex.ingest.show import load_show_meta

    transcript_path = Path(args.transcript)
    show_folder = Path(args.show_folder)

    if not transcript_path.exists():
        logger.error(f"Transcript not found: {transcript_path}")
        sys.exit(1)

    # Resolve show name
    meta = load_show_meta(show_folder)
    show_name = args.show or (meta.name if meta else "")
    if not show_name:
        logger.error(
            "No show name — pass --show or create show.toml with podcodex init"
        )
        sys.exit(1)

    # Resolve episode stem
    episode_stem = args.episode or transcript_path.stem

    dest = import_transcript(transcript_path, show_folder, episode_stem, show_name)
    print(f"Imported → {dest}")


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

    # init
    p_init = sub.add_parser("init", help="Create show.toml for a show folder.")
    p_init.add_argument("show_folder", help="Path to the show folder.")
    p_init.add_argument("--name", required=True, help="Canonical show name.")
    p_init.add_argument("--rss", default=None, help="RSS feed URL.")
    p_init.add_argument("--language", default=None, help="Primary language.")
    p_init.add_argument(
        "--speakers",
        default=None,
        help="Comma-separated list of known speakers.",
    )
    p_init.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing show.toml.",
    )

    # rss
    p_rss = sub.add_parser(
        "rss", help="Fetch RSS feed, register episodes, optionally download audio."
    )
    p_rss.add_argument("show_folder", help="Path to the show folder.")
    p_rss.add_argument(
        "--rss", default=None, help="RSS feed URL (overrides show.toml)."
    )
    p_rss.add_argument(
        "--episode", default=None, help="Filter to a single episode by title or guid."
    )
    p_rss.add_argument(
        "--download",
        action="store_true",
        help="Download audio files for episodes with enclosures.",
    )

    # import
    p_imp = sub.add_parser(
        "import", help="Import an external transcript into a show folder."
    )
    p_imp.add_argument(
        "transcript", metavar="transcript.json", help="Path to transcript JSON."
    )
    p_imp.add_argument("show_folder", help="Path to the show folder.")
    p_imp.add_argument(
        "--episode", default=None, help="Episode stem (default: transcript filename)."
    )
    p_imp.add_argument(
        "--show", default=None, help="Show name (default: from show.toml)."
    )

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
        default="auto",
        metavar="SOURCE",
        help=(
            "Which file to vectorize: 'auto' (default — picks best available), "
            "'transcript' (raw), 'polished', or a language name such as 'english'."
        ),
    )
    p_vec.add_argument(
        "--device",
        default="cpu",
        help="Torch device for embedding (default: cpu). Use 'cuda' or 'mps' for GPU.",
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
    p_query.add_argument(
        "--source-filter",
        default=None,
        dest="source_filter",
        metavar="SOURCE",
        help="Only search chunks from this source (e.g. 'polished', 'transcript').",
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

    if args.command == "init":
        cmd_init(args)
    elif args.command == "rss":
        cmd_rss(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "vectorize":
        cmd_vectorize(args)
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)
