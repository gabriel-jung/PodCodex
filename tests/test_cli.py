"""Tests for podcodex.cli — all heavy deps mocked."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_transcript_file(tmp_path, meta=None, episode_name="my_episode"):
    data = {
        "meta": meta or {"show": "TestShow", "episode": episode_name},
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "Alice",
                "text": "Hello world this is a valid segment that is long enough.",
            },
        ],
    }
    p = tmp_path / f"{episode_name}.json"
    p.write_text(json.dumps(data))
    return p


def _make_vectorize_args(
    transcript,
    show,
    model="bge-m3",
    chunking="semantic",
    episode=None,
    chunk_size=256,
    threshold=0.5,
    overwrite=False,
    source="transcript",
):
    args = MagicMock()
    args.transcript = str(transcript)
    args.show = show
    args.model = model
    args.chunking = chunking
    args.episode = episode
    args.chunk_size = chunk_size
    args.threshold = threshold
    args.overwrite = overwrite
    args.source = source
    args.qdrant_url = None
    return args


def _mock_local_not_indexed():
    """LocalStore mock where episode is NOT already cached."""
    mock = MagicMock()
    mock.episode_is_indexed.return_value = False
    return mock


# ──────────────────────────────────────────────
# cmd_vectorize
# ──────────────────────────────────────────────


def test_cmd_vectorize_calls_store_methods(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="my_show")

    mock_chunks = [{"text": "chunk1", "start": 0.0, "end": 5.0}]
    mock_embeddings = np.zeros((1, 1024), dtype=np.float32)
    mock_embedder = MagicMock()
    mock_embedder.encode_passages.return_value = mock_embeddings
    mock_store = MagicMock()
    mock_local = _mock_local_not_indexed()

    with (
        patch("podcodex.cli.semantic_chunks", return_value=mock_chunks),
        patch("podcodex.cli.get_embedder", return_value=mock_embedder),
        patch("podcodex.cli.QdrantStore", return_value=mock_store),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_store.create_collection.assert_called_once_with(
        "my_show__bge-m3__semantic", model="bge-m3", overwrite=False
    )
    mock_store.upsert.assert_called_once_with(
        "my_show__bge-m3__semantic", mock_chunks, mock_embeddings
    )


def test_cmd_vectorize_episode_from_args(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="s", episode="custom_ep")

    mock_local = _mock_local_not_indexed()
    with (
        patch("podcodex.cli.semantic_chunks", return_value=[{"text": "t"}]),
        patch(
            "podcodex.rag.embedder.get_embedder",
            return_value=MagicMock(
                encode_passages=MagicMock(
                    return_value=np.zeros((1, 1024), dtype=np.float32)
                )
            ),
        ),
        patch("podcodex.cli.QdrantStore", return_value=MagicMock()),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)  # should not raise


def test_cmd_vectorize_episode_falls_back_to_meta(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path, meta={"show": "S", "episode": "meta_ep"})
    args = _make_vectorize_args(p, show="s", episode=None)

    captured = {}

    def fake_semantic_chunks(t, **kwargs):
        captured["episode"] = t["meta"].get("episode")
        return [{"text": "x"}]

    mock_local = _mock_local_not_indexed()
    with (
        patch("podcodex.cli.semantic_chunks", side_effect=fake_semantic_chunks),
        patch(
            "podcodex.rag.embedder.get_embedder",
            return_value=MagicMock(
                encode_passages=MagicMock(
                    return_value=np.zeros((1, 1024), dtype=np.float32)
                )
            ),
        ),
        patch("podcodex.cli.QdrantStore", return_value=MagicMock()),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    assert captured["episode"] == "meta_ep"


def test_cmd_vectorize_episode_falls_back_to_filename(tmp_path):
    """When no --episode and no meta.episode, stem of the filename is used."""
    from podcodex.cli import cmd_vectorize

    data = {
        "meta": {},
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "A",
                "text": "Long enough sentence for testing purposes.",
            },
        ],
    }
    p = tmp_path / "stem_fallback.json"
    p.write_text(json.dumps(data))
    args = _make_vectorize_args(p, show="s", episode=None)

    mock_store = MagicMock()
    mock_local = _mock_local_not_indexed()
    with (
        patch("podcodex.cli.semantic_chunks", return_value=[{"text": "t"}]),
        patch(
            "podcodex.rag.embedder.get_embedder",
            return_value=MagicMock(
                encode_passages=MagicMock(
                    return_value=np.zeros((1, 1024), dtype=np.float32)
                )
            ),
        ),
        patch("podcodex.cli.QdrantStore", return_value=mock_store),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_store.create_collection.assert_called_once()


def test_cmd_vectorize_no_chunks_returns_early(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="s")

    mock_store = MagicMock()
    mock_local = _mock_local_not_indexed()
    with (
        patch("podcodex.cli.semantic_chunks", return_value=[]),
        patch("podcodex.cli.QdrantStore", return_value=mock_store),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_store.upsert.assert_not_called()


def test_cmd_vectorize_file_not_found_exits(tmp_path):
    from podcodex.cli import cmd_vectorize

    args = _make_vectorize_args(tmp_path / "nonexistent.json", show="s")
    with pytest.raises(SystemExit):
        cmd_vectorize(args)


def test_cmd_vectorize_touches_rag_indexed_marker(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="my_show")

    mock_local = _mock_local_not_indexed()
    with (
        patch(
            "podcodex.rag.chunker.semantic_chunks", return_value=[{"text": "chunk1"}]
        ),
        patch(
            "podcodex.rag.embedder.get_embedder",
            return_value=MagicMock(
                encode_passages=MagicMock(
                    return_value=np.zeros((1, 1024), dtype=np.float32)
                )
            ),
        ),
        patch("podcodex.cli.QdrantStore", return_value=MagicMock()),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    marker = p.parent / ".rag_indexed"
    assert marker.exists(), ".rag_indexed marker should be touched after upsert"


def test_cmd_vectorize_overwrite_flag(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="s", overwrite=True)

    mock_store = MagicMock()
    mock_local = _mock_local_not_indexed()
    with (
        patch("podcodex.cli.semantic_chunks", return_value=[{"text": "x"}]),
        patch(
            "podcodex.rag.embedder.get_embedder",
            return_value=MagicMock(
                encode_passages=MagicMock(
                    return_value=np.zeros((1, 1024), dtype=np.float32)
                )
            ),
        ),
        patch("podcodex.cli.QdrantStore", return_value=mock_store),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_store.create_collection.assert_called_once_with(
        "s__bge-m3__semantic", model="bge-m3", overwrite=True
    )


def test_cmd_vectorize_skips_embed_when_cached(tmp_path):
    """When episode is already in LocalStore, embedding is skipped."""
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="my_show")

    cached_chunks = [
        {"text": "cached chunk", "embedding": np.zeros(1024, dtype=np.float32)}
    ]
    mock_local = MagicMock()
    mock_local.episode_is_indexed.return_value = True
    mock_local.load_chunks.return_value = cached_chunks

    mock_store = MagicMock()
    mock_embedder = MagicMock()

    with (
        patch("podcodex.cli.get_embedder", return_value=mock_embedder),
        patch("podcodex.cli.QdrantStore", return_value=mock_store),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_embedder.encode_passages.assert_not_called()
    mock_store.upsert.assert_called_once()


def test_cmd_vectorize_overwrite_deletes_local_episode(tmp_path):
    """With --overwrite, delete_episode is called before re-embedding."""
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="my_show", overwrite=True)

    mock_local = MagicMock()
    mock_local.episode_is_indexed.return_value = True

    with (
        patch("podcodex.cli.semantic_chunks", return_value=[{"text": "t"}]),
        patch(
            "podcodex.rag.embedder.get_embedder",
            return_value=MagicMock(
                encode_passages=MagicMock(
                    return_value=np.zeros((1, 1024), dtype=np.float32)
                )
            ),
        ),
        patch("podcodex.cli.QdrantStore", return_value=MagicMock()),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_local.delete_episode.assert_called_once()


# ──────────────────────────────────────────────
# cmd_sync
# ──────────────────────────────────────────────


def _make_sync_args(show=None, episode=None, overwrite=False, qdrant_url=None, db=None):
    args = MagicMock()
    args.show = show
    args.episode = episode
    args.overwrite = overwrite
    args.qdrant_url = qdrant_url
    args.db = db
    return args


def test_cmd_sync_pushes_all_episodes():
    from podcodex.cli import cmd_sync

    emb = np.zeros((2, 8), dtype=np.float32)
    cached = [
        {"text": "a", "embedding": emb[0]},
        {"text": "b", "embedding": emb[1]},
    ]

    mock_local = MagicMock()
    mock_local.list_collections.return_value = ["my_show__bge-m3__semantic"]
    mock_local._conn.execute.return_value.fetchone.return_value = (
        "My Show",
        "bge-m3",
        1024,
    )
    mock_local.list_episodes.return_value = ["ep1", "ep2"]
    mock_local.load_chunks.return_value = cached

    mock_store = MagicMock()
    args = _make_sync_args()

    with (
        patch("podcodex.cli.LocalStore", return_value=mock_local),
        patch("podcodex.cli.QdrantStore", return_value=mock_store),
    ):
        cmd_sync(args)

    assert mock_store.upsert.call_count == 2


def test_cmd_sync_filters_by_show():
    from podcodex.cli import cmd_sync

    mock_local = MagicMock()
    mock_local.list_collections.return_value = []
    args = _make_sync_args(show="My Show")

    with (
        patch("podcodex.cli.LocalStore", return_value=mock_local),
        patch("podcodex.cli.QdrantStore", return_value=MagicMock()),
    ):
        cmd_sync(args)

    mock_local.list_collections.assert_called_once_with(show="My Show")


def test_cmd_sync_episode_without_show_exits():
    from podcodex.cli import cmd_sync

    args = _make_sync_args(show=None, episode="ep1")
    with (
        patch("podcodex.cli.LocalStore", return_value=MagicMock()),
        patch("podcodex.cli.QdrantStore", return_value=MagicMock()),
        pytest.raises(SystemExit),
    ):
        cmd_sync(args)


def test_cmd_sync_no_collections_warns():
    from podcodex.cli import cmd_sync

    mock_local = MagicMock()
    mock_local.list_collections.return_value = []
    mock_store = MagicMock()
    args = _make_sync_args()

    with (
        patch("podcodex.cli.LocalStore", return_value=mock_local),
        patch("podcodex.cli.QdrantStore", return_value=mock_store),
    ):
        cmd_sync(args)

    mock_store.upsert.assert_not_called()


# ──────────────────────────────────────────────
# cmd_query
# ──────────────────────────────────────────────


def test_cmd_query_calls_retriever(capsys):
    from podcodex.cli import cmd_query

    args = MagicMock()
    args.query = "film music"
    args.show = "my_show"
    args.model = "bge-m3"
    args.chunking = "semantic"
    args.top_k = 3
    args.alpha = 0.5

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "score": 0.9,
            "speaker": "Alice",
            "start": 1.0,
            "end": 5.0,
            "text": "Hello world",
            "episode": "ep1",
        },
    ]

    with patch("podcodex.cli.Retriever", return_value=mock_retriever):
        cmd_query(args)

    mock_retriever.retrieve.assert_called_once_with(
        "film music", "my_show__bge-m3__semantic", top_k=3, alpha=0.5
    )
    out = capsys.readouterr().out
    assert "Hello world" in out
    assert "Alice" in out


def test_cmd_query_normalizes_show_name():
    from podcodex.cli import cmd_query

    args = MagicMock()
    args.query = "something"
    args.show = "My Podcast"
    args.model = "bge-m3"
    args.chunking = "semantic"
    args.top_k = 5
    args.alpha = 0.5

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch("podcodex.cli.Retriever", return_value=mock_retriever):
        cmd_query(args)

    mock_retriever.retrieve.assert_called_once_with(
        "something", "my_podcast__bge-m3__semantic", top_k=5, alpha=0.5
    )


def test_cmd_query_no_results(capsys):
    from podcodex.cli import cmd_query

    args = MagicMock()
    args.query = "q"
    args.show = "s"
    args.model = "bge-m3"
    args.chunking = "semantic"
    args.top_k = 5
    args.alpha = 0.5

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch("podcodex.cli.Retriever", return_value=mock_retriever):
        cmd_query(args)

    out = capsys.readouterr().out
    assert "No results" in out


# ──────────────────────────────────────────────
# cmd_list
# ──────────────────────────────────────────────


def test_cmd_list_no_filter(capsys):
    from podcodex.cli import cmd_list

    args = MagicMock()
    args.show = None

    mock_store = MagicMock()
    mock_store.list_collections.return_value = ["show_a", "show_b"]

    with patch("podcodex.cli.QdrantStore", return_value=mock_store):
        cmd_list(args)

    mock_store.list_collections.assert_called_once_with(show="")
    out = capsys.readouterr().out
    assert "show_a" in out
    assert "show_b" in out


def test_cmd_list_filtered_by_show(capsys):
    from podcodex.cli import cmd_list

    args = MagicMock()
    args.show = "my_show"

    mock_store = MagicMock()
    mock_store.list_collections.return_value = ["my_show"]

    with patch("podcodex.cli.QdrantStore", return_value=mock_store):
        cmd_list(args)

    mock_store.list_collections.assert_called_once_with(show="my_show")


def test_cmd_list_empty(capsys):
    from podcodex.cli import cmd_list

    args = MagicMock()
    args.show = None

    mock_store = MagicMock()
    mock_store.list_collections.return_value = []

    with patch("podcodex.cli.QdrantStore", return_value=mock_store):
        cmd_list(args)

    out = capsys.readouterr().out
    assert "No collections found" in out


# ──────────────────────────────────────────────
# cmd_delete
# ──────────────────────────────────────────────


def test_cmd_delete_calls_store(capsys):
    from podcodex.cli import cmd_delete

    args = MagicMock()
    args.collection = "my_show"

    mock_store = MagicMock()
    with patch("podcodex.cli.QdrantStore", return_value=mock_store):
        cmd_delete(args)

    mock_store.delete_collection.assert_called_once_with("my_show")
    out = capsys.readouterr().out
    assert "my_show" in out


# ──────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────


def test_parser_vectorize_required_args():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["vectorize", "ep.json", "--show", "myshow"])
    assert args.command == "vectorize"
    assert args.transcript == "ep.json"
    assert args.show == "myshow"
    assert args.model == "bge-m3"
    assert args.episode is None
    assert args.overwrite is False
    assert args.chunk_size == 256
    assert args.threshold == pytest.approx(0.5)


def test_parser_vectorize_optional_args():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "vectorize",
            "ep.json",
            "--show",
            "s",
            "--episode",
            "E42",
            "--overwrite",
            "--chunk-size",
            "128",
            "--threshold",
            "0.7",
            "--model",
            "e5-small",
        ]
    )
    assert args.episode == "E42"
    assert args.overwrite is True
    assert args.chunk_size == 128
    assert args.threshold == pytest.approx(0.7)
    assert args.model == "e5-small"


def test_parser_sync_defaults():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["sync"])
    assert args.command == "sync"
    assert args.show is None
    assert args.episode is None
    assert args.overwrite is False
    assert args.qdrant_url is None
    assert args.db is None


def test_parser_sync_with_show_and_episode():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["sync", "--show", "My Show", "--episode", "ep1"])
    assert args.show == "My Show"
    assert args.episode == "ep1"


def test_parser_query_required_args():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["query", "film music", "--show", "My Podcast"])
    assert args.command == "query"
    assert args.query == "film music"
    assert args.show == "My Podcast"
    assert args.top_k == 5  # default
    assert args.alpha == pytest.approx(0.5)  # default


def test_parser_query_top_k_and_alpha():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["query", "q", "--show", "s", "--top-k", "3", "--alpha", "0.8"]
    )
    assert args.top_k == 3
    assert args.alpha == pytest.approx(0.8)


def test_parser_list_no_show():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["list"])
    assert args.command == "list"
    assert args.show is None


def test_parser_list_with_show():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["list", "--show", "myshow"])
    assert args.show == "myshow"


def test_parser_delete():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["delete", "my_col"])
    assert args.command == "delete"
    assert args.collection == "my_col"
