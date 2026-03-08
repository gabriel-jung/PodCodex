"""Tests for podcodex.cli — all heavy deps mocked."""

import json
from unittest.mock import MagicMock, patch

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
    episode=None,
    chunk_size=256,
    threshold=0.5,
    overwrite=False,
    source="transcript",
):
    args = MagicMock()
    args.transcript = str(transcript)
    args.show = show
    args.episode = episode
    args.chunk_size = chunk_size
    args.threshold = threshold
    args.overwrite = overwrite
    args.source = source
    return args


# ──────────────────────────────────────────────
# cmd_vectorize
# ──────────────────────────────────────────────


def test_cmd_vectorize_calls_store_methods(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="my_show")

    mock_chunks = [{"text": "chunk1", "start": 0.0, "end": 5.0}]
    mock_embeddings = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.encode_passages.return_value = mock_embeddings
    mock_store = MagicMock()

    with (
        patch("podcodex.rag.chunker.semantic_chunks", return_value=mock_chunks),
        patch("podcodex.rag.embedder.BGEEmbedder", return_value=mock_embedder),
        patch("podcodex.rag.store.QdrantStore", return_value=mock_store),
    ):
        cmd_vectorize(args)

    mock_store.create_collection.assert_called_once_with("my_show", overwrite=False)
    mock_store.upsert.assert_called_once_with("my_show", mock_chunks, mock_embeddings)


def test_cmd_vectorize_episode_from_args(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="s", episode="custom_ep")

    with (
        patch("podcodex.rag.chunker.semantic_chunks", return_value=[{"text": "t"}]),
        patch("podcodex.rag.embedder.BGEEmbedder", return_value=MagicMock()),
        patch("podcodex.rag.store.QdrantStore", return_value=MagicMock()),
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

    with (
        patch("podcodex.rag.chunker.semantic_chunks", side_effect=fake_semantic_chunks),
        patch("podcodex.rag.embedder.BGEEmbedder", return_value=MagicMock()),
        patch("podcodex.rag.store.QdrantStore", return_value=MagicMock()),
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
    with (
        patch("podcodex.rag.chunker.semantic_chunks", return_value=[{"text": "t"}]),
        patch("podcodex.rag.embedder.BGEEmbedder", return_value=MagicMock()),
        patch("podcodex.rag.store.QdrantStore", return_value=mock_store),
    ):
        cmd_vectorize(args)

    mock_store.create_collection.assert_called_once()


def test_cmd_vectorize_no_chunks_returns_early(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="s")

    mock_store = MagicMock()
    with (
        patch("podcodex.rag.chunker.semantic_chunks", return_value=[]),
        patch("podcodex.rag.store.QdrantStore", return_value=mock_store),
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

    with (
        patch(
            "podcodex.rag.chunker.semantic_chunks", return_value=[{"text": "chunk1"}]
        ),
        patch("podcodex.rag.embedder.BGEEmbedder", return_value=MagicMock()),
        patch("podcodex.rag.store.QdrantStore", return_value=MagicMock()),
    ):
        cmd_vectorize(args)

    marker = p.parent / ".rag_indexed"
    assert marker.exists(), ".rag_indexed marker should be touched after upsert"


def test_cmd_vectorize_overwrite_flag(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="s", overwrite=True)

    mock_store = MagicMock()
    with (
        patch("podcodex.rag.chunker.semantic_chunks", return_value=[{"text": "x"}]),
        patch("podcodex.rag.embedder.BGEEmbedder", return_value=MagicMock()),
        patch("podcodex.rag.store.QdrantStore", return_value=mock_store),
    ):
        cmd_vectorize(args)

    mock_store.create_collection.assert_called_once_with("s", overwrite=True)


# ──────────────────────────────────────────────
# cmd_query
# ──────────────────────────────────────────────


def test_cmd_query_calls_retriever(capsys):
    from podcodex.cli import cmd_query

    args = MagicMock()
    args.query = "film music"
    args.show = "my_show"
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

    with patch("podcodex.rag.retriever.Retriever", return_value=mock_retriever):
        cmd_query(args)

    mock_retriever.retrieve.assert_called_once_with(
        "film music", "my_show", top_k=3, alpha=0.5
    )
    out = capsys.readouterr().out
    assert "Hello world" in out
    assert "Alice" in out


def test_cmd_query_normalizes_show_name():
    from podcodex.cli import cmd_query

    args = MagicMock()
    args.query = "something"
    args.show = "My Podcast"
    args.top_k = 5
    args.alpha = 0.5

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch("podcodex.rag.retriever.Retriever", return_value=mock_retriever):
        cmd_query(args)

    mock_retriever.retrieve.assert_called_once_with(
        "something", "my_podcast", top_k=5, alpha=0.5
    )


def test_cmd_query_no_results(capsys):
    from podcodex.cli import cmd_query

    args = MagicMock()
    args.query = "q"
    args.show = "s"
    args.top_k = 5
    args.alpha = 0.5

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch("podcodex.rag.retriever.Retriever", return_value=mock_retriever):
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

    with patch("podcodex.rag.store.QdrantStore", return_value=mock_store):
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

    with patch("podcodex.rag.store.QdrantStore", return_value=mock_store):
        cmd_list(args)

    mock_store.list_collections.assert_called_once_with(show="my_show")


def test_cmd_list_empty(capsys):
    from podcodex.cli import cmd_list

    args = MagicMock()
    args.show = None

    mock_store = MagicMock()
    mock_store.list_collections.return_value = []

    with patch("podcodex.rag.store.QdrantStore", return_value=mock_store):
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
    with patch("podcodex.rag.store.QdrantStore", return_value=mock_store):
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
        ]
    )
    assert args.episode == "E42"
    assert args.overwrite is True
    assert args.chunk_size == 128
    assert args.threshold == pytest.approx(0.7)


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
