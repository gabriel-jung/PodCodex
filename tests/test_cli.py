"""Tests for podcodex.cli — all heavy deps mocked."""

import json
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────
# _load_transcript
# ──────────────────────────────────────────────


def test_load_transcript_dict_format(tmp_path):
    from podcodex.cli import _load_transcript

    data = {
        "meta": {"show": "S", "episode": "E1"},
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}],
    }
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(data))

    result = _load_transcript(p)
    assert result["meta"]["episode"] == "E1"
    assert len(result["segments"]) == 1


def test_load_transcript_list_format(tmp_path):
    from podcodex.cli import _load_transcript

    data = [{"start": 0.0, "end": 1.0, "text": "Hi"}]
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(data))

    result = _load_transcript(p)
    assert result["meta"] == {}
    assert result["segments"] == data


# ──────────────────────────────────────────────
# _chunk dispatch
# ──────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["pplx_context", "bge_speaker"])
def test_chunk_speaker_strategies(strategy):
    from podcodex.cli import _chunk

    transcript = {"meta": {}, "segments": []}
    mock_fn = MagicMock(return_value=[])
    with patch("podcodex.rag.chunker.speaker_chunks", mock_fn):
        _chunk(transcript, strategy)
    mock_fn.assert_called_once_with(transcript)


@pytest.mark.parametrize("strategy", ["e5_semantic", "bge_semantic"])
def test_chunk_semantic_strategies(strategy):
    from podcodex.cli import _chunk

    transcript = {"meta": {}, "segments": []}
    mock_fn = MagicMock(return_value=[])
    with patch("podcodex.rag.chunker.semantic_chunks", mock_fn):
        _chunk(transcript, strategy)
    mock_fn.assert_called_once_with(transcript)


# ──────────────────────────────────────────────
# cmd_vectorize
# ──────────────────────────────────────────────


def _make_vectorize_args(transcript, show, strategy, episode=None, overwrite=False):
    args = MagicMock()
    args.transcript = str(transcript)
    args.show = show
    args.strategy = strategy
    args.episode = episode
    args.overwrite = overwrite
    return args


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


def test_cmd_vectorize_calls_store_methods(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="my_show", strategy="bge_speaker")

    mock_chunks = [{"text": "chunk1", "start": 0.0, "end": 5.0}]
    mock_embeddings = MagicMock()
    mock_store = MagicMock()

    with (
        patch("podcodex.cli._chunk", return_value=mock_chunks) as mock_chunk,
        patch("podcodex.cli._embed", return_value=mock_embeddings) as mock_embed,
        patch("podcodex.rag.store.QdrantStore", return_value=mock_store),
    ):
        cmd_vectorize(args)

    mock_chunk.assert_called_once()
    mock_embed.assert_called_once_with(mock_chunks, "bge_speaker")
    mock_store.create_collection.assert_called_once_with(
        "my_show__bge_speaker", "bge_speaker", overwrite=False
    )
    mock_store.upsert.assert_called_once_with(
        "my_show__bge_speaker", mock_chunks, mock_embeddings
    )


def test_cmd_vectorize_episode_from_args(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(
        p, show="s", strategy="e5_semantic", episode="custom_ep"
    )

    mock_chunks = [{"text": "t"}]
    with (
        patch("podcodex.cli._chunk", return_value=mock_chunks),
        patch("podcodex.cli._embed", return_value=MagicMock()),
        patch("podcodex.rag.store.QdrantStore", return_value=MagicMock()),
    ):
        cmd_vectorize(args)  # should not raise; episode arg used


def test_cmd_vectorize_episode_falls_back_to_meta(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path, meta={"show": "S", "episode": "meta_ep"})
    args = _make_vectorize_args(p, show="s", strategy="e5_semantic", episode=None)

    captured_transcript = {}

    def fake_chunk(t, strategy):
        captured_transcript.update(t)
        return [{"text": "x"}]

    with (
        patch("podcodex.cli._chunk", side_effect=fake_chunk),
        patch("podcodex.cli._embed", return_value=MagicMock()),
        patch("podcodex.rag.store.QdrantStore", return_value=MagicMock()),
    ):
        cmd_vectorize(args)

    assert captured_transcript["meta"]["episode"] == "meta_ep"


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
    args = _make_vectorize_args(p, show="s", strategy="bge_speaker", episode=None)

    mock_store = MagicMock()
    with (
        patch("podcodex.cli._chunk", return_value=[{"text": "t"}]),
        patch("podcodex.cli._embed", return_value=MagicMock()),
        patch("podcodex.rag.store.QdrantStore", return_value=mock_store),
    ):
        cmd_vectorize(args)

    mock_store.create_collection.assert_called_once()


def test_cmd_vectorize_no_chunks_returns_early(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="s", strategy="bge_speaker")

    mock_store = MagicMock()
    with (
        patch("podcodex.cli._chunk", return_value=[]),
        patch("podcodex.rag.store.QdrantStore", return_value=mock_store),
    ):
        cmd_vectorize(args)

    mock_store.upsert.assert_not_called()


def test_cmd_vectorize_file_not_found_exits(tmp_path):
    from podcodex.cli import cmd_vectorize

    args = _make_vectorize_args(
        tmp_path / "nonexistent.json", show="s", strategy="e5_semantic"
    )
    with pytest.raises(SystemExit):
        cmd_vectorize(args)


def test_cmd_vectorize_touches_rag_indexed_marker(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="my_show", strategy="bge_speaker")

    mock_store = MagicMock()
    with (
        patch("podcodex.cli._chunk", return_value=[{"text": "chunk1"}]),
        patch("podcodex.cli._embed", return_value=MagicMock()),
        patch("podcodex.rag.store.QdrantStore", return_value=mock_store),
    ):
        cmd_vectorize(args)

    marker = p.parent / ".rag_indexed"
    assert marker.exists(), ".rag_indexed marker should be touched after upsert"


def test_cmd_vectorize_overwrite_flag(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="s", strategy="bge_semantic", overwrite=True)

    mock_store = MagicMock()
    with (
        patch("podcodex.cli._chunk", return_value=[{"text": "x"}]),
        patch("podcodex.cli._embed", return_value=MagicMock()),
        patch("podcodex.rag.store.QdrantStore", return_value=mock_store),
    ):
        cmd_vectorize(args)

    mock_store.create_collection.assert_called_once_with(
        "s__bge_semantic", "bge_semantic", overwrite=True
    )


# ──────────────────────────────────────────────
# cmd_query
# ──────────────────────────────────────────────


def test_cmd_query_calls_retriever(capsys):
    from podcodex.cli import cmd_query

    args = MagicMock()
    args.query = "film music"
    args.show = "my_show"
    args.strategy = "bge_speaker"
    args.top_k = 3

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "score": 0.9,
            "speaker": "Alice",
            "start": 1.0,
            "end": 5.0,
            "text": "Hello world",
        },
    ]

    with patch("podcodex.rag.retriever.Retriever", return_value=mock_retriever):
        cmd_query(args)

    mock_retriever.retrieve.assert_called_once_with(
        "film music", "my_show__bge_speaker", top_k=3
    )
    out = capsys.readouterr().out
    assert "Hello world" in out
    assert "Alice" in out


def test_cmd_query_normalizes_show_name():
    from podcodex.cli import cmd_query

    args = MagicMock()
    args.query = "something"
    args.show = "Total Trax"
    args.strategy = "bge_speaker"
    args.top_k = 5

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch("podcodex.rag.retriever.Retriever", return_value=mock_retriever):
        cmd_query(args)

    mock_retriever.retrieve.assert_called_once_with(
        "something", "total_trax__bge_speaker", top_k=5
    )


def test_cmd_query_no_results(capsys):
    from podcodex.cli import cmd_query

    args = MagicMock()
    args.query = "q"
    args.show = "s"
    args.strategy = "e5_semantic"
    args.top_k = 5

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
    mock_store.list_collections.return_value = [
        "show_a__bge_semantic",
        "show_b__e5_semantic",
    ]

    with patch("podcodex.rag.store.QdrantStore", return_value=mock_store):
        cmd_list(args)

    mock_store.list_collections.assert_called_once_with(show="")
    out = capsys.readouterr().out
    assert "show_a__bge_semantic" in out
    assert "show_b__e5_semantic" in out


def test_cmd_list_filtered_by_show(capsys):
    from podcodex.cli import cmd_list

    args = MagicMock()
    args.show = "my_show"

    mock_store = MagicMock()
    mock_store.list_collections.return_value = ["my_show__bge_semantic"]

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
    args.collection = "my_show__bge_semantic"

    mock_store = MagicMock()
    with patch("podcodex.rag.store.QdrantStore", return_value=mock_store):
        cmd_delete(args)

    mock_store.delete_collection.assert_called_once_with("my_show__bge_semantic")
    out = capsys.readouterr().out
    assert "my_show__bge_semantic" in out


# ──────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────


def test_parser_vectorize_required_args():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["vectorize", "ep.json", "--show", "myshow", "--strategy", "bge_speaker"]
    )
    assert args.command == "vectorize"
    assert args.transcript == "ep.json"
    assert args.show == "myshow"
    assert args.strategy == "bge_speaker"
    assert args.episode is None
    assert args.overwrite is False


def test_parser_vectorize_optional_args():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "vectorize",
            "ep.json",
            "--show",
            "s",
            "--strategy",
            "e5_semantic",
            "--episode",
            "E42",
            "--overwrite",
        ]
    )
    assert args.episode == "E42"
    assert args.overwrite is True


def test_parser_query_required_args():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["query", "film music", "--show", "Total Trax", "--strategy", "bge_speaker"]
    )
    assert args.command == "query"
    assert args.query == "film music"
    assert args.show == "Total Trax"
    assert args.strategy == "bge_speaker"
    assert args.top_k == 5  # default


def test_parser_query_top_k():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["query", "q", "--show", "s", "--strategy", "e5_semantic", "--top-k", "3"]
    )
    assert args.top_k == 3


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


def test_parser_invalid_strategy_exits():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vectorize", "ep.json", "--show", "s", "--strategy", "bad"])
