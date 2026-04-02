"""Tests for podcodex.cli — all heavy deps mocked."""

import json
from unittest.mock import ANY, MagicMock, patch

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
    device="cpu",
    dry_run=False,
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
    args.device = device
    args.db = None
    args.dry_run = dry_run
    return args


def _mock_local_not_indexed():
    """LocalStore mock where episode is NOT already cached."""
    mock = MagicMock()
    mock.episode_is_indexed.return_value = False
    return mock


# ──────────────────────────────────────────────
# cmd_vectorize
# ──────────────────────────────────────────────


def test_cmd_vectorize_saves_to_local(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="my_show")

    mock_chunks = [{"text": "chunk1", "start": 0.0, "end": 5.0}]
    mock_embeddings = np.zeros((1, 1024), dtype=np.float32)
    mock_embedder = MagicMock()
    mock_embedder.encode_passages.return_value = mock_embeddings
    mock_local = _mock_local_not_indexed()

    with (
        patch("podcodex.cli.semantic_chunks", return_value=mock_chunks),
        patch("podcodex.cli.get_embedder", return_value=mock_embedder),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_local.save_chunks.assert_called_once_with(
        "my_show__bge-m3__semantic", ANY, mock_chunks, mock_embeddings
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
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_local.save_chunks.assert_called_once()


def test_cmd_vectorize_no_chunks_returns_early(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="s")

    mock_local = _mock_local_not_indexed()
    with (
        patch("podcodex.cli.semantic_chunks", return_value=[]),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_local.save_chunks.assert_not_called()


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
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    marker = p.parent / ".rag_indexed"
    assert marker.exists(), ".rag_indexed marker should be touched after indexing"


def test_cmd_vectorize_overwrite_flag(tmp_path):
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="s", overwrite=True)

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
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_local.save_chunks.assert_called_once()


def test_cmd_vectorize_skips_when_locally_cached(tmp_path):
    """When episode is already in LocalStore, skip embedding entirely."""
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="my_show")

    mock_local = MagicMock()
    mock_local.episode_is_indexed.return_value = True
    mock_local.episode_chunk_count.return_value = 1
    mock_local.load_chunks_no_embeddings.return_value = [
        {"text": "cached chunk", "source": "transcript"}
    ]
    mock_embedder = MagicMock()

    with (
        patch("podcodex.cli.get_embedder", return_value=mock_embedder),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_embedder.encode_passages.assert_not_called()


def test_cmd_vectorize_re_embeds_on_source_upgrade(tmp_path):
    """When source changed (e.g. transcript → polished), re-chunk and re-embed."""
    from podcodex.cli import cmd_vectorize

    p = _make_transcript_file(tmp_path)
    args = _make_vectorize_args(p, show="my_show")

    mock_local = MagicMock()
    mock_local.episode_is_indexed.side_effect = [True, False]
    mock_local.load_chunks_no_embeddings.return_value = [
        {"text": "cached chunk", "source": "polished"}  # different from auto-resolved
    ]
    mock_embedder = MagicMock()
    mock_embedder.encode_passages.return_value = np.zeros((1, 1024), dtype=np.float32)

    with (
        patch("podcodex.cli.semantic_chunks", return_value=[{"text": "new"}]),
        patch("podcodex.cli.get_embedder", return_value=mock_embedder),
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_local.delete_episode.assert_called_once()
    mock_embedder.encode_passages.assert_called_once()


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
        patch("podcodex.cli.LocalStore", return_value=mock_local),
    ):
        cmd_vectorize(args)

    mock_local.delete_episode.assert_called_once()


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
    args.source_filter = None

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
        "film music", "my_show__bge-m3__semantic", top_k=3, alpha=0.5, source=None
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
    args.source_filter = None

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch("podcodex.cli.Retriever", return_value=mock_retriever):
        cmd_query(args)

    mock_retriever.retrieve.assert_called_once_with(
        "something", "my_podcast__bge-m3__semantic", top_k=5, alpha=0.5, source=None
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
    args.source_filter = None

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
    args.db = None

    mock_local = MagicMock()
    mock_local.list_collections.return_value = ["show_a", "show_b"]

    with patch("podcodex.cli.LocalStore", return_value=mock_local):
        cmd_list(args)

    mock_local.list_collections.assert_called_once_with(show="")
    out = capsys.readouterr().out
    assert "show_a" in out
    assert "show_b" in out


def test_cmd_list_filtered_by_show(capsys):
    from podcodex.cli import cmd_list

    args = MagicMock()
    args.show = "my_show"
    args.db = None

    mock_local = MagicMock()
    mock_local.list_collections.return_value = ["my_show"]

    with patch("podcodex.cli.LocalStore", return_value=mock_local):
        cmd_list(args)

    mock_local.list_collections.assert_called_once_with(show="my_show")


def test_cmd_list_empty(capsys):
    from podcodex.cli import cmd_list

    args = MagicMock()
    args.show = None
    args.db = None

    mock_local = MagicMock()
    mock_local.list_collections.return_value = []

    with patch("podcodex.cli.LocalStore", return_value=mock_local):
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
    args.db = None

    mock_local = MagicMock()
    with patch("podcodex.cli.LocalStore", return_value=mock_local):
        cmd_delete(args)

    mock_local.delete_collection.assert_called_once_with("my_show")
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


def test_source_default_is_auto():
    from podcodex.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["vectorize", "ep.json", "--show", "s"])
    assert args.source == "auto"


# ──────────────────────────────────────────────
# _resolve_source auto mode
# ──────────────────────────────────────────────


def test_resolve_source_auto_picks_polished_validated(tmp_path):
    from podcodex.cli import _resolve_source

    ep_dir = tmp_path / "my_ep"
    ep_dir.mkdir()
    transcript = ep_dir / "my_ep.json"
    transcript.write_text("{}")
    polished = ep_dir / "my_ep.polished.json"
    polished.write_text("{}")
    polished_raw = ep_dir / "my_ep.polished.raw.json"
    polished_raw.write_text("{}")

    result = _resolve_source(transcript, "auto")
    assert result == polished


def test_resolve_source_auto_falls_back_to_polished_raw(tmp_path):
    from podcodex.cli import _resolve_source

    ep_dir = tmp_path / "my_ep"
    ep_dir.mkdir()
    transcript = ep_dir / "my_ep.json"
    transcript.write_text("{}")
    polished_raw = ep_dir / "my_ep.polished.raw.json"
    polished_raw.write_text("{}")

    result = _resolve_source(transcript, "auto")
    assert result == polished_raw


def test_resolve_source_auto_falls_back_to_transcript(tmp_path):
    from podcodex.cli import _resolve_source

    ep_dir = tmp_path / "my_ep"
    ep_dir.mkdir()
    transcript = ep_dir / "my_ep.json"
    transcript.write_text("{}")

    result = _resolve_source(transcript, "auto")
    assert result == transcript


# ──────────────────────────────────────────────
# _source_label
# ──────────────────────────────────────────────


def test_source_label_transcript():
    from podcodex.cli import _source_label
    from pathlib import Path

    t = Path("/shows/ep/ep.json")
    assert _source_label(t, t) == "transcript"


def test_source_label_polished():
    from podcodex.cli import _source_label
    from pathlib import Path

    t = Path("/shows/ep/ep.json")
    s = Path("/shows/ep/ep.polished.json")
    assert _source_label(s, t) == "polished"


def test_source_label_polished_raw():
    from podcodex.cli import _source_label
    from pathlib import Path

    t = Path("/shows/ep/ep.json")
    s = Path("/shows/ep/ep.polished.raw.json")
    assert _source_label(s, t) == "polished"


def test_source_label_translation():
    from podcodex.cli import _source_label
    from pathlib import Path

    t = Path("/shows/ep/ep.json")
    s = Path("/shows/ep/ep.english.json")
    assert _source_label(s, t) == "english"


def test_source_label_dotted_stem():
    from podcodex.cli import _source_label
    from pathlib import Path

    t = Path("/shows/ep.01/ep.01.json")
    s = Path("/shows/ep.01/ep.01.polished.json")
    assert _source_label(s, t) == "polished"


# ──────────────────────────────────────────────
# Incremental sync: vectorize_episode
# ──────────────────────────────────────────────


def test_vectorize_episode_skips_when_locally_cached(tmp_path):
    """When episode is already in LocalStore, skip embedding."""
    from podcodex.cli import vectorize_episode

    transcript = {
        "meta": {"show": "S", "episode": "E1", "source": "transcript"},
        "segments": [{"start": 0.0, "end": 5.0, "speaker": "A", "text": "hello world"}],
    }
    mock_local = MagicMock()
    mock_local.episode_is_indexed.return_value = True
    mock_local.episode_chunk_count.return_value = 2
    mock_local.load_chunks_no_embeddings.return_value = [
        {"text": "a", "source": "transcript"},
        {"text": "b", "source": "transcript"},
    ]

    _, n = vectorize_episode(transcript, "S", "E1", "bge-m3", "semantic", mock_local)

    assert n == 0


def test_vectorize_episode_upgrades_on_source_change(tmp_path):
    """When source changed, delete local and re-embed."""
    from podcodex.cli import vectorize_episode

    transcript = {
        "meta": {"show": "S", "episode": "E1", "source": "polished"},
        "segments": [{"start": 0.0, "end": 5.0, "speaker": "A", "text": "hello world"}],
    }
    mock_local = MagicMock()
    mock_local.episode_is_indexed.side_effect = [True, False]
    mock_local.load_chunks_no_embeddings.return_value = [
        {"text": "a", "source": "transcript"},  # old source
    ]

    mock_embedder = MagicMock()
    mock_embedder.encode_passages.return_value = np.zeros((1, 1024), dtype=np.float32)

    with patch("podcodex.cli.get_embedder", return_value=mock_embedder):
        _, n = vectorize_episode(
            transcript,
            "S",
            "E1",
            "bge-m3",
            "semantic",
            mock_local,
            chunks=[{"text": "t"}],
        )

    mock_local.delete_episode.assert_called_once()
    mock_embedder.encode_passages.assert_called_once()
    assert n == 1


def test_vectorize_overwrite_always_deletes(tmp_path):
    """--overwrite deletes local episode and re-embeds."""
    from podcodex.cli import vectorize_episode

    transcript = {
        "meta": {"show": "S", "episode": "E1"},
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "A",
                "text": "Hello world this is a valid segment that is long enough.",
            }
        ],
    }
    mock_local = MagicMock()
    mock_local.episode_is_indexed.return_value = True

    mock_embedder = MagicMock()
    mock_embedder.encode_passages.return_value = np.zeros((1, 1024), dtype=np.float32)

    with patch("podcodex.cli.get_embedder", return_value=mock_embedder):
        vectorize_episode(
            transcript,
            "S",
            "E1",
            "bge-m3",
            "semantic",
            mock_local,
            chunks=[{"text": "t"}],
            overwrite=True,
        )

    mock_local.delete_episode.assert_called_once()
    mock_local.save_chunks.assert_called_once()
