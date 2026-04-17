"""Tests for podcodex.rag.indexing — all heavy deps mocked."""

from unittest.mock import MagicMock, patch

import numpy as np


# ──────────────────────────────────────────────
# vectorize_episode — incremental / upgrade / overwrite
# ──────────────────────────────────────────────


def test_vectorize_episode_skips_when_locally_cached(tmp_path):
    """When episode is already in LocalStore, skip embedding."""
    from podcodex.rag.indexing import vectorize_episode

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
    from podcodex.rag.indexing import vectorize_episode

    transcript = {
        "meta": {"show": "S", "episode": "E1", "source": "corrected"},
        "segments": [{"start": 0.0, "end": 5.0, "speaker": "A", "text": "hello world"}],
    }
    mock_local = MagicMock()
    mock_local.episode_is_indexed.side_effect = [True, False]
    mock_local.load_chunks_no_embeddings.return_value = [
        {"text": "a", "source": "transcript"},  # old source
    ]

    mock_embedder = MagicMock()
    mock_embedder.encode_passages.return_value = np.zeros((1, 1024), dtype=np.float32)

    with patch("podcodex.rag.indexing.get_embedder", return_value=mock_embedder):
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
    """overwrite=True deletes the local episode and re-embeds."""
    from podcodex.rag.indexing import vectorize_episode

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

    with patch("podcodex.rag.indexing.get_embedder", return_value=mock_embedder):
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
