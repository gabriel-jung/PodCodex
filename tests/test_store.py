"""Tests for podcodex.rag.store — Qdrant client is fully mocked."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_qdrant_mock():
    mock_mod = MagicMock()
    mock_client = MagicMock()
    mock_mod.QdrantClient.return_value = mock_client
    mock_client.collection_exists.return_value = False
    mock_client.get_collections.return_value.collections = []
    return mock_mod, mock_client


def _make_store(mock_mod):
    with patch.dict(
        "sys.modules",
        {"qdrant_client": mock_mod, "qdrant_client.models": mock_mod.models},
    ):
        from importlib import reload
        import podcodex.rag.store as store_mod

        reload(store_mod)
        store = store_mod.QdrantStore(url="http://localhost:6333")
    return store, store_mod


def _embeddings(n: int, dim: int = 1024) -> np.ndarray:
    return np.random.rand(n, dim).astype(np.float32)


def _chunks(n: int = 3, episode: str = "E1") -> list[dict]:
    return [
        {
            "episode": episode,
            "show": "S",
            "start": float(i),
            "end": float(i + 1),
            "speaker": "Alice",
            "text": f"chunk {i}",
        }
        for i in range(n)
    ]


# ──────────────────────────────────────────────
# collection_name
# ──────────────────────────────────────────────


def test_collection_name_basic():
    from podcodex.rag.store import collection_name

    assert collection_name("my_podcast", "bge-m3") == "my_podcast__bge-m3"


def test_collection_name_normalizes_spaces():
    from podcodex.rag.store import collection_name

    assert collection_name("My Podcast", "e5-small") == "my_podcast__e5-small"


def test_collection_name_normalizes_mixed_case():
    from podcodex.rag.store import collection_name

    assert collection_name("MyPodcast", "e5-large") == "mypodcast__e5-large"


def test_collection_name_normalizes_special_chars():
    from podcodex.rag.store import collection_name

    assert collection_name("My Podcast!", "bge-m3") == "my_podcast__bge-m3"


def test_collection_name_idempotent():
    from podcodex.rag.store import collection_name

    assert collection_name("my_podcast", "bge-m3") == collection_name(
        "My Podcast", "bge-m3"
    )


# ──────────────────────────────────────────────
# _serialize_payload
# ──────────────────────────────────────────────


def test_serialize_payload_converts_numpy_int():
    from podcodex.rag.store import _serialize_payload

    result = _serialize_payload({"x": np.int64(42)})
    assert result["x"] == 42
    assert isinstance(result["x"], int)


def test_serialize_payload_converts_numpy_float():
    from podcodex.rag.store import _serialize_payload

    result = _serialize_payload({"x": np.float32(3.14)})
    assert isinstance(result["x"], float)


def test_serialize_payload_converts_nested_list_of_dicts():
    from podcodex.rag.store import _serialize_payload

    result = _serialize_payload({"turns": [{"score": np.float32(0.9), "text": "hi"}]})
    assert isinstance(result["turns"][0]["score"], float)


def test_serialize_payload_passthrough_plain_types():
    from podcodex.rag.store import _serialize_payload

    payload = {"a": "hello", "b": 1, "c": 3.14, "d": ["x", "y"]}
    assert _serialize_payload(payload) == payload


# ──────────────────────────────────────────────
# create_collection
# ──────────────────────────────────────────────


def test_create_collection_calls_client():
    mock_mod, mock_client = _make_qdrant_mock()
    store, _ = _make_store(mock_mod)
    with patch.dict(
        "sys.modules",
        {"qdrant_client": mock_mod, "qdrant_client.models": mock_mod.models},
    ):
        store.create_collection("my_show__bge-m3", "bge-m3")
    mock_client.create_collection.assert_called_once()
    kwargs = mock_client.create_collection.call_args[1]
    assert kwargs["collection_name"] == "my_show__bge-m3"


def test_create_collection_unknown_model_raises():
    mock_mod, _ = _make_qdrant_mock()
    store, _ = _make_store(mock_mod)
    with pytest.raises(ValueError, match="Unknown model"):
        store.create_collection("col", "bad_model")


def test_create_collection_skips_if_exists():
    mock_mod, mock_client = _make_qdrant_mock()
    mock_client.collection_exists.return_value = True
    store, _ = _make_store(mock_mod)
    with patch.dict(
        "sys.modules",
        {"qdrant_client": mock_mod, "qdrant_client.models": mock_mod.models},
    ):
        store.create_collection("col", "e5-small", overwrite=False)
    mock_client.create_collection.assert_not_called()


def test_create_collection_overwrites_if_requested():
    mock_mod, mock_client = _make_qdrant_mock()
    mock_client.collection_exists.return_value = True
    store, _ = _make_store(mock_mod)
    with patch.dict(
        "sys.modules",
        {"qdrant_client": mock_mod, "qdrant_client.models": mock_mod.models},
    ):
        store.create_collection("col", "bge-m3", overwrite=True)
    mock_client.delete_collection.assert_called_once_with("col")
    mock_client.create_collection.assert_called_once()


# ──────────────────────────────────────────────
# list_collections / delete_collection
# ──────────────────────────────────────────────


def test_list_collections_returns_all():
    mock_mod, mock_client = _make_qdrant_mock()
    col_a, col_b = MagicMock(), MagicMock()
    col_a.name = "show_a__bge-m3"
    col_b.name = "show_b__e5-small"
    mock_client.get_collections.return_value.collections = [col_a, col_b]
    store, _ = _make_store(mock_mod)
    result = store.list_collections()
    assert "show_a__bge-m3" in result
    assert "show_b__e5-small" in result


def test_list_collections_filtered_by_show():
    mock_mod, mock_client = _make_qdrant_mock()
    col_a, col_b = MagicMock(), MagicMock()
    col_a.name = "show_a__bge-m3"
    col_b.name = "show_b__e5-small"
    mock_client.get_collections.return_value.collections = [col_a, col_b]
    store, _ = _make_store(mock_mod)
    result = store.list_collections(show="show_a")
    assert result == ["show_a__bge-m3"]


def test_list_collections_filtered_by_model():
    mock_mod, mock_client = _make_qdrant_mock()
    col_a, col_b = MagicMock(), MagicMock()
    col_a.name = "show_a__bge-m3"
    col_b.name = "show_b__bge-m3"
    col_c = MagicMock()
    col_c.name = "show_a__e5-small"
    mock_client.get_collections.return_value.collections = [col_a, col_b, col_c]
    store, _ = _make_store(mock_mod)
    result = store.list_collections(model="bge-m3")
    assert set(result) == {"show_a__bge-m3", "show_b__bge-m3"}


def test_list_collections_normalizes_show_name():
    mock_mod, mock_client = _make_qdrant_mock()
    col = MagicMock()
    col.name = "my_podcast__bge-m3"
    mock_client.get_collections.return_value.collections = [col]
    store, _ = _make_store(mock_mod)
    result = store.list_collections(show="My Podcast")
    assert result == ["my_podcast__bge-m3"]


def test_delete_collection_calls_client():
    mock_mod, mock_client = _make_qdrant_mock()
    store, _ = _make_store(mock_mod)
    store.delete_collection("my_col")
    mock_client.delete_collection.assert_called_once_with("my_col")


# ──────────────────────────────────────────────
# upsert
# ──────────────────────────────────────────────


def test_upsert_calls_client():
    mock_mod, mock_client = _make_qdrant_mock()
    store, _ = _make_store(mock_mod)
    with patch.dict(
        "sys.modules",
        {"qdrant_client": mock_mod, "qdrant_client.models": mock_mod.models},
    ):
        store.upsert("col", _chunks(3), _embeddings(3))
    mock_client.upsert.assert_called_once()
    _, kwargs = mock_client.upsert.call_args
    assert kwargs["collection_name"] == "col"
    assert len(kwargs["points"]) == 3


def test_upsert_batches_large_inputs():
    mock_mod, mock_client = _make_qdrant_mock()
    store, _ = _make_store(mock_mod)
    n = 10
    with patch.dict(
        "sys.modules",
        {"qdrant_client": mock_mod, "qdrant_client.models": mock_mod.models},
    ):
        store.upsert("col", _chunks(n), _embeddings(n), batch_size=3)
    assert mock_client.upsert.call_count == 4  # ceil(10/3) = 4


def test_upsert_points_have_unique_ids():
    mock_mod, mock_client = _make_qdrant_mock()
    store, _ = _make_store(mock_mod)
    with patch.dict(
        "sys.modules",
        {"qdrant_client": mock_mod, "qdrant_client.models": mock_mod.models},
    ):
        store.upsert("col", _chunks(5), _embeddings(5))
    calls = mock_mod.models.PointStruct.call_args_list
    ids = [c.kwargs["id"] for c in calls]
    assert len(set(ids)) == 5


def test_upsert_works_with_e5_small_dim():
    """E5-small produces 384-dim vectors."""
    mock_mod, mock_client = _make_qdrant_mock()
    store, _ = _make_store(mock_mod)
    with patch.dict(
        "sys.modules",
        {"qdrant_client": mock_mod, "qdrant_client.models": mock_mod.models},
    ):
        store.upsert("col", _chunks(2), _embeddings(2, dim=384))
    mock_client.upsert.assert_called_once()
