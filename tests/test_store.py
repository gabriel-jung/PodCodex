"""Tests for podcodex.rag.store (collection_name helper)."""

from podcodex.rag.store import collection_name


def test_collection_name_basic():
    assert collection_name("my_podcast", "bge-m3") == "my_podcast__bge-m3__semantic"


def test_collection_name_with_chunker():
    assert (
        collection_name("my_podcast", "bge-m3", "speaker")
        == "my_podcast__bge-m3__speaker"
    )


def test_collection_name_normalizes_spaces():
    assert collection_name("My Podcast", "e5-small") == "my_podcast__e5-small__semantic"


def test_collection_name_normalizes_mixed_case():
    assert collection_name("MyPodcast", "e5-large") == "mypodcast__e5-large__semantic"


def test_collection_name_normalizes_special_chars():
    assert collection_name("My Podcast!", "bge-m3") == "my_podcast__bge-m3__semantic"


def test_collection_name_idempotent():
    assert collection_name("my_podcast", "bge-m3") == collection_name(
        "My Podcast", "bge-m3"
    )
