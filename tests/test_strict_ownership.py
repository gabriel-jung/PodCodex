"""A show folder with no id owns only index rows with no id either.

The bot knows no ids and matches rows by label; the app's folder-based
callers pass ``strict`` so a same-named show that has an id keeps its tables.
"""

from __future__ import annotations

import numpy as np
import pytest

from podcodex.rag import index_store as rag_index_store
from podcodex.rag.index_store import IndexStore

DIM = 8


def test_row_is_show_matches_by_id_and_by_label_when_no_id_is_known():
    with_id = {"show_id": "talk_1", "show": "Talk"}

    # The bot's reading: no id known, match by label.
    assert IndexStore.row_is_show(with_id, "", "Talk")
    # With an id, strict changes nothing.
    assert IndexStore.row_is_show(with_id, "talk_1", "Talk", strict=True)
    assert not IndexStore.row_is_show(with_id, "other_2", "Talk", strict=True)


@pytest.mark.legacy("show-id")
def test_row_is_show_strict_only_changes_the_empty_id_case():
    with_id = {"show_id": "talk_1", "show": "Talk"}
    legacy = {"show_id": "", "show": "Talk"}

    # The app's reading: this show has no id, so it cannot own a row with one.
    assert not IndexStore.row_is_show(with_id, "", "Talk", strict=True)
    assert IndexStore.row_is_show(legacy, "", "Talk", strict=True)


@pytest.mark.legacy("show-id")
def test_an_unminted_folder_is_not_already_indexed_by_a_same_named_show(
    tmp_path, monkeypatch
):
    """The batch skip check must not borrow the other show's table and skip
    an episode that was never indexed."""
    from podcodex.rag.index_job import already_indexed

    index_path = tmp_path / "index"
    store = IndexStore(index_path)
    store.ensure_collection(
        "talk__bge-m3__semantic",
        show="Talk",
        model="bge-m3",
        chunker="semantic",
        dim=DIM,
        show_id="talk_11111111",
    )
    chunk = {
        "text": "x",
        "episode": "ep1",
        "show": "Talk",
        "source": "transcript",
        "dominant_speaker": "A",
        "start": 0.0,
        "end": 1.0,
    }
    store.save_chunks(
        "talk__bge-m3__semantic",
        "ep1",
        [chunk],
        np.random.default_rng(0).random((1, DIM), dtype=np.float32),
    )
    monkeypatch.setenv("PODCODEX_INDEX", str(index_path))
    rag_index_store.get_index_store.cache_clear()

    folder = tmp_path / "talk"
    folder.mkdir()
    (folder / "show.toml").write_text('name = "Talk"\n', encoding="utf-8")
    (folder / "ep1.mp3").write_bytes(b"fake")
    try:
        assert not already_indexed(
            str(folder / "ep1.mp3"), "ep1", "Talk", ["bge-m3"], ["semantic"]
        )
    finally:
        rag_index_store.get_index_store.cache_clear()
