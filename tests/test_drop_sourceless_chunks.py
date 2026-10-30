"""Deleting the version an index was cut from takes the episode out of it.

Chunks of a deleted transcript kept answering searches, and the outdated flag
could never clear them: there was nothing left to re-index from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcodex.core.pipeline_db import close_pipeline_db
from podcodex.core.versions import delete_version, save_version
from podcodex.ingest.show import ShowMeta, save_show_meta, show_id
from podcodex.rag.index_store import get_index_store
from tests.fixtures.index import add_show, chunk

SEGS = [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hello there"}]


@pytest.fixture
def show(tmp_path: Path, isolated_index: Path):
    folder = tmp_path / "Show"
    folder.mkdir()
    save_show_meta(folder, ShowMeta(name="Show"))
    yield folder
    close_pipeline_db(folder)


def _version(show: Path, step: str = "transcript") -> str:
    return save_version(show / "ep" / "ep", step, SEGS, {"step": step, "type": "raw"})


def _index(
    show: Path, *, ref: str = "auto", version: str = "", step: str = "transcript"
):
    store = get_index_store()
    add_show(
        store,
        "Show",
        {
            "ep": [
                chunk(
                    "hello there",
                    episode="ep",
                    show="Show",
                    source=step,
                    source_hash="h",
                    source_ref=ref,
                    source_version=version,
                )
            ]
        },
        show_id=show_id(show),
    )
    return store


def _indexed(store) -> bool:
    return any(store.episode_is_indexed(c, "ep") for c in store.list_collections())


def test_deleting_the_last_transcript_unindexes_the_episode(show):
    first, second = _version(show), _version(show)
    store = _index(show)

    delete_version(show / "ep" / "ep", "transcript", first)
    assert _indexed(store)  # the canonical seglist still exists

    delete_version(show / "ep" / "ep", "transcript", second)
    assert not _indexed(store)


def test_deleting_a_pinned_version_unindexes_it_even_with_others_left(show):
    pinned, _other = _version(show), _version(show)
    store = _index(show, ref="pinned", version=pinned)

    delete_version(show / "ep" / "ep", "transcript", pinned)
    assert not _indexed(store)


def test_deleting_a_version_of_another_step_leaves_the_index(show):
    _version(show)
    corrected = _version(show, "corrected")
    store = _index(show, ref="step", step="transcript")

    delete_version(show / "ep" / "ep", "corrected", corrected)
    assert _indexed(store)


def test_unindexing_demotes_the_indexed_flag(show):
    from podcodex.core.pipeline_db import get_pipeline_db

    only = _version(show)
    _index(show)
    get_pipeline_db(show).mark("ep", indexed=True)

    delete_version(show / "ep" / "ep", "transcript", only)
    assert not get_pipeline_db(show).get_episode("ep")["indexed"]


def test_a_pin_whose_file_is_gone_counts_as_gone(show):
    """A delete whose row removal failed leaves the row: the file decides."""
    from podcodex.core.delete_episode import drop_sourceless_chunks
    from podcodex.core.versions import version_path

    pinned, _other = _version(show), _version(show)
    store = _index(show, ref="pinned", version=pinned)
    version_path(show / "ep" / "ep", "transcript", pinned).unlink()

    assert drop_sourceless_chunks(show / "ep" / "ep", "transcript", store=store)
    assert not _indexed(store)


def test_a_show_without_an_id_leaves_a_same_named_shows_chunks(
    tmp_path, isolated_index
):
    """Strict lookup: no claiming another show's collections by label."""
    from podcodex.core.delete_episode import drop_sourceless_chunks

    loose = tmp_path / "Loose"
    loose.mkdir()
    (loose / "show.toml").write_text('name = "Show"\n', encoding="utf-8")
    store = get_index_store()
    add_show(store, "Show", {"ep": [chunk("other show", episode="ep", show="Show")]})

    assert drop_sourceless_chunks(loose / "ep" / "ep", "transcript", store=store) == []
    assert _indexed(store)
    close_pipeline_db(loose)
