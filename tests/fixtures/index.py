"""Seeded LanceDB indexes for tests (explicit import; no root conftest).

One builder instead of a ``_seed_store`` per module. Shows are created the
way the app creates them today: every collection carries its show's id.
A test about data from before ids existed (``@pytest.mark.legacy("show-id")``)
passes ``show_id=""`` explicitly, so the pre-migration shape is visible where
it is used and nowhere else.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Mapping, Sequence

import numpy as np

DIM = 8


def default_show_id(label: str) -> str:
    """The id ``add_show`` gives *label* when none is passed; write it into a
    test's ``show.toml`` so the folder and its collections agree."""
    return f"{_slug(label)}_{zlib.crc32(label.encode()):08x}"


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def chunk(
    text: str = "hello",
    *,
    episode: str = "ep1",
    show: str = "S",
    speaker: str = "Alice",
    start: float = 0.0,
    end: float | None = None,
    source: str = "transcript",
    **extra: object,
) -> dict:
    """One chunk in the shape the chunkers write."""
    return {
        "text": text,
        "episode": episode,
        "show": show,
        "source": source,
        "dominant_speaker": speaker,
        "start": start,
        "end": start + 5.0 if end is None else end,
        **extra,
    }


def add_show(
    store,
    label: str,
    episodes: Mapping[str, Sequence[str | dict]] | None = None,
    *,
    show_id: str | None = None,
    model: str = "bge-m3",
    chunker: str = "semantic",
    dim: int = DIM,
    password: str = "",
    seed: int = 0,
    collection: str = "",
) -> str:
    """Index a show and return its collection name.

    ``episodes`` maps a stem to its chunks, each a text (spoken by "Alice",
    5 s apart) or a full chunk dict; the default is one episode ``ep1`` with
    one chunk. ``show_id`` defaults to a slug of the label; pass ``""`` for a
    pre-migration collection. ``collection`` overrides the table name, for
    two shows sharing a label. Embeddings are seeded random vectors.
    """
    from podcodex.core.show_passwords import hash_show_password

    if show_id is None:
        show_id = default_show_id(label)
    col = collection or f"{_slug(label)}__{model}__{chunker}"
    store.ensure_collection(
        col, show=label, model=model, chunker=chunker, dim=dim, show_id=show_id
    )
    rng = np.random.default_rng(seed)
    for stem, items in (episodes or {"ep1": [f"{label} talk"]}).items():
        chunks = [
            item
            if isinstance(item, dict)
            else chunk(item, episode=stem, show=label, start=10.0 * i)
            for i, item in enumerate(items)
        ]
        vectors = rng.random((len(chunks), dim), dtype=np.float32)
        store.save_chunks(col, stem, chunks, vectors)
    if password:
        key = show_id or label
        store.set_show_password(key, hash_show_password(password), show_label=label)
    return col


def show_id_of(store, label: str) -> str:
    """The id ``add_show`` gave *label*, read back from the index."""
    return store.show_id_for_label(label)
