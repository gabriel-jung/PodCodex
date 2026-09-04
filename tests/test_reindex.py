"""`podcodex-reindex`: a user-facing CLI whose first act is destructive.

`_reindex_show` drops the show's collections and only then scans for
episodes, so a folder that resolves wrong or holds nothing logs its warning
after the data is already gone. Dropping is what the user asked for — the
tables are derived state — but the dry run must not touch anything, and the
audio-less episodes a subtitle-driven show is made of must be rebuilt rather
than skipped, which is what used to empty such a show on every rebuild.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from podcodex.rag import index_store as rag_index_store  # noqa: E402
from podcodex.rag import reindex as reindex_mod  # noqa: E402

DIM = 8
COLLECTION = "show_1111aaaa__bge-m3__semantic"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("PODCODEX_INDEX", str(tmp_path / "index"))
    rag_index_store.get_index_store.cache_clear()
    st = rag_index_store.get_index_store()
    st.ensure_collection(
        COLLECTION, show="Show", model="bge-m3", chunker="semantic", dim=DIM
    )
    st.set_collection_identity(COLLECTION, show_id="show_1111aaaa", show="Show")
    st.save_chunks(
        COLLECTION,
        "old-ep",
        [
            {
                "text": "stale",
                "episode": "old-ep",
                "show": "Show",
                "source": "transcript",
                "dominant_speaker": "sp",
                "start": 0.0,
                "end": 1.0,
            }
        ],
        np.zeros((1, DIM), dtype=np.float32),
    )
    yield st
    rag_index_store.get_index_store.cache_clear()


@pytest.fixture
def wiring(monkeypatch):
    """Resolve the show id, and record what gets vectorized."""
    monkeypatch.setattr(
        reindex_mod, "show_id_for_label", lambda _n: "show_1111aaaa", raising=False
    )
    import podcodex.ingest.show_registry as registry

    monkeypatch.setattr(registry, "show_id_for_label", lambda _n: "show_1111aaaa")

    seen: list[dict] = []
    import podcodex.rag.indexing as indexing

    def _vectorize(transcript, *_a, **_kw):
        seen.append(transcript)
        return len(transcript.get("segments") or [])

    monkeypatch.setattr(indexing, "vectorize_batch", _vectorize)
    return seen


class _Episode:
    def __init__(self, stem, output_dir, audio_path=None):
        self.stem = stem
        self.output_dir = output_dir
        self.audio_path = audio_path


def _seed_scan(monkeypatch, episodes):
    monkeypatch.setattr(reindex_mod, "scan_folder", lambda _f: episodes)


def _seed_transcript(monkeypatch, calls):
    import podcodex.core.source as helpers

    def _build(audio_path, show, stem, output_dir=None):
        calls.append({"audio_path": audio_path, "output_dir": output_dir, "stem": stem})
        return {"segments": [{"text": "hi", "start": 0.0, "end": 1.0, "speaker": "sp"}]}

    monkeypatch.setattr(helpers, "build_index_transcript", _build)


def test_dry_run_drops_nothing(store, wiring, monkeypatch, tmp_path):
    calls: list[dict] = []
    _seed_scan(monkeypatch, [_Episode("ep1", tmp_path / "ep1")])
    _seed_transcript(monkeypatch, calls)

    reindex_mod._reindex_show(tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=True)

    assert COLLECTION in store.list_collections()
    assert store.count_rows(COLLECTION) == 1  # the pre-existing chunk survives
    assert wiring == []  # nothing was written


def test_an_empty_folder_still_drops_the_collection(
    store, wiring, monkeypatch, tmp_path
):
    """The documented behaviour, pinned so the ordering is a decision rather
    than an accident: the drop happens before the scan, so a folder that
    resolves to nothing leaves the show unindexed."""
    _seed_scan(monkeypatch, [])

    reindex_mod._reindex_show(tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False)

    assert COLLECTION not in store.list_collections()


def test_an_audio_less_episode_is_indexed_through_its_output_dir(
    store, wiring, monkeypatch, tmp_path
):
    """Subtitle imports and flat YouTube extraction have no audio file;
    skipping them emptied a subtitle-driven show on every rebuild while the
    command still reported success."""
    calls: list[dict] = []
    _seed_scan(monkeypatch, [_Episode("subs-ep", tmp_path / "subs-ep")])
    _seed_transcript(monkeypatch, calls)

    reindex_mod._reindex_show(tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False)

    assert calls == [
        {
            "audio_path": None,
            "output_dir": str(tmp_path / "subs-ep"),
            "stem": "subs-ep",
        }
    ]
    assert len(wiring) == 1


def test_an_episode_with_audio_is_indexed_through_its_audio_path(
    store, wiring, monkeypatch, tmp_path
):
    calls: list[dict] = []
    audio = tmp_path / "ep1.mp3"
    _seed_scan(monkeypatch, [_Episode("ep1", tmp_path / "ep1", audio_path=audio)])
    _seed_transcript(monkeypatch, calls)

    reindex_mod._reindex_show(tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False)

    assert calls[0]["audio_path"] == str(audio)
    assert calls[0]["output_dir"] is None


def test_an_unreadable_episode_is_skipped_not_fatal(
    store, wiring, monkeypatch, tmp_path
):
    import podcodex.core.source as helpers

    def _build(audio_path, _show, stem, output_dir=None):
        if stem == "bad":
            raise ValueError("no transcript")
        return {"segments": [{"text": "hi", "start": 0.0, "end": 1.0, "speaker": "sp"}]}

    monkeypatch.setattr(helpers, "build_index_transcript", _build)
    _seed_scan(
        monkeypatch,
        [
            _Episode("bad", tmp_path / "bad"),
            _Episode("good", tmp_path / "good"),
        ],
    )

    reindex_mod._reindex_show(tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False)

    assert len(wiring) == 1


def test_cli_rejects_an_unknown_model_and_chunker():
    """A typo used to reach the indexing loop as a bare KeyError, and an
    unknown chunker built a collection nothing else can resolve."""
    import subprocess
    import sys

    for flag, value in (("--model", "not-a-model"), ("--chunker", "not-a-chunker")):
        proc = subprocess.run(
            [sys.executable, "-c", _CLI_SNIPPET, "Show", flag, value],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2, proc.stderr
        assert "invalid choice" in proc.stderr


_CLI_SNIPPET = (
    "import sys; sys.argv[0] = 'podcodex-reindex';"
    "from podcodex.rag.reindex import main; main()"
)


def test_dry_run_reports_without_writing(store, wiring, monkeypatch, tmp_path, caplog):
    calls: list[dict] = []
    _seed_scan(monkeypatch, [_Episode("ep1", Path(tmp_path) / "ep1")])
    _seed_transcript(monkeypatch, calls)

    reindex_mod._reindex_show(tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=True)

    assert calls  # the transcript was still resolved, so the count is real
    assert wiring == []
