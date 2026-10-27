"""`podcodex-reindex`: a user-facing CLI that rewrites a show's index.

The rebuild is in place: episodes are re-embedded over their old rows and
only rows the scan did not rebuild are removed, after a run with no failure,
so a kill or a failed episode never leaves the show emptier than its
failures. The dry run must not touch anything, a same-named show's table is
never touched, and the audio-less episodes a subtitle-driven show is made of
must be rebuilt rather than skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from podcodex.rag import index_store as rag_index_store  # noqa: E402
from podcodex.rag import reindex as reindex_mod  # noqa: E402
from tests.fixtures.index import add_show, chunk  # noqa: E402

DIM = 8
COLLECTION = "show_1111aaaa__bge-m3__semantic"


@pytest.fixture
def store(isolated_index):
    st = rag_index_store.get_index_store()
    add_show(
        st,
        "Show",
        {"old-ep": [chunk("stale", episode="old-ep", show="Show")]},
        show_id="show_1111aaaa",
        collection=COLLECTION,
    )
    return st


@pytest.fixture
def wiring(monkeypatch):
    """Resolve the show id, and record what gets vectorized."""
    import podcodex.ingest.show_registry as registry

    monkeypatch.setattr(registry, "show_id_for_label", lambda _n: "show_1111aaaa")
    # The id comes from the reindexed folder itself, not from its name.
    monkeypatch.setattr(
        registry, "show_id_for_folder", lambda *_a, **_k: "show_1111aaaa"
    )

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


def test_an_empty_folder_leaves_the_index_as_it_was(
    store, wiring, monkeypatch, tmp_path
):
    """A folder that resolves to nothing is likelier a wrong path than a show
    whose every episode was deleted: nothing is pruned."""
    _seed_scan(monkeypatch, [])

    reindex_mod._reindex_show(tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False)

    assert store.list_episodes(COLLECTION) == ["old-ep"]


def test_episodes_the_scan_did_not_rebuild_are_pruned(
    store, wiring, monkeypatch, tmp_path
):
    _seed_scan(monkeypatch, [_Episode("ep1", tmp_path / "ep1")])
    _seed_transcript(monkeypatch, [])

    ok = reindex_mod._reindex_show(
        tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False
    )

    assert ok is True
    assert "old-ep" not in store.list_episodes(COLLECTION)


def test_a_failed_run_prunes_nothing(store, monkeypatch, tmp_path):
    """Resumable: what was indexed before the failure stays searchable."""
    import podcodex.ingest.show_registry as registry
    import podcodex.rag.indexing as indexing

    monkeypatch.setattr(
        registry, "show_id_for_folder", lambda *_a, **_k: "show_1111aaaa"
    )

    def _fail(*_a, **_kw):
        raise indexing.IndexingError("embedder failed to load")

    monkeypatch.setattr(indexing, "vectorize_batch", _fail)
    _seed_scan(monkeypatch, [_Episode("ep1", tmp_path / "ep1")])
    _seed_transcript(monkeypatch, [])

    reindex_mod._reindex_show(tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False)

    assert store.list_episodes(COLLECTION) == ["old-ep"]


def test_fresh_drops_the_collection_first(store, wiring, monkeypatch, tmp_path):
    _seed_scan(monkeypatch, [])

    reindex_mod._reindex_show(
        tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False, fresh=True
    )

    assert COLLECTION not in store.list_collections()


def test_a_same_named_show_is_never_touched(store, monkeypatch, tmp_path):
    """An id-less folder owns only id-less rows. Resolved loosely, the other
    "Show" (which has an id) was dropped or pruned in its place."""
    import podcodex.ingest.show_registry as registry

    monkeypatch.setattr(registry, "show_id_for_folder", lambda *_a, **_k: "")
    _seed_scan(monkeypatch, [_Episode("ep1", tmp_path / "ep1")])
    _seed_transcript(monkeypatch, [])
    import podcodex.rag.indexing as indexing

    monkeypatch.setattr(indexing, "vectorize_batch", lambda *_a, **_kw: 1)

    reindex_mod._reindex_show(
        tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False, fresh=True
    )
    reindex_mod._reindex_show(tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False)

    assert store.list_episodes(COLLECTION) == ["old-ep"]


def test_the_rebuild_embeds_on_the_resolved_device(store, monkeypatch, tmp_path):
    import podcodex.core.device as device
    import podcodex.ingest.show_registry as registry
    import podcodex.rag.indexing as indexing

    monkeypatch.setattr(
        registry, "show_id_for_folder", lambda *_a, **_k: "show_1111aaaa"
    )
    monkeypatch.setattr(device, "device_str", lambda: "cuda")
    devices: list[str] = []
    monkeypatch.setattr(
        indexing,
        "vectorize_batch",
        lambda *_a, device="cpu", **_kw: devices.append(device) or 1,
    )
    _seed_scan(monkeypatch, [_Episode("ep1", tmp_path / "ep1")])
    _seed_transcript(monkeypatch, [])

    reindex_mod._reindex_show(tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False)

    assert devices == ["cuda"]


def test_an_audio_less_episode_is_indexed_through_its_output_dir(
    store, wiring, monkeypatch, tmp_path
):
    """Subtitle imports and flat YouTube extraction have no audio file;
    skipping them would empty a subtitle-driven show on every rebuild while
    the command still reports success."""
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


def _seed_builder(monkeypatch, failures: dict):
    """build_index_transcript raising ``failures[stem]`` for listed stems."""
    import podcodex.core.source as helpers

    def _build(audio_path, _show, stem, output_dir=None):
        if stem in failures:
            raise failures[stem]
        return {"segments": [{"text": "hi", "start": 0.0, "end": 1.0, "speaker": "sp"}]}

    monkeypatch.setattr(helpers, "build_index_transcript", _build)


def test_an_untranscribed_episode_is_skipped_and_its_rows_pruned(
    store, wiring, monkeypatch, tmp_path
):
    from podcodex.core.source import NoTranscriptError

    _seed_builder(monkeypatch, {"old-ep": NoTranscriptError("no transcript")})
    _seed_scan(
        monkeypatch,
        [_Episode("old-ep", tmp_path / "old-ep"), _Episode("good", tmp_path / "good")],
    )

    ok = reindex_mod._reindex_show(
        tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False
    )

    assert ok is True and len(wiring) == 1
    assert "old-ep" not in store.list_episodes(COLLECTION)


def test_an_unreadable_transcript_fails_the_run_and_keeps_its_rows(
    store, wiring, monkeypatch, tmp_path
):
    """Unreadable is not absent: pruning would delete good chunks while the
    run reported success."""
    _seed_builder(monkeypatch, {"old-ep": OSError("corrupt version file")})
    _seed_scan(
        monkeypatch,
        [_Episode("old-ep", tmp_path / "old-ep"), _Episode("good", tmp_path / "good")],
    )

    ok = reindex_mod._reindex_show(
        tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False
    )

    assert ok is False and len(wiring) == 1
    assert "old-ep" in store.list_episodes(COLLECTION)


def test_cli_rejects_an_unknown_model_and_chunker():
    """Unknown names are refused up front: a model typo would reach the
    indexing loop as a bare KeyError, and an unknown chunker would build a
    collection nothing else can resolve."""
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


def test_a_failed_episode_fails_the_run(store, monkeypatch, tmp_path):
    """Reporting "Done" would hide an episode missing from the index."""
    import podcodex.ingest.show_registry as registry
    import podcodex.rag.indexing as indexing

    monkeypatch.setattr(registry, "show_id_for_label", lambda _n: "show_1111aaaa")
    # The id comes from the reindexed folder itself, not from its name.
    monkeypatch.setattr(
        registry, "show_id_for_folder", lambda *_a, **_k: "show_1111aaaa"
    )

    def _fail(*_a, **_kw):
        raise indexing.IndexingError("embedder failed to load")

    monkeypatch.setattr(indexing, "vectorize_batch", _fail)
    _seed_scan(monkeypatch, [_Episode("ep1", tmp_path / "ep1")])
    _seed_transcript(monkeypatch, [])

    ok = reindex_mod._reindex_show(
        tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=False
    )

    assert ok is False


def test_a_dry_run_never_creates_an_index(
    isolated_index, wiring, monkeypatch, tmp_path
):
    """Opening a store creates the index and stamps this machine as its
    owner; on a machine with no index yet, a dry run must leave none behind."""
    _seed_scan(monkeypatch, [_Episode("ep1", tmp_path / "ep1")])
    _seed_transcript(monkeypatch, [])

    ok = reindex_mod._reindex_show(
        tmp_path, "Show", ["bge-m3"], ["semantic"], dry_run=True
    )

    assert ok is True
    assert not isolated_index.exists() or not any(isolated_index.iterdir())
