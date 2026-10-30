"""`rag.index_job`: the only code that marks an episode indexed.

The embedder, chunker and index are stubbed at the module seams the job
imports from; what is under test is the sequence: a failed combination, a
cancel or an empty transcript must never reach ``mark_step``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from podcodex.rag import index_job
from podcodex.rag.indexing import IndexingError


@pytest.fixture
def seams(monkeypatch, tmp_path):
    """Stub every seam; returns the recorder and the episode's audio path."""
    import podcodex.core.pipeline_db as pipeline_db
    import podcodex.core.provenance as provenance
    import podcodex.core.source as source
    import podcodex.ingest.show_registry as registry
    import podcodex.rag.index_store as index_store
    import podcodex.rag.indexing as indexing

    rec = SimpleNamespace(
        marked=[], vectorized=[], transcripts=[], compacted=0, chunks=3, raise_=None
    )
    segments = [{"text": "hi", "start": 0.0, "end": 1.0, "speaker": "A"}]

    monkeypatch.setattr(
        source,
        "load_source",
        lambda *_a, **_k: source.SourceVersion(segments, "corrected", "v1"),
    )
    import podcodex.core.versions as versions

    monkeypatch.setattr(
        versions, "version_content_hash", lambda _b, _s, vid: f"sha256:{vid}"
    )
    monkeypatch.setattr(
        versions, "resolve_canonical_ref", lambda _b: ("corrected", "v1")
    )
    monkeypatch.setattr(
        source,
        "build_index_transcript",
        lambda *_a, segments=None, **_k: {
            "segments": segments,
            "meta": {"source": "corrected"},
        },
    )
    monkeypatch.setattr(registry, "show_id_for_folder", lambda *_a, **_k: "s_1")
    monkeypatch.setattr(provenance, "build_provenance", lambda *_a, **k: k)

    class _Store:
        def compact(self):
            rec.compacted += 1

    monkeypatch.setattr(index_store, "get_index_store", lambda: _Store())

    def _vectorize(*_a, on_progress=None, **kw):
        rec.vectorized.append(kw)
        rec.transcripts.append(_a[0])
        if on_progress:
            on_progress(0, 1, "BGE-M3 / semantic")
        if rec.raise_ is not None:
            raise rec.raise_
        return rec.chunks

    monkeypatch.setattr(indexing, "vectorize_batch", _vectorize)
    monkeypatch.setattr(
        pipeline_db, "mark_step", lambda *a, **k: rec.marked.append((a, k))
    )
    audio = tmp_path / "Show" / "ep1.mp3"
    audio.parent.mkdir()
    return rec, str(audio)


def _run(audio, cancelled=lambda: False):
    return index_job.run(
        progress_cb=lambda *_a: None,
        cancelled=cancelled,
        audio_path=audio,
        output_dir=None,
        show="Show",
        source="corrected",
        version_id=None,
        model_keys=["bge-m3"],
        chunkings=["semantic"],
        chunk_size=256,
        threshold=0.5,
        overwrite=False,
    )


def _batch(audio, cancelled=lambda: False):
    return index_job.run_for_batch(
        progress_cb=lambda *_a: None,
        cancelled=cancelled,
        audio_path=audio,
        stem="ep1",
        show_name="Show",
        model_keys=["bge-m3"],
        chunkings=["semantic"],
        force=True,
    )


def test_run_marks_the_episode_indexed_with_its_provenance(seams):
    rec, audio = seams
    out = _run(audio)

    assert out == {"chunks_upserted": 3, "source": "corrected", "stem": "ep1"}
    assert rec.compacted == 1
    ((_args, kwargs),) = rec.marked
    assert kwargs["indexed"] is True
    params = kwargs["provenance"]["indexed"]["params"]
    assert params["chunk_size"] == 256 and params["source"] == "corrected"


def test_batch_records_the_same_provenance_as_the_panel(seams):
    rec, audio = seams
    assert _batch(audio) == {"upserted": 3, "indexed": True, "skipped": False}
    params = rec.marked[0][1]["provenance"]["indexed"]["params"]
    assert {"source", "chunk_size", "threshold", "overwrite"} <= set(params)
    assert rec.vectorized[0]["overwrite"] is True


@pytest.mark.parametrize("entry", [_run, _batch])
def test_a_failed_combination_never_marks_indexed(seams, entry):
    rec, audio = seams
    rec.raise_ = IndexingError("BGE-M3 / semantic: boom")
    with pytest.raises(IndexingError):
        entry(audio)
    assert rec.marked == []


@pytest.mark.parametrize("entry", [_run, _batch])
def test_a_cancel_stops_before_marking_and_still_compacts(seams, entry):
    """What was written before the cancel is reclaimed by this subprocess,
    the only store that knows about it."""
    rec, audio = seams
    from podcodex.api.tasks import TaskCancelled

    with pytest.raises(TaskCancelled):
        entry(audio, cancelled=lambda: True)
    assert rec.marked == []
    assert rec.compacted == 1


def test_run_refuses_an_episode_that_chunks_to_nothing(seams):
    rec, audio = seams
    rec.chunks = 0
    with pytest.raises(ValueError, match="0 chunks"):
        _run(audio)
    assert rec.marked == []


def test_batch_skips_an_episode_with_no_transcript(seams, monkeypatch):
    import podcodex.core.source as source

    rec, audio = seams

    from podcodex.core.source import NoTranscriptError

    def _missing(*_a, **_k):
        raise NoTranscriptError("no transcript")

    monkeypatch.setattr(source, "load_source", _missing)
    assert _batch(audio) == {"upserted": 0, "indexed": False, "skipped": False}
    assert rec.vectorized == [] and rec.marked == []


def test_batch_fails_an_episode_whose_transcript_is_unreadable(seams, monkeypatch):
    """Corrupt is not absent: reporting it as "nothing to index" hid it."""
    import json

    import podcodex.core.source as source

    rec, audio = seams

    def _corrupt(*_a, **_k):
        raise json.JSONDecodeError("bad", "{", 0)

    monkeypatch.setattr(source, "load_source", _corrupt)
    with pytest.raises(ValueError):
        _batch(audio)
    assert rec.marked == []


def test_the_index_records_the_source_versions_content_hash(seams):
    """The hash is the index's own record of what it holds: an edit of the
    source changes it, which is how an outdated index is detected."""
    rec, audio = seams
    _run(audio)  # the panel, with a step picked
    _batch(audio)  # a batch, canonical source
    assert [t["meta"]["source_hash"] for t in rec.transcripts] == ["sha256:v1"] * 2
    assert [t["meta"]["source_ref"] for t in rec.transcripts] == ["step", "auto"]
    assert [t["meta"]["source_version"] for t in rec.transcripts] == ["v1", "v1"]


def test_an_auto_load_that_fell_back_still_records_its_hash(seams, monkeypatch):
    """The canonical file was unreadable (a sync still copying it) and the
    load read another version: recording what was read is what flags the
    index outdated once the canonical file is readable again. Recording
    nothing made it read as current forever."""
    import podcodex.core.versions as versions

    rec, audio = seams
    monkeypatch.setattr(
        versions, "resolve_canonical_ref", lambda _b: ("corrected", "v2")
    )
    _batch(audio)
    assert rec.transcripts[0]["meta"]["source_hash"] == "sha256:v1"


@pytest.mark.parametrize(
    "source, version_id, ref",
    [
        ("auto", None, "auto"),
        ("", None, "auto"),
        ("french", None, "step"),
        # Picking the version that is canonical now follows the canonical
        # seglist: a pin would freeze the index on it through later edits.
        ("auto", "v1", "auto"),
        ("corrected", "v1", "auto"),
        ("auto", "v0", "pinned"),
        ("corrected", "v0", "pinned"),
        # The same id under another step (legacy ids repeat): not canonical.
        ("transcript", "v1", "pinned"),
    ],
)
def test_source_ref_names_how_the_source_was_chosen(
    tmp_path, monkeypatch, source, version_id, ref
):
    import podcodex.core.versions as versions

    monkeypatch.setattr(
        versions, "resolve_canonical_ref", lambda _b: ("corrected", "v1")
    )
    from podcodex.core.source import SourceVersion

    step = "transcript" if source == "transcript" else "corrected"
    consumed = SourceVersion([], step, version_id or "v1")
    assert index_job._source_ref(tmp_path, source, version_id, consumed) == ref


@pytest.mark.parametrize(
    "record, version_id, skipped",
    [
        ({"hash": "sha256:a", "ref": "auto", "step": "corrected"}, None, True),
        # Edited since: the canonical source moved on, re-index.
        ({"hash": "sha256:old", "ref": "auto", "step": "corrected"}, None, False),
        # A step or version the user picked is kept by an unpinned batch.
        ({"hash": "sha256:fr", "ref": "step", "step": "french"}, None, True),
        # The picked step was edited since: re-index (from that step).
        ({"hash": "sha256:fr-old", "ref": "step", "step": "french"}, None, False),
        ({"hash": "sha256:old", "ref": "pinned", "step": "transcript"}, None, True),
        # Chunks from before the record: current.
        (None, None, True),
        # A batch pin wants exactly that version.
        ({"hash": "sha256:a", "ref": "auto", "step": "corrected"}, "v-pin", False),
        ({"hash": "sha256:pin", "ref": "pinned", "step": "corrected"}, "v-pin", True),
        # Chunks from before the record may come from any version: a pick
        # re-indexes them.
        (None, "v-pin", False),
    ],
)
def test_the_batch_skip_check_follows_how_the_episode_was_indexed(
    tmp_path, monkeypatch, record, version_id, skipped
):
    import podcodex.core.versions as versions
    import podcodex.ingest.show_registry as registry
    import podcodex.rag.index_store as index_store

    class _Store:
        def resolve_collection(self, *_a, **_k):
            return "col"

        def episode_is_indexed(self, _col, _stem):
            return True

        def episode_source_record(self, _col, _stem):
            return record

    monkeypatch.setattr(index_store, "get_index_store", lambda: _Store())
    monkeypatch.setattr(registry, "show_id_for_folder", lambda *_a, **_k: "s_1")
    monkeypatch.setattr(
        versions,
        "current_source_hash",
        lambda _b, _s, ref: {"auto": "sha256:a", "step": "sha256:fr"}.get(ref),
    )
    monkeypatch.setattr(
        versions,
        "version_meta_by_id",
        lambda _b, vid: {"content_hash": "sha256:pin"} if vid == "v-pin" else None,
    )
    audio = tmp_path / "Show" / "ep1.mp3"
    assert (
        index_job.already_indexed(
            str(audio), "ep1", "Show", ["bge-m3"], ["semantic"], version_id=version_id
        )
        is skipped
    )


@pytest.mark.parametrize(
    "record, source",
    [
        (
            {"hash": "h", "ref": "step", "step": "french", "version": "v"},
            ("french", None, "step"),
        ),
        (
            {"hash": "h", "ref": "auto", "step": "corrected", "version": "v"},
            ("auto", None, "auto"),
        ),
        (
            {"hash": "h", "ref": "pinned", "step": "transcript", "version": "v-pin"},
            ("auto", "v-pin", "pinned"),
        ),
        # The pinned version was deleted since: canonical, not a failed run.
        (
            {"hash": "h", "ref": "pinned", "step": "transcript", "version": "v-gone"},
            ("auto", None, "auto"),
        ),
        (None, ("auto", None, "auto")),
    ],
)
def test_an_unpinned_batch_keeps_what_the_user_picked(
    tmp_path, monkeypatch, record, source
):
    """A missing collection or force makes the batch run; re-indexing from the
    canonical source then silently replaced a picked step or pinned version."""
    import podcodex.ingest.show_registry as registry
    import podcodex.rag.index_store as index_store

    class _Store:
        def resolve_collection(self, *_a, **_k):
            return "col"

        def episode_source_record(self, _col, _stem):
            return record

    import podcodex.core.versions as versions

    monkeypatch.setattr(index_store, "get_index_store", lambda: _Store())
    monkeypatch.setattr(registry, "show_id_for_folder", lambda *_a, **_k: "s_1")
    pinned = tmp_path / "pinned.json"
    pinned.write_text("[]")
    monkeypatch.setattr(
        versions,
        "version_meta_by_id",
        lambda _b, vid: {"step": "transcript"} if vid in ("v-pin", "v-gone") else None,
    )
    monkeypatch.setattr(
        versions,
        "version_path",
        lambda _b, _s, vid: pinned if vid == "v-pin" else tmp_path / "gone.json",
    )
    audio = str(tmp_path / "Show" / "ep1.mp3")
    assert (
        index_job.batch_index_source(audio, "ep1", "Show", ["bge-m3"], ["semantic"])
        == source
    )


def test_stale_index_combos_lists_only_the_stale_collections(tmp_path, monkeypatch):
    import podcodex.core.versions as versions
    import podcodex.ingest.show_registry as registry
    import podcodex.rag.index_store as index_store

    records = {
        "s__bge-m3__semantic": {"hash": "sha256:a", "ref": "auto", "step": "corrected"},
        "s__e5-small__semantic": {
            "hash": "sha256:old",
            "ref": "auto",
            "step": "corrected",
        },
        "s__e5-small__speaker": {
            "hash": "sha256:x",
            "ref": "pinned",
            "step": "transcript",
        },
    }

    class _Store:
        def get_all_collection_info(self):
            return {
                name: {"model": name.split("__")[1], "chunker": name.split("__")[2]}
                for name in records
            }

        def collections_for_show(self, *_a, **_k):
            return list(records)

        def episode_source_record(self, col, _stem):
            return records[col]

    monkeypatch.setattr(index_store, "get_index_store", lambda: _Store())
    monkeypatch.setattr(registry, "show_id_for_folder", lambda *_a, **_k: "s_1")
    monkeypatch.setattr(versions, "current_source_hash", lambda *_a: "sha256:a")
    audio = str(tmp_path / "Show" / "ep1.mp3")
    assert index_job.stale_index_combos(audio, "ep1", "Show") == [
        ("e5-small", "semantic")
    ]


def test_stale_combos_leave_out_models_this_install_does_not_know(
    tmp_path, monkeypatch
):
    import podcodex.ingest.show_registry as registry
    import podcodex.rag.index_store as index_store

    stale = {"hash": "sha256:old", "ref": "auto", "step": "corrected"}

    class _Store:
        def get_all_collection_info(self):
            return {
                "known": {"model": "bge-m3", "chunker": "semantic"},
                "gone": {"model": "retired-model", "chunker": "semantic"},
            }

        def collections_for_show(self, *_a, **_k):
            return ["known", "gone"]

        def episode_source_record(self, _col, _stem):
            return stale

    monkeypatch.setattr(index_store, "get_index_store", lambda: _Store())
    monkeypatch.setattr(registry, "show_id_for_folder", lambda *_a, **_k: "s_1")
    monkeypatch.setattr(index_job, "_record_is_stale", lambda _b, r: r is stale)
    audio = str(tmp_path / "Show" / "ep1.mp3")
    assert index_job.stale_index_combos(audio, "ep1", "Show") == [
        ("bge-m3", "semantic")
    ]
