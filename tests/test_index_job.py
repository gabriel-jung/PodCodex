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

    rec = SimpleNamespace(marked=[], vectorized=[], compacted=0, chunks=3, raise_=None)
    segments = [{"text": "hi", "start": 0.0, "end": 1.0, "speaker": "A"}]

    monkeypatch.setattr(
        source,
        "load_source",
        lambda *_a, **_k: SimpleNamespace(segments=segments, step="corrected"),
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
    with pytest.raises(RuntimeError, match="Cancelled"):
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
