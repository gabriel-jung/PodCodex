"""Tests for podcodex.core.pipeline_db — per-show SQLite pipeline status."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from podcodex.core.pipeline_db import PipelineDB, get_pipeline_db, close_pipeline_db


@pytest.fixture
def db():
    """In-memory PipelineDB for fast tests."""
    d = PipelineDB(":memory:")
    yield d
    d.close()


# ── Basic CRUD ────────────────────────────────────────────


def test_empty_db(db):
    assert db.all_episodes() == []
    assert db.get_episode("nonexistent") is None
    assert db.episode_count() == 0


def test_mark_updates_row(db):
    db.mark("ep1", transcribed=True)
    db.mark("ep1", corrected=True)
    row = db.get_episode("ep1")
    assert row["transcribed"] is True
    assert row["corrected"] is True


def test_mark_invalid_column(db):
    with pytest.raises(ValueError, match="Unknown columns"):
        db.mark("ep1", bogus=True)


def test_mark_empty_is_noop(db):
    db.mark("ep1")
    assert db.get_episode("ep1") is None


# ── Translations ──────────────────────────────────────────


def test_mark_translations_overwrite(db):
    db.mark("ep1", translations=["english"])
    db.mark("ep1", translations=["english", "french", "german"])
    row = db.get_episode("ep1")
    assert row["translations"] == ["english", "french", "german"]


# ── Bulk populate ─────────────────────────────────────────


@dataclass
class FakeEpisode:
    stem: str
    audio_path: Path | None = None
    transcribed: bool = False
    corrected: bool = False
    indexed: bool = False
    synthesized: bool = False
    translations: list[str] = field(default_factory=list)


def test_populate_from_scan(db):
    episodes = [
        FakeEpisode(stem="ep1", transcribed=True, corrected=True, translations=["en"]),
        FakeEpisode(stem="ep2", audio_path=Path("/a/ep2.mp3"), indexed=True),
        FakeEpisode(stem="ep3"),
    ]
    db._populate_from_scan(episodes)
    assert db.episode_count() == 3

    ep1 = db.get_episode("ep1")
    assert ep1["transcribed"] is True
    assert ep1["corrected"] is True
    assert ep1["translations"] == ["en"]

    ep2 = db.get_episode("ep2")
    assert ep2["audio_path"] == "/a/ep2.mp3"
    assert ep2["indexed"] is True

    ep3 = db.get_episode("ep3")
    assert ep3["transcribed"] is False


def test_populate_keeps_existing_rows(db):
    """A folder scan is older news than a row a writer already created: it
    must not overwrite that row's flags or wipe its provenance to {}."""
    db.mark("ep1", transcribed=True, provenance={"transcript": {"model": "m"}})
    episodes = [
        FakeEpisode(stem="ep1", transcribed=False, corrected=True),
        FakeEpisode(stem="ep2", transcribed=True),
    ]
    db._populate_from_scan(episodes)

    row = db.get_episode("ep1")
    assert row["transcribed"] is True
    assert row["corrected"] is False
    assert row["provenance"] == {"transcript": {"model": "m"}}
    assert db.get_episode("ep2")["transcribed"] is True
    assert db.get_episode("ep2")["provenance"] == {}


# ── all_episodes ordering ────────────────────────────────


def test_all_episodes_sorted(db):
    db.mark("c", transcribed=True)
    db.mark("a", corrected=True)
    db.mark("b", indexed=True)
    stems = [row["stem"] for row in db.all_episodes()]
    assert stems == ["a", "b", "c"]


# ── Module-level cache ────────────────────────────────────


def test_one_show_shares_one_connection(tmp_path):
    """Thread safety rests on this: one PipelineDB, so one connection and one
    lock, per show folder (keyed on ``Path(show_folder)``, str or Path)."""
    db = get_pipeline_db(tmp_path)
    assert get_pipeline_db(tmp_path / ".") is db
    assert get_pipeline_db(str(tmp_path)) is db
    close_pipeline_db(tmp_path)


def test_close_pipeline_db(tmp_path):
    db = get_pipeline_db(tmp_path)
    db.mark("ep", transcribed=True)
    close_pipeline_db(tmp_path)
    # Re-open — data persists.
    db2 = get_pipeline_db(tmp_path)
    assert db2.get_episode("ep")["transcribed"] is True
    close_pipeline_db(tmp_path)


# ── Provenance ───────────────────────────────────────────


def test_provenance_merge_across_steps(db):
    """Each step key merges into the existing provenance dict."""
    db.mark("ep1", transcribed=True, provenance={"transcript": {"model": "large-v3"}})
    db.mark("ep1", corrected=True, provenance={"corrected": {"model": "qwen3:4b"}})
    db.mark("ep1", provenance={"english": {"model": "gpt-4o"}})
    row = db.get_episode("ep1")
    assert row["provenance"]["transcript"]["model"] == "large-v3"
    assert row["provenance"]["corrected"]["model"] == "qwen3:4b"
    assert row["provenance"]["english"]["model"] == "gpt-4o"


def test_provenance_overwrite_same_step(db):
    """Writing the same step key overwrites it."""
    db.mark("ep1", provenance={"transcript": {"model": "small"}})
    db.mark("ep1", provenance={"transcript": {"model": "large-v3"}})
    row = db.get_episode("ep1")
    assert row["provenance"]["transcript"]["model"] == "large-v3"


def test_provenance_merge_runs_in_an_immediate_transaction(db):
    """The read-modify-write must take the write lock before it reads.

    Provenance is one JSON blob merged key by key, and pipeline steps run in
    spawned subprocesses writing this same file while the API process writes
    manual saves. Without BEGIN IMMEDIATE both sides read the blob, each
    updates its own key and the second commit discards the first's, leaving a
    step whose status pills show no model.
    """
    seen: list[str] = []
    db._conn.set_trace_callback(seen.append)
    try:
        db.mark("ep1", provenance={"transcript": {"model": "large-v3"}})
    finally:
        db._conn.set_trace_callback(None)

    begin = next(i for i, sql in enumerate(seen) if "BEGIN IMMEDIATE" in sql)
    read = next(i for i, sql in enumerate(seen) if "SELECT provenance" in sql)
    assert begin < read
    assert db.get_episode("ep1")["provenance"]["transcript"]["model"] == "large-v3"


# ── Verified pointer ──────────────────────────────────────


def test_set_get_clear_verified(db):
    db.set_verified("ep1", "corrected", "v-123")
    ptr = db.get_verified("ep1")
    assert ptr == {"step": "corrected", "version_id": "v-123"}
    row = db.get_episode("ep1")
    assert row["verified"] == {"step": "corrected", "version_id": "v-123"}

    db.clear_verified("ep1")
    assert db.get_verified("ep1") is None


def test_set_verified_singleton_replaces(db):
    db.set_verified("ep1", "transcript", "v-1")
    db.set_verified("ep1", "corrected", "v-2")
    ptr = db.get_verified("ep1")
    assert ptr == {"step": "corrected", "version_id": "v-2"}


def test_verified_pointers_bulk(db):
    db.set_verified("ep1", "transcript", "v-1")
    db.set_verified("ep2", "corrected", "v-2")
    pointers = db.verified_pointers()
    assert pointers == {
        "ep1": {"step": "transcript", "version_id": "v-1"},
        "ep2": {"step": "corrected", "version_id": "v-2"},
    }


def test_version_ids_by_stem(db):
    db.insert_version(
        "ep1",
        "transcript",
        {
            "id": "v-1",
            "timestamp": "2026-05-28T10:00:00Z",
            "type": "raw",
            "content_hash": "sha256:abc",
            "segment_count": 5,
        },
    )
    db.insert_version(
        "ep1",
        "transcript",
        {
            "id": "v-2",
            "timestamp": "2026-05-28T11:00:00Z",
            "type": "raw",
            "content_hash": "sha256:def",
            "segment_count": 6,
        },
    )
    db.insert_version(
        "ep2",
        "transcript",
        {
            "id": "v-3",
            "timestamp": "2026-05-28T12:00:00Z",
            "type": "raw",
            "content_hash": "sha256:ghi",
            "segment_count": 7,
        },
    )
    ids = db.version_ids_by_stem("transcript")
    assert ids == {"ep1": {"v-1", "v-2"}, "ep2": {"v-3"}}


# ── aggregate_status ──────────────────────────────────────


def test_aggregate_status_counts_stages_and_edits(db):
    """Every show card's counts come from here; a raise is swallowed by the
    route into a warning, so a broken count would go unnoticed."""
    edited = {"type": "validated", "manual_edit": True}
    raw = {"type": "raw"}
    db.mark("a", transcribed=True, corrected=True, provenance={"transcript": edited})
    db.mark(
        "b",
        transcribed=True,
        translations=["french", "german"],
        provenance={"transcript": raw, "german": edited},
    )
    db.mark("c", indexed=True, synthesized=True, translations=["french"])
    db.mark("d", corrected=True, provenance={"corrected": raw})

    assert db.aggregate_status() == {
        "total": 4,
        "transcribed": 2,
        "transcribed_edited": 1,
        "corrected": 2,
        "corrected_edited": 0,
        "translated": 2,
        "translated_edited": 1,
        "synthesized": 1,
        "indexed": 1,
    }


def test_mark_bulk_writes_many_rows_at_once(db):
    db.mark("a", transcribed=True)
    db.mark_bulk({"a": {"transcribed": False}, "b": {"translations": ["french"]}})
    assert db.get_episode("a")["transcribed"] is False
    assert db.get_episode("b")["translations"] == ["french"]
    with pytest.raises(ValueError):
        db.mark_bulk({"a": {"provenance": "{}"}})


# ── Migrations ────────────────────────────────────────────


@pytest.mark.legacy("pipeline-db-schema")
def test_an_old_schema_db_is_migrated_with_its_flags(tmp_path):
    """Every other test opens a fresh schema, so this is where a migration
    runs under test. This is a pre-provenance DB whose correction column is
    still `polished` and whose versions table predates verified pointers and
    tombstones."""
    import sqlite3

    path = tmp_path / "pipeline.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE episodes (
            stem TEXT PRIMARY KEY, audio_path TEXT,
            transcribed INTEGER DEFAULT 0, polished INTEGER DEFAULT 0,
            indexed INTEGER DEFAULT 0, synthesized INTEGER DEFAULT 0,
            translations TEXT DEFAULT '[]', updated_at REAL
        );
        CREATE TABLE versions (
            id TEXT NOT NULL, stem TEXT NOT NULL, step TEXT NOT NULL,
            timestamp TEXT NOT NULL, type TEXT NOT NULL, model TEXT,
            params TEXT DEFAULT '{}', manual_edit INTEGER DEFAULT 0,
            content_hash TEXT NOT NULL, segment_count INTEGER NOT NULL,
            input_hash TEXT, PRIMARY KEY (id, stem, step)
        );
        INSERT INTO episodes (stem, transcribed, polished, translations)
            VALUES ('ep1', 1, 1, '["french"]');
        INSERT INTO versions VALUES
            ('v1', 'ep1', 'transcript', '2026-01-01', 'raw', 'm', '{}', 0, 'h', 3, NULL);
        """
    )
    conn.commit()
    conn.close()

    db = PipelineDB(path)
    try:
        row = db.get_episode("ep1")
        assert row["transcribed"] is True
        assert row["corrected"] is True
        assert row["translations"] == ["french"]
        assert row["provenance"] == {}
        assert row["verified"] is None
        (version,) = db.list_versions("ep1", "transcript")
        assert version["missing_since"] is None
    finally:
        db.close()


def test_a_version_id_is_looked_up_within_its_episode_and_step(db):
    """Ids are timestamps: legacy second-precision ids repeat across the
    steps of one run and across episodes."""

    def meta(step):
        return {
            "id": "20260101T000000Z_raw",
            "timestamp": "2026-01-01",
            "type": "raw",
            "model": step,
            "content_hash": "h",
            "segment_count": 1,
        }

    db.insert_version("ep1", "segments", meta("segments"))
    db.insert_version("ep1", "transcript", meta("transcript"))
    db.insert_version("ep2", "transcript", meta("other-episode"))

    vid = "20260101T000000Z_raw"
    assert db.get_version(vid, stem="ep1", step="transcript")["model"] == "transcript"
    assert db.get_version(vid, stem="ep2")["model"] == "other-episode"
