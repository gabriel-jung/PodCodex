"""Tests for podcodex.core.versions — generation versioning."""

import json
import pytest

from podcodex.core.versions import (
    compute_hash,
    delete_version,
    has_matching_version,
    has_version,
    list_versions,
    load_latest,
    load_version,
    resolve_verified_source,
    save_version,
    version_count,
    versions_dir,
)
from podcodex.core.pipeline_db import get_pipeline_db


@pytest.fixture
def episode_dir(tmp_path):
    """Create a show/episode structure and return the 'base' path.

    Layout: tmp_path/show/episode/episode  (base = episode dir / stem)
    The show dir is base.parent.parent, which is where pipeline.db lives.
    """
    show = tmp_path / "show"
    show.mkdir()
    ep = show / "my_episode"
    ep.mkdir()
    return ep / "my_episode"  # base = dir / stem


SAMPLE_SEGMENTS = [
    {"speaker": "Alice", "text": "Hello", "start": 0.0, "end": 1.0},
    {"speaker": "Bob", "text": "Hi there", "start": 1.0, "end": 2.5},
]

SAMPLE_PROVENANCE = {
    "step": "corrected",
    "type": "raw",
    "model": "gpt-4o",
    "params": {"llm_mode": "api"},
    "manual_edit": False,
}


def _prov(step="corrected", type_="raw", model=None, params=None, manual_edit=False):
    """Build a provenance dict for tests."""
    return {
        "step": step,
        "type": type_,
        "model": model,
        "params": params or {},
        "manual_edit": manual_edit,
    }


class TestComputeHash:
    def test_deterministic(self):
        h1 = compute_hash(SAMPLE_SEGMENTS)
        h2 = compute_hash(SAMPLE_SEGMENTS)
        assert h1 == h2

    def test_starts_with_sha256(self):
        h = compute_hash(SAMPLE_SEGMENTS)
        assert h.startswith("sha256:")

    def test_different_content_different_hash(self):
        other = [{"speaker": "Alice", "text": "Goodbye", "start": 0.0, "end": 1.0}]
        assert compute_hash(SAMPLE_SEGMENTS) != compute_hash(other)

    def test_key_order_irrelevant(self):
        seg1 = [{"a": 1, "b": 2}]
        seg2 = [{"b": 2, "a": 1}]
        assert compute_hash(seg1) == compute_hash(seg2)


class TestSaveVersion:
    def test_creates_version(self, episode_dir):
        vid = save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o"),
        )

        assert vid.endswith("_raw")

        # Segment file exists
        seg_path = versions_dir(episode_dir) / "corrected" / f"{vid}.json"
        assert seg_path.exists()
        segments = json.loads(seg_path.read_text())
        assert len(segments) == 2

    def test_multiple_versions(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(model="v1"))
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(type_="validated", manual_edit=True),
        )

        entries = list_versions(episode_dir, "corrected")
        assert len(entries) == 2
        # Newest first
        assert entries[0]["type"] == "validated"
        assert entries[1]["type"] == "raw"

    def test_params_stored(self, episode_dir):
        save_version(
            episode_dir,
            "transcript",
            SAMPLE_SEGMENTS,
            _prov(step="transcript", model="large-v3", params={"language": "fr"}),
        )
        entries = list_versions(episode_dir, "transcript")
        assert entries[0]["params"]["language"] == "fr"

    def test_none_provenance_is_noop(self, episode_dir):
        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, None)
        assert vid == ""
        assert version_count(episode_dir, "corrected") == 0

    def test_input_hash_stored(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            {**_prov(), "input_hash": "sha256:abcdef1234567890"},
        )
        entries = list_versions(episode_dir, "corrected")
        assert entries[0]["input_hash"] == "sha256:abcdef1234567890"


class TestListVersions:
    def test_empty_when_no_versions(self, episode_dir):
        assert list_versions(episode_dir, "corrected") == []

    def test_returns_entries(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov())
        entries = list_versions(episode_dir, "corrected")
        assert len(entries) == 1


class TestLoadVersion:
    def test_load_existing(self, episode_dir):
        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov())
        segments = load_version(episode_dir, "corrected", vid)
        assert len(segments) == 2
        assert segments[0]["text"] == "Hello"

    def test_load_missing_raises(self, episode_dir):
        with pytest.raises(FileNotFoundError):
            load_version(episode_dir, "corrected", "nonexistent")


class TestLoadLatest:
    def test_returns_none_when_empty(self, episode_dir):
        assert load_latest(episode_dir, "corrected") is None

    def test_returns_latest(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            [{"text": "old"}],
            _prov(model="v1"),
        )
        save_version(
            episode_dir,
            "corrected",
            [{"text": "new"}],
            _prov(model="v2"),
        )
        segments = load_latest(episode_dir, "corrected")
        assert segments == [{"text": "new"}]


class TestVersionCount:
    def test_zero_when_empty(self, episode_dir):
        assert version_count(episode_dir, "corrected") == 0

    def test_counts_correctly(self, episode_dir):
        for i in range(3):
            save_version(
                episode_dir,
                "corrected",
                SAMPLE_SEGMENTS,
                _prov(model=f"m{i}"),
            )
        assert version_count(episode_dir, "corrected") == 3


class TestDifferentSteps:
    def test_steps_isolated(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov())
        save_version(
            episode_dir,
            "english",
            SAMPLE_SEGMENTS,
            _prov(step="english"),
        )

        assert version_count(episode_dir, "corrected") == 1
        assert version_count(episode_dir, "english") == 1


class TestHasVersion:
    def test_false_when_empty(self, episode_dir):
        assert has_version(episode_dir, "corrected") is False

    def test_true_when_exists(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov())
        assert has_version(episode_dir, "corrected") is True


class TestHasMatchingVersion:
    def test_no_versions(self, episode_dir):
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "gpt-4o"}) is False
        )

    def test_matching_model(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"llm_mode": "api"}),
        )
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "gpt-4o"}) is True
        )

    def test_different_model(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"llm_mode": "api"}),
        )
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "claude"}) is False
        )

    def test_matching_params(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"llm_mode": "api", "llm_provider": "openai"}),
        )
        assert (
            has_matching_version(
                episode_dir,
                "corrected",
                {"model": "gpt-4o", "llm_mode": "api", "llm_provider": "openai"},
            )
            is True
        )

    def test_partial_param_mismatch(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"llm_mode": "api", "llm_provider": "openai"}),
        )
        # Different provider
        assert (
            has_matching_version(
                episode_dir,
                "corrected",
                {"model": "gpt-4o", "llm_mode": "api", "llm_provider": "anthropic"},
            )
            is False
        )

    def test_empty_params_matches_any(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov())
        assert has_matching_version(episode_dir, "corrected", {}) is True

    def test_multiple_versions_one_matches(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="old-model"),
        )
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="new-model"),
        )
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "old-model"})
            is True
        )
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "new-model"})
            is True
        )
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "other"}) is False
        )


# ──────────────────────────────────────────────
# Dual-write & DB→file fallback
# ──────────────────────────────────────────────


class TestSaveAndLoad:
    """Verify save_version writes both file and DB, and load_latest uses the DB."""

    def test_save_writes_both_file_and_db(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

        version_id = save_version(
            episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE
        )

        # File half
        version_file = versions_dir(episode_dir) / "corrected" / f"{version_id}.json"
        assert version_file.exists()
        assert json.loads(version_file.read_text()) == SAMPLE_SEGMENTS

        # DB half — pipeline.db lives at show level (base.parent.parent)
        show_dir = episode_dir.parent.parent
        assert (show_dir / "pipeline.db").exists()
        db = get_pipeline_db(show_dir)
        meta = db.get_latest_version(episode_dir.name, "corrected")
        assert meta is not None
        assert meta["id"] == version_id
        assert meta["content_hash"] == compute_hash(SAMPLE_SEGMENTS)
        close_pipeline_db(show_dir)

    def test_load_latest_returns_none_when_db_empty(self, episode_dir):
        """No versions saved → load_latest returns None (no filesystem fallback)."""
        assert load_latest(episode_dir, "corrected") is None

    def test_load_latest_round_trip(self, episode_dir):
        """Save then load returns the same segments."""
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        loaded = load_latest(episode_dir, "corrected")
        assert loaded == SAMPLE_SEGMENTS


class TestResolveVerifiedSource:
    """Verify resolve_verified_source helper + delete cleanup."""

    def test_returns_none_when_no_pointer(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        assert resolve_verified_source(episode_dir) is None

    def test_returns_pointer_when_set(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.set_verified(episode_dir.name, "corrected", vid)
        resolved = resolve_verified_source(episode_dir)
        assert resolved is not None
        step, version_id, path = resolved
        assert step == "corrected"
        assert version_id == vid
        assert path.exists()
        close_pipeline_db(show_dir)

    def test_returns_none_when_file_missing(self, episode_dir):
        """Stale pointer (file deleted out-of-band) resolves to None."""
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.set_verified(episode_dir.name, "corrected", vid)
        # Wipe file behind the pointer; DB still holds the reference.
        (versions_dir(episode_dir) / "corrected" / f"{vid}.json").unlink()
        assert resolve_verified_source(episode_dir) is None
        close_pipeline_db(show_dir)

    def test_returns_none_for_non_verifiable_step(self, episode_dir):
        """A pointer at e.g. 'synthesize' is not a valid verified source."""
        from podcodex.core.pipeline_db import close_pipeline_db

        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.set_verified(episode_dir.name, "english", "v-1")
        assert resolve_verified_source(episode_dir) is None
        close_pipeline_db(show_dir)

    def test_delete_clears_pointer_when_target_removed(self, episode_dir):
        """Deleting the verified version clears the pointer via refresh hook."""
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.set_verified(episode_dir.name, "corrected", vid)
        delete_version(episode_dir, "corrected", vid)
        assert db.get_verified(episode_dir.name) is None
        close_pipeline_db(show_dir)

    def test_delete_keeps_pointer_when_other_version_removed(self, episode_dir):
        """Deleting a non-verified version leaves the pointer intact."""
        from podcodex.core.pipeline_db import close_pipeline_db

        v1 = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        v2 = save_version(
            episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(model="other")
        )
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.set_verified(episode_dir.name, "corrected", v1)
        delete_version(episode_dir, "corrected", v2)
        ptr = db.get_verified(episode_dir.name)
        assert ptr == {"step": "corrected", "version_id": v1}
        close_pipeline_db(show_dir)


class TestBackfillFromDisk:
    """Rebuilding pipeline.db must make on-disk versions reachable again."""

    def test_restores_rows_for_orphaned_files(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db
        from podcodex.core.versions import backfill_versions_from_disk

        vid = save_version(
            episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(type_="validated")
        )
        show_dir = episode_dir.parent.parent
        close_pipeline_db(show_dir)
        (show_dir / "pipeline.db").unlink()

        assert backfill_versions_from_disk(show_dir) == 1
        rows = get_pipeline_db(show_dir).list_versions(episode_dir.name, "corrected")
        assert [r["id"] for r in rows] == [vid]
        # The type suffix survives, so an edited version still reads as edited
        # rather than silently demoting to a raw model output.
        assert rows[0]["type"] == "validated"
        assert rows[0]["segment_count"] == len(SAMPLE_SEGMENTS)
        assert load_version(episode_dir, "corrected", vid) == SAMPLE_SEGMENTS
        close_pipeline_db(show_dir)

    def test_speaker_map_survives_the_rebuild(self, episode_dir):
        """A hand-assigned speaker map must still load after a rebuild.

        Its input_hash pointed at the label source's sha256, which a rebuild
        cannot reproduce (parquet gets a stat hash), so without re-binding it
        every assigned name would silently vanish.
        """
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import (
            backfill_versions_from_disk,
            load_latest_speaker_map,
            save_speaker_map_version,
        )

        save_version(
            episode_dir,
            "diarized_segments",
            SAMPLE_SEGMENTS,
            _prov(step="diarized_segments"),
        )
        save_speaker_map_version(episode_dir, {"SPEAKER_00": "Alice"})
        show_dir = episode_dir.parent.parent
        assert load_latest_speaker_map(episode_dir) == {"SPEAKER_00": "Alice"}

        close_pipeline_db(show_dir)
        (show_dir / "pipeline.db").unlink()
        backfill_versions_from_disk(show_dir)
        assert load_latest_speaker_map(episode_dir) == {"SPEAKER_00": "Alice"}
        close_pipeline_db(show_dir)

    def test_hand_edited_version_stays_edited(self, episode_dir):
        """A manual edit must not be demoted to model output by a rebuild.

        `manual_edit` lives only in the DB, so the type suffix in the filename
        is the sole carrier across a rebuild. build_provenance keeps the two
        in step; /translate/save-manual once wrote manual_edit=True with type
        "raw", and that version came back reading as un-edited.
        """
        from podcodex.api.routes._helpers import build_provenance
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import backfill_versions_from_disk, is_edited

        prov = build_provenance(
            "french", params={"llm_mode": "manual"}, manual_edit=True
        )
        assert prov["type"] == "validated"
        save_version(episode_dir, "french", SAMPLE_SEGMENTS, prov)
        show_dir = episode_dir.parent.parent
        assert is_edited(list_versions(episode_dir, "french")[0])

        close_pipeline_db(show_dir)
        (show_dir / "pipeline.db").unlink()
        backfill_versions_from_disk(show_dir)
        assert is_edited(list_versions(episode_dir, "french")[0])
        close_pipeline_db(show_dir)

    def test_is_idempotent(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import backfill_versions_from_disk

        save_version(
            episode_dir, "transcript", SAMPLE_SEGMENTS, _prov(step="transcript")
        )
        show_dir = episode_dir.parent.parent
        # Rows already exist, so a second pass must not duplicate them.
        assert backfill_versions_from_disk(show_dir) == 0
        close_pipeline_db(show_dir)


class TestStaleVersionRows:
    """A row whose file is gone must not hide the readable versions behind it."""

    def test_provenance_and_segments_name_the_same_version(self, episode_dir):
        """The status surfaces must not describe a version load_latest walked past."""
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import (
            get_latest_provenance,
            load_latest,
            version_path,
        )

        save_version(
            episode_dir,
            "transcript",
            SAMPLE_SEGMENTS,
            _prov(step="transcript", model="kept-model"),
        )
        newest = save_version(
            episode_dir,
            "transcript",
            SAMPLE_SEGMENTS,
            _prov(step="transcript", model="lost-model"),
        )
        version_path(episode_dir, "transcript", newest).unlink()

        assert load_latest(episode_dir, "transcript") == SAMPLE_SEGMENTS
        prov = get_latest_provenance(episode_dir, "transcript")
        assert prov is not None
        assert prov["model"] == "kept-model"
        close_pipeline_db(episode_dir.parent.parent)

    def test_provenance_is_none_when_every_file_is_gone(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import (
            get_latest_provenance,
            load_latest,
            version_path,
        )

        only = save_version(
            episode_dir, "transcript", SAMPLE_SEGMENTS, _prov(step="transcript")
        )
        version_path(episode_dir, "transcript", only).unlink()

        assert load_latest(episode_dir, "transcript") is None
        assert get_latest_provenance(episode_dir, "transcript") is None
        close_pipeline_db(episode_dir.parent.parent)

    def test_canonical_ref_skips_a_row_without_a_file(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import (
            load_canonical_segments,
            resolve_canonical_ref,
            version_path,
        )

        old = save_version(
            episode_dir, "transcript", SAMPLE_SEGMENTS, _prov(step="transcript")
        )
        newest = save_version(
            episode_dir, "transcript", SAMPLE_SEGMENTS, _prov(step="transcript")
        )
        # Lost out of band (sync conflict, manual cleanup, a crash between the
        # unlink and the row delete): the row survives, the file does not.
        version_path(episode_dir, "transcript", newest).unlink()

        assert resolve_canonical_ref(episode_dir) == ("transcript", old)
        assert load_canonical_segments(episode_dir) == SAMPLE_SEGMENTS
        close_pipeline_db(episode_dir.parent.parent)

    def test_bulk_canonical_refs_skip_a_row_without_a_file(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import resolve_canonical_refs, version_path

        old = save_version(
            episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(step="corrected")
        )
        newest = save_version(
            episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(step="corrected")
        )
        version_path(episode_dir, "corrected", newest).unlink()

        show_dir = episode_dir.parent.parent
        refs = resolve_canonical_refs(show_dir, [episode_dir.name])
        assert refs[episode_dir.name] == ("corrected", old)
        close_pipeline_db(show_dir)

    def test_reconcile_prunes_rows_missing_past_the_grace_and_demotes(
        self, episode_dir, monkeypatch
    ):
        import time

        from podcodex.core import versions as versions_mod
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import (
            backfill_versions_from_disk,
            version_path,
        )

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.mark(episode_dir.name, corrected=True)
        version_path(episode_dir, "corrected", vid).unlink()

        # First miss: stamped, kept, flag untouched.
        backfill_versions_from_disk(show_dir)
        assert [v["id"] for v in list_versions(episode_dir, "corrected")] == [vid]
        assert db.get_episode(episode_dir.name)["corrected"] is True

        # Still missing once the grace has passed: pruned and demoted.
        later = time.time() + versions_mod.MISSING_ROW_GRACE_S + 1
        monkeypatch.setattr(time, "time", lambda: later)
        backfill_versions_from_disk(show_dir)

        assert list_versions(episode_dir, "corrected") == []
        assert db.get_episode(episode_dir.name)["corrected"] is False
        close_pipeline_db(show_dir)

    def test_a_late_file_keeps_its_row_and_provenance(self, episode_dir, monkeypatch):
        """A synced pipeline.db can arrive before the files it indexes; the
        first backfill used to delete those rows with their provenance."""
        import time

        from podcodex.core import versions as versions_mod
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import backfill_versions_from_disk, version_path

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        path = version_path(episode_dir, "corrected", vid)
        content = path.read_bytes()
        path.unlink()
        backfill_versions_from_disk(show_dir)

        path.write_bytes(content)  # the sync catches up
        backfill_versions_from_disk(show_dir)
        later = time.time() + versions_mod.MISSING_ROW_GRACE_S + 1
        monkeypatch.setattr(time, "time", lambda: later)
        backfill_versions_from_disk(show_dir)

        (row,) = list_versions(episode_dir, "corrected")
        assert row["model"] == SAMPLE_PROVENANCE["model"]
        assert row["missing_since"] is None
        close_pipeline_db(show_dir)

    def test_backfill_never_replaces_an_existing_row(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import backfill_versions_from_disk

        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent

        assert backfill_versions_from_disk(show_dir) == 0
        (row,) = list_versions(episode_dir, "corrected")
        assert row["model"] == SAMPLE_PROVENANCE["model"]
        close_pipeline_db(show_dir)

    def test_reconcile_keeps_rows_whose_files_are_present(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import backfill_versions_from_disk

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        backfill_versions_from_disk(show_dir)
        assert [v["id"] for v in list_versions(episode_dir, "corrected")] == [vid]
        close_pipeline_db(show_dir)


class TestStatusDemotionOnDelete:
    """Deleting the last version of a step demotes its pipeline_db flag."""

    def test_last_delete_demotes_flag(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.mark(episode_dir.name, corrected=True)
        delete_version(episode_dir, "corrected", vid)
        assert db.get_episode(episode_dir.name)["corrected"] is False
        close_pipeline_db(show_dir)

    def test_demote_skipped_when_a_version_survives(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db

        v1 = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(model="other"))
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.mark(episode_dir.name, corrected=True)
        delete_version(episode_dir, "corrected", v1)
        assert db.get_episode(episode_dir.name)["corrected"] is True
        close_pipeline_db(show_dir)

    def test_demote_is_atomic_against_a_racing_save(self, episode_dir):
        """A version registered mid-delete must win over the demotion.

        Pipeline steps run in spawned subprocesses writing to this same DB,
        so the emptiness check and the flag write have to be one transaction.
        Simulated here by registering a new version from inside the check.
        """
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.mark(episode_dir.name, corrected=True)

        # The racing save lands after the delete removes the only row, but
        # before the demotion commits.
        original = db.demote_step_if_no_versions

        def racing_demote(stem, step, flag):
            save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(model="race"))
            return original(stem, step, flag)

        db.demote_step_if_no_versions = racing_demote
        try:
            delete_version(episode_dir, "corrected", vid)
        finally:
            db.demote_step_if_no_versions = original

        assert list_versions(episode_dir, "corrected"), "racing save should survive"
        assert db.get_episode(episode_dir.name)["corrected"] is True
        close_pipeline_db(show_dir)

    def test_last_translation_delete_drops_the_language(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(
            episode_dir, "english", SAMPLE_SEGMENTS, _prov(step="english")
        )
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.mark(episode_dir.name, translations=["english", "french"])
        delete_version(episode_dir, "english", vid)
        assert db.get_episode(episode_dir.name)["translations"] == ["french"]
        close_pipeline_db(show_dir)


class TestSynthesizeVersions:
    """The synthesize step's four touchpoints: path, save, delete, demotion.

    Its version file is a ``.wav`` and its content hash is a stat hash, so it
    is the one step that does not go through ``save_version``. Per-step
    symmetry still has to hold: same ``version_path`` layout, same
    ``delete_version``, same flag demotion.
    """

    def _assemble(self, episode_dir, *, payload=b"RIFFfake-audio-bytes"):
        """Write a .wav where the route would, and register it."""
        from podcodex.core.source import SourceRef
        from podcodex.core.versions import (
            new_version_id,
            save_synthesize_version,
            version_path,
        )

        now, version_id = new_version_id()
        path = version_path(episode_dir, "synthesize", version_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        save_synthesize_version(
            episode_dir,
            path,
            version_id=version_id,
            now=now,
            strategy="pad",
            silence_duration=0.5,
            source=SourceRef("french", "20260101T000000000000Z_raw"),
            language="french",
            model_size="qwen-tts",
            segment_count=len(SAMPLE_SEGMENTS),
            duration_s=12.345,
        )
        return version_id, path

    def test_version_path_is_a_wav_beside_the_other_steps(self, episode_dir):
        from podcodex.core.versions import step_ext, version_path

        path = version_path(episode_dir, "synthesize", "20260101T000000000000Z_raw")
        assert step_ext("synthesize") == ".wav"
        assert path.suffix == ".wav"
        assert path.parent == episode_dir.parent / "synthesize"

    def test_save_round_trips_through_list_versions(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db

        payload = b"RIFF" + b"\0" * 40
        version_id, path = self._assemble(episode_dir, payload=payload)

        versions = list_versions(episode_dir, "synthesize")
        assert [v["id"] for v in versions] == [version_id]
        meta = versions[0]
        assert meta["step"] == "synthesize"
        assert meta["type"] == "raw"
        assert meta["manual_edit"] is False
        # Stat hash, not a sha256 of the audio.
        assert meta["content_hash"] == f"size:{len(payload)}"
        assert meta["segment_count"] == len(SAMPLE_SEGMENTS)
        assert meta["params"]["strategy"] == "pad"
        assert meta["params"]["language"] == "french"
        assert meta["params"]["duration_s"] == 12.35
        assert meta["params"]["file_size_bytes"] == len(payload)
        assert path.is_file()
        close_pipeline_db(episode_dir.parent.parent)

    def test_synthesize_version_path_resolves_only_a_present_file(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import synthesize_version_path

        version_id, path = self._assemble(episode_dir)
        assert synthesize_version_path(episode_dir, version_id) == path
        path.unlink()
        assert synthesize_version_path(episode_dir, version_id) is None
        close_pipeline_db(episode_dir.parent.parent)

    def test_delete_removes_the_wav_and_the_row(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db

        version_id, path = self._assemble(episode_dir)
        assert delete_version(episode_dir, "synthesize", version_id) is True
        assert not path.exists()
        assert list_versions(episode_dir, "synthesize") == []
        close_pipeline_db(episode_dir.parent.parent)

    def test_last_delete_demotes_the_synthesized_flag(self, episode_dir):
        """Otherwise the UI keeps showing a synthesized episode with no audio."""
        from podcodex.core.pipeline_db import close_pipeline_db

        version_id, _ = self._assemble(episode_dir)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.mark(episode_dir.name, synthesized=True)

        delete_version(episode_dir, "synthesize", version_id)

        assert db.get_episode(episode_dir.name)["synthesized"] is False
        close_pipeline_db(show_dir)

    def test_demote_skipped_while_another_wav_survives(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db

        first, _ = self._assemble(episode_dir)
        self._assemble(episode_dir, payload=b"RIFFother")
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.mark(episode_dir.name, synthesized=True)

        delete_version(episode_dir, "synthesize", first)

        assert db.get_episode(episode_dir.name)["synthesized"] is True
        close_pipeline_db(show_dir)

    def test_delete_by_id_resolves_the_step_from_the_db(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import delete_version_by_id

        version_id, path = self._assemble(episode_dir)
        assert delete_version_by_id(episode_dir, version_id) is True
        assert not path.exists()
        close_pipeline_db(episode_dir.parent.parent)


# ── Index integrity (save guard, file-backed matches, cascade) ─────────────


class TestIndexIntegrity:
    def test_a_traversal_step_is_refused_on_save(self, episode_dir):
        """Reads and deletes refused `../../x`; saves wrote outside the episode."""
        with pytest.raises(ValueError, match="Invalid version path"):
            save_version(
                episode_dir, "../../escape", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE
            )
        assert not (episode_dir.parent.parent.parent / "escape").exists()

    def test_a_failed_row_insert_takes_the_file_back(self, episode_dir, monkeypatch):
        from podcodex.core.pipeline_db import PipelineDB

        def boom(*_a, **_k):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(PipelineDB, "insert_version", boom)
        with pytest.raises(RuntimeError):
            save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        assert not list((episode_dir.parent / "corrected").glob("*.json"))

    def test_a_row_without_its_file_is_not_a_version(self, episode_dir):
        """Batch runs skip on these; a file-less row used to count as done."""
        from podcodex.core.versions import version_path

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        version_path(episode_dir, "corrected", vid).unlink()

        assert not has_version(episode_dir, "corrected")
        assert not has_matching_version(episode_dir, "corrected", {"model": "gpt-4o"})

    def test_current_only_ignores_an_older_match(self, episode_dir):
        from podcodex.core.versions import find_matching_version

        old = save_version(
            episode_dir, "segments", SAMPLE_SEGMENTS, _prov("segments", model="turbo")
        )
        save_version(
            episode_dir, "segments", SAMPLE_SEGMENTS, _prov("segments", model="medium")
        )

        assert find_matching_version(episode_dir, "segments", {"model": "turbo"}) == old
        assert (
            find_matching_version(
                episode_dir, "segments", {"model": "turbo"}, current_only=True
            )
            is None
        )

    def test_deleting_a_label_source_drops_its_speaker_maps(self, episode_dir):
        """A map references the IDs of the diarization it was made for; when
        that diarization goes, so must the map. A map bound to another
        diarization survives."""
        from podcodex.core.versions import (
            load_latest_speaker_map,
            save_speaker_map_version,
        )

        first = save_version(
            episode_dir,
            "diarized_segments",
            [{"speaker": "SPEAKER_00", "text": "a", "start": 0.0, "end": 1.0}],
            _prov("diarized_segments"),
        )
        save_speaker_map_version(episode_dir, {"SPEAKER_00": "Alice"})
        second = save_version(
            episode_dir,
            "diarized_segments",
            [{"speaker": "SPEAKER_00", "text": "b", "start": 0.0, "end": 1.0}],
            _prov("diarized_segments"),
        )
        save_speaker_map_version(episode_dir, {"SPEAKER_00": "Bob"})

        delete_version(episode_dir, "diarized_segments", first)

        maps = list_versions(episode_dir, "speaker_map")
        assert len(maps) == 1
        assert load_latest_speaker_map(episode_dir) == {"SPEAKER_00": "Bob"}
        assert second

    def test_bulk_and_single_canonical_refs_agree(self, episode_dir):
        """The speaker roster resolves every episode in bulk; the rules are
        one function now, and this pins that both entry points use it."""
        from podcodex.core.versions import resolve_canonical_ref, resolve_canonical_refs

        save_version(episode_dir, "transcript", SAMPLE_SEGMENTS, _prov("transcript"))
        edited = save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov("corrected", type_="validated", manual_edit=True),
        )
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov("corrected"))

        single = resolve_canonical_ref(episode_dir)
        bulk = resolve_canonical_refs(episode_dir.parent.parent, [episode_dir.name])
        assert single == bulk[episode_dir.name] == ("corrected", edited)


# ── Provenance ─────────────────────────────────────────────────────────────


class TestProvenance:
    def _audio(self, episode_dir):
        audio = episode_dir.parent.parent / f"{episode_dir.name}.mp3"
        audio.touch()
        return str(audio)

    def test_edit_provenance_inherits_the_parent_and_is_edited(self, episode_dir):
        from podcodex.core.provenance import build_edit_provenance

        audio = self._audio(episode_dir)
        params = {"llm_mode": "api", "source_chain": ["whisper", "gpt-4o"]}
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params=params),
        )

        prov = build_edit_provenance("corrected", audio, None)

        assert prov["type"] == "validated" and prov["manual_edit"] is True
        assert prov["model"] == "gpt-4o"
        assert prov["params"] == params
        prov["params"]["llm_mode"] = "changed"
        assert load_latest_provenance_params(episode_dir)["llm_mode"] == "api"

    def test_source_chain_follows_the_pinned_version(self, episode_dir):
        """The chain used to read the latest transcript even when the step
        ran on an older one the user picked."""
        from podcodex.core.provenance import build_provenance

        audio = self._audio(episode_dir)
        pinned = save_version(
            episode_dir,
            "transcript",
            SAMPLE_SEGMENTS,
            _prov("transcript", params={"source_chain": ["youtube-subtitles"]}),
        )
        save_version(
            episode_dir,
            "transcript",
            SAMPLE_SEGMENTS,
            _prov("transcript", params={"source_chain": ["whisper/large-v3"]}),
        )

        from podcodex.core.source import load_source

        latest_src = load_source(audio, None, step="transcript")
        pinned_src = load_source(audio, None, pinned)
        latest = build_provenance(
            "corrected", model="m", audio_path=audio, source=latest_src
        )
        pinned_prov = build_provenance(
            "corrected", model="m", audio_path=audio, source=pinned_src
        )

        assert latest["params"]["source_chain"] == ["whisper/large-v3", "m"]
        assert pinned_prov["params"]["source_chain"] == ["youtube-subtitles", "m"]
        # No consumed version known, no guessed chain.
        assert (
            "source_chain"
            not in build_provenance("corrected", model="m", audio_path=audio)["params"]
        )


def load_latest_provenance_params(base):
    from podcodex.core.versions import get_latest_provenance

    return get_latest_provenance(base, "corrected")["params"]
