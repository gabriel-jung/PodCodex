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


def test_ensure_episode(db):
    db.ensure_episode("ep1", audio_path="/audio/ep1.mp3")
    row = db.get_episode("ep1")
    assert row is not None
    assert row["stem"] == "ep1"
    assert row["audio_path"] == "/audio/ep1.mp3"
    assert row["transcribed"] is False
    assert row["translations"] == []
    assert db.episode_count() == 1

    # Idempotent — does not overwrite.
    db.ensure_episode("ep1", audio_path="/other.mp3")
    assert db.get_episode("ep1")["audio_path"] == "/audio/ep1.mp3"


def test_mark_creates_row(db):
    """mark() on a non-existent stem creates the row."""
    db.mark("ep1", transcribed=True)
    row = db.get_episode("ep1")
    assert row["transcribed"] is True
    assert row["polished"] is False


def test_mark_updates_row(db):
    db.mark("ep1", transcribed=True)
    db.mark("ep1", polished=True)
    row = db.get_episode("ep1")
    assert row["transcribed"] is True
    assert row["polished"] is True


def test_mark_invalid_column(db):
    with pytest.raises(ValueError, match="Unknown columns"):
        db.mark("ep1", bogus=True)


def test_mark_empty_is_noop(db):
    db.mark("ep1")
    assert db.get_episode("ep1") is None


# ── Translations ──────────────────────────────────────────


def test_mark_translations(db):
    db.mark("ep1", translations=["english", "french"])
    row = db.get_episode("ep1")
    assert row["translations"] == ["english", "french"]


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
    polished: bool = False
    indexed: bool = False
    synthesized: bool = False
    translations: list[str] = field(default_factory=list)


def test_populate_from_scan(db):
    episodes = [
        FakeEpisode(stem="ep1", transcribed=True, polished=True, translations=["en"]),
        FakeEpisode(stem="ep2", audio_path=Path("/a/ep2.mp3"), indexed=True),
        FakeEpisode(stem="ep3"),
    ]
    db.populate_from_scan(episodes)
    assert db.episode_count() == 3

    ep1 = db.get_episode("ep1")
    assert ep1["transcribed"] is True
    assert ep1["polished"] is True
    assert ep1["translations"] == ["en"]

    ep2 = db.get_episode("ep2")
    assert ep2["audio_path"] == "/a/ep2.mp3"
    assert ep2["indexed"] is True

    ep3 = db.get_episode("ep3")
    assert ep3["transcribed"] is False


def test_populate_upserts(db):
    """populate_from_scan updates existing rows."""
    db.mark("ep1", transcribed=True)
    episodes = [FakeEpisode(stem="ep1", transcribed=True, polished=True)]
    db.populate_from_scan(episodes)
    row = db.get_episode("ep1")
    assert row["polished"] is True


# ── all_episodes ordering ────────────────────────────────


def test_all_episodes_sorted(db):
    db.mark("c", transcribed=True)
    db.mark("a", polished=True)
    db.mark("b", indexed=True)
    stems = [row["stem"] for row in db.all_episodes()]
    assert stems == ["a", "b", "c"]


# ── Remove ────────────────────────────────────────────────


def test_remove_episode(db):
    db.mark("ep1", transcribed=True)
    db.remove_episode("ep1")
    assert db.get_episode("ep1") is None
    assert db.episode_count() == 0


def test_remove_nonexistent(db):
    db.remove_episode("nope")  # should not raise


# ── Module-level cache ────────────────────────────────────


def test_get_pipeline_db_caches(tmp_path):
    db1 = get_pipeline_db(tmp_path)
    db2 = get_pipeline_db(tmp_path)
    assert db1 is db2
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


def test_provenance_stored_and_read(db):
    prov = {
        "transcript": {
            "step": "transcript",
            "model": "large-v3",
            "params": {"diarize": True},
        }
    }
    db.mark("ep1", transcribed=True, provenance=prov)
    row = db.get_episode("ep1")
    assert row["provenance"]["transcript"]["model"] == "large-v3"
    assert row["provenance"]["transcript"]["params"]["diarize"] is True


def test_provenance_merge_across_steps(db):
    """Each step key merges into the existing provenance dict."""
    db.mark("ep1", transcribed=True, provenance={"transcript": {"model": "large-v3"}})
    db.mark("ep1", polished=True, provenance={"polished": {"model": "qwen3:4b"}})
    db.mark("ep1", provenance={"english": {"model": "gpt-4o"}})
    row = db.get_episode("ep1")
    assert row["provenance"]["transcript"]["model"] == "large-v3"
    assert row["provenance"]["polished"]["model"] == "qwen3:4b"
    assert row["provenance"]["english"]["model"] == "gpt-4o"


def test_provenance_overwrite_same_step(db):
    """Writing the same step key overwrites it."""
    db.mark("ep1", provenance={"transcript": {"model": "small"}})
    db.mark("ep1", provenance={"transcript": {"model": "large-v3"}})
    row = db.get_episode("ep1")
    assert row["provenance"]["transcript"]["model"] == "large-v3"


def test_provenance_empty_by_default(db):
    db.mark("ep1", transcribed=True)
    row = db.get_episode("ep1")
    assert row["provenance"] == {}


def test_provenance_in_populate(db):
    """populate_from_scan creates rows with empty provenance."""
    episodes = [FakeEpisode(stem="ep1", transcribed=True)]
    db.populate_from_scan(episodes)
    row = db.get_episode("ep1")
    assert row["provenance"] == {}


# ── Step statuses ────────────────────────────────────────


def _make_status_row(
    transcribed=False,
    polished=False,
    translations=None,
    provenance=None,
):
    """Build a minimal status dict like PipelineDB.all_episodes() returns."""
    return {
        "transcribed": transcribed,
        "polished": polished,
        "indexed": False,
        "synthesized": False,
        "translations": translations or [],
        "provenance": provenance or {},
    }


class TestStepStatuses:
    """Test the _step_statuses() comparison logic from shows.py."""

    @staticmethod
    def _step_statuses(st, provenance, effective):
        from podcodex.api.routes.shows import _step_statuses

        return _step_statuses(st, provenance, effective)

    def test_none_when_not_done(self):
        st = _make_status_row()
        result = self._step_statuses(st, {}, {"model_size": "large-v3"})
        assert result["transcribe_status"] == "none"
        assert result["polish_status"] == "none"
        assert result["translate_status"] == "none"

    def test_done_when_matching(self):
        prov = {
            "transcript": {
                "model": "large-v3",
                "type": "validated",
                "params": {"diarize": True},
            },
            "polished": {
                "model": "qwen3:4b",
                "type": "validated",
                "params": {"mode": "ollama", "provider": ""},
            },
        }
        st = _make_status_row(transcribed=True, polished=True, provenance=prov)
        effective = {"model_size": "large-v3", "diarize": True, "llm_mode": "ollama"}
        result = self._step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "done"
        assert result["polish_status"] == "done"

    def test_outdated_model_mismatch(self):
        prov = {"transcript": {"model": "small", "params": {"diarize": True}}}
        st = _make_status_row(transcribed=True, provenance=prov)
        effective = {"model_size": "large-v3", "diarize": True}
        result = self._step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "outdated"

    def test_outdated_diarize_mismatch(self):
        prov = {"transcript": {"model": "large-v3", "params": {"diarize": False}}}
        st = _make_status_row(transcribed=True, provenance=prov)
        effective = {"model_size": "large-v3", "diarize": True}
        result = self._step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "outdated"

    def test_outdated_polish_provider_mismatch(self):
        prov = {
            "polished": {
                "model": "qwen3:4b",
                "params": {"mode": "ollama", "provider": ""},
            }
        }
        st = _make_status_row(polished=True, provenance=prov)
        effective = {"llm_mode": "api", "llm_provider": "openai"}
        result = self._step_statuses(st, prov, effective)
        assert result["polish_status"] == "outdated"

    def test_done_no_provenance(self):
        """Episodes without provenance default to 'done' (pre-existing episodes)."""
        st = _make_status_row(transcribed=True, polished=True)
        result = self._step_statuses(st, {}, {"model_size": "large-v3"})
        assert result["transcribe_status"] == "done"
        assert result["polish_status"] == "done"

    def test_done_no_defaults(self):
        """No defaults configured → everything is 'done'."""
        prov = {"transcript": {"model": "small", "type": "validated", "params": {}}}
        st = _make_status_row(transcribed=True, provenance=prov)
        result = self._step_statuses(st, prov, {})
        assert result["transcribe_status"] == "done"

    def test_translate_target_lang(self):
        prov = {
            "english": {
                "model": "gpt-4o",
                "params": {"mode": "api", "provider": "openai"},
            }
        }
        st = _make_status_row(translations=["english"], provenance=prov)
        effective = {
            "target_lang": "english",
            "llm_mode": "api",
            "llm_provider": "openai",
        }
        result = self._step_statuses(st, prov, effective)
        assert result["translate_status"] == "done"

    def test_translate_missing_target_lang(self):
        """Target lang configured but not translated → 'none'."""
        st = _make_status_row(translations=["french"])
        effective = {"target_lang": "english"}
        result = self._step_statuses(st, {}, effective)
        assert result["translate_status"] == "none"

    def test_translate_outdated_model(self):
        prov = {
            "english": {
                "model": "old-model",
                "params": {"mode": "api", "provider": "openai"},
            }
        }
        st = _make_status_row(translations=["english"], provenance=prov)
        effective = {
            "target_lang": "english",
            "llm_mode": "api",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
        }
        result = self._step_statuses(st, prov, effective)
        assert result["translate_status"] == "outdated"


# ── Resolve defaults ─────────────────────────────────────


class TestResolveDefaults:
    """Test the _resolve_defaults() merging logic from shows.py."""

    @staticmethod
    def _resolve_defaults(app_defaults, show_meta):
        from podcodex.api.routes.shows import _resolve_defaults

        return _resolve_defaults(app_defaults, show_meta)

    def test_app_defaults_only(self):
        result = self._resolve_defaults(
            {"model_size": "large-v3", "diarize": True}, None
        )
        assert result["model_size"] == "large-v3"
        assert result["diarize"] is True

    def test_show_overrides_app(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(name="test", pipeline=PipelineDefaults(model_size="small"))
        result = self._resolve_defaults({"model_size": "large-v3"}, show)
        assert result["model_size"] == "small"

    def test_show_empty_falls_back(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(name="test", pipeline=PipelineDefaults())  # all defaults
        result = self._resolve_defaults(
            {"model_size": "large-v3", "llm_mode": "ollama"}, show
        )
        assert result["model_size"] == "large-v3"
        assert result["llm_mode"] == "ollama"

    def test_show_diarize_false_overrides(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(name="test", pipeline=PipelineDefaults(diarize=False))
        result = self._resolve_defaults({"diarize": True}, show)
        assert result["diarize"] is False
