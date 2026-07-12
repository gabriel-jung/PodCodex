"""Tests for podcodex.core.llm_failures — per-batch LLM outcome records."""

from podcodex.core.llm_failures import (
    clear_step,
    failures_path,
    load_failures,
    save_batch_records,
)


def _base(tmp_path):
    """An AudioPaths-style base: {episode_dir}/{stem}. Episode dir is the parent."""
    ep_dir = tmp_path / "episode"
    ep_dir.mkdir()
    return ep_dir / "episode"


def test_load_missing_returns_empty(tmp_path):
    assert load_failures(_base(tmp_path)) == {}


def test_save_and_load_roundtrip(tmp_path):
    base = _base(tmp_path)
    records = [
        {"batch": 1, "status": "ok", "expected": 60, "got": 60},
        {"batch": 2, "status": "rejected", "expected": 60, "got": 58},
    ]
    save_batch_records(base, "corrected", model="m", mode="ollama", records=records)

    data = load_failures(base)
    section = data["corrected"]
    assert section["total_batches"] == 2
    assert section["rejected"] == 1
    assert section["batches"] == records
    assert failures_path(base).is_file()


def test_clear_step_removes_section_and_file(tmp_path):
    base = _base(tmp_path)
    save_batch_records(base, "corrected", model="m", mode="api", records=[])
    save_batch_records(base, "fr", model="m", mode="api", records=[])

    assert clear_step(base, "corrected") is True
    assert "corrected" not in load_failures(base)
    assert "fr" in load_failures(base)  # other section survives

    assert clear_step(base, "fr") is True
    assert not failures_path(base).exists()  # file gone when last section cleared

    assert clear_step(base, "corrected") is False  # nothing left to clear
