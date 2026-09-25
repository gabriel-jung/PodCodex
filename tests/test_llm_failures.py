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


# ── The rejected-batch loop: record, list, resolve ───────────────────────


def _records(*statuses):
    return [
        {"batch": i + 1, "status": st, "expected": 2, "got": 2}
        for i, st in enumerate(statuses)
    ]


def test_record_run_writes_the_episode_section(tmp_path):
    from podcodex.core.llm_failures import get_step, record_run

    show = tmp_path / "show"
    (show / "ep").mkdir(parents=True)
    audio = show / "ep.mp3"
    audio.touch()

    record_run(
        str(audio),
        None,
        "french",
        model="m",
        mode="api",
        records=_records("ok", "rejected"),
    )

    section = get_step(str(audio), None, "french")
    assert section["rejected"] == 1 and section["total_batches"] == 2
    assert section["model"] == "m" and section["mode"] == "api"


def test_record_run_is_a_no_op_without_records_or_an_episode(tmp_path):
    from podcodex.core.llm_failures import record_run

    record_run(None, None, "corrected", model="m", mode="api", records=_records("ok"))
    record_run(
        str(tmp_path / "x.mp3"), None, "corrected", model="m", mode="api", records=[]
    )
    assert not list(tmp_path.rglob("llm_failures.json"))


def test_rejected_steps_lists_only_steps_with_a_rejected_batch(tmp_path):
    from podcodex.core.llm_failures import rejected_steps

    base = _base(tmp_path)
    save_batch_records(base, "corrected", model="m", mode="api", records=_records("ok"))
    save_batch_records(
        base, "french", model="m", mode="api", records=_records("ok", "rejected")
    )

    assert rejected_steps(base.parent) == ["french"]


def test_resolving_the_last_rejected_batch_drops_the_section(tmp_path):
    from podcodex.core.llm_failures import resolve_batches

    base = _base(tmp_path)
    save_batch_records(
        base,
        "french",
        model="m",
        mode="api",
        records=_records("rejected", "ok", "rejected"),
    )
    save_batch_records(base, "corrected", model="m", mode="api", records=_records("ok"))

    assert resolve_batches(base, "french", [1]) == 1
    assert load_failures(base)["french"]["rejected"] == 1

    assert resolve_batches(base, "french", [3]) == 0
    assert "french" not in load_failures(base)
    assert "corrected" in load_failures(base)
    assert resolve_batches(base, "missing", [1]) == 0
