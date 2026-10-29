"""The closing summary of an auto correct / translate run."""

from podcodex.api.routes._helpers import llm_run_outcome
from podcodex.core.llm_failures import save_batch_records
from tests.fixtures.episodes import audio_and_output_dir


def _record(tmp_path, statuses, version_id):
    audio, _ = audio_and_output_dir(tmp_path)
    base = tmp_path / "s" / "ep" / "ep"
    save_batch_records(
        base,
        "french",
        model="m",
        mode="ollama",
        records=[{"batch": i, "status": s} for i, s in enumerate(statuses, 1)],
        version_id=version_id,
    )
    return audio


def test_a_fully_rejected_run_says_so(tmp_path):
    audio = _record(tmp_path, ["rejected"], "v1")
    out = llm_run_outcome(audio, None, "french", "v1")
    assert out["rejected"] == 1
    assert "1 of 1 batches rejected" in out["message"]


def test_a_clean_run_adds_nothing(tmp_path):
    audio = _record(tmp_path, ["ok", "ok"], "v1")
    assert llm_run_outcome(audio, None, "french", "v1") == {}


def test_an_older_runs_record_is_not_reported(tmp_path):
    audio = _record(tmp_path, ["rejected"], "v0")
    assert llm_run_outcome(audio, None, "french", "v1") == {}
