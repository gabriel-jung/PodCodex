"""The local-Ollama LLM path and the correct / translate entry points.

`test_llm_api.py` covers the hosted-API retry ladder; this file drives the
batch loop through a fake ``ollama.Client``: positional application across
batches and ``[BREAK]`` markers, the per-batch records that become
``llm_failures.json``, the correction length guard (and translation being
exempt from it), provider failures, and the parser's handling of answers
that are not a JSON array.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from podcodex.core import _utils
from podcodex.core import llm
from podcodex.core.llm import batch_segments_by_duration, call_and_parse


def _seg(text: str, start: float) -> dict:
    return {"speaker": "A", "text": text, "start": start, "end": start + 1.0}


BREAK = {"speaker": _utils.BREAK_SPEAKER, "text": "", "start": 50.0, "end": 50.0}


class _FakeOllama:
    """Stands in for ``ollama.Client``; each outcome answers one chat call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.chats: list[dict] = []

    def __call__(self, **_kwargs):  # the class is patched with this instance
        return self

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model="m:latest")])

    def chat(self, **kwargs):
        self.chats.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            message=SimpleNamespace(content=outcome),
            prompt_eval_count=10,
            done_reason="stop",
            eval_count=10,
            done=True,
        )


def _answer(*texts: str) -> str:
    return json.dumps([{"text": t} for t in texts])


@pytest.fixture
def ollama(monkeypatch):
    """Patch the ollama client, the version probe and sleep."""
    import time

    import ollama as ollama_pkg

    monkeypatch.setattr(llm, "_warn_if_ollama_too_old", lambda _host: None)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    def install(outcomes) -> _FakeOllama:
        fake = _FakeOllama(outcomes)
        monkeypatch.setattr(ollama_pkg, "Client", fake)
        return fake

    return install


def _run(segments, *, sink, batch_minutes=1.0, min_length_ratio=0.7):
    return llm.run_llm_pipeline(
        segments,
        "system",
        mode="ollama",
        model="m",
        batch_minutes=batch_minutes,
        merge=False,
        batch_sink=sink,
        min_length_ratio=min_length_ratio,
    )


# ── batch loop ───────────────────────────────────────────────────────────


def test_two_batches_apply_by_position_and_keep_breaks(ollama):
    """Batch 1 is two segments and a [BREAK], batch 2 one segment; indices
    restart per batch in the answer but land on the right segments."""
    segs = [_seg("one", 0.0), _seg("two", 10.0), BREAK, _seg("three", 100.0)]
    fake = ollama([_answer("ONE", "TWO"), _answer("THREE")])
    sink: list[dict] = []

    out = _run(segs, sink=sink)

    assert [s["text"] for s in out] == ["ONE", "TWO", "", "THREE"]
    assert out[2]["speaker"] == _utils.BREAK_SPEAKER
    assert [(b["batch"], b["status"], b["expected"]) for b in sink] == [
        (1, "ok", 2),
        (2, "ok", 1),
    ]
    # Absolute indices in the prompt and in the record, across batches.
    assert "[2] three" in fake.chats[1]["messages"][1]["content"]
    assert sink[1]["input"] == [{"index": 2, "text": "three"}]
    # The [BREAK] never reaches the model, and costs no call of its own.
    assert len(fake.chats) == 2
    assert all("[BREAK]" not in c["messages"][1]["content"] for c in fake.chats)


def test_a_dense_batch_is_split_by_item_count(ollama):
    """Past OLLAMA_MAX_BATCH_ITEMS the token budget no longer fits its caps."""
    n = llm.OLLAMA_MAX_BATCH_ITEMS + 5
    segs = [_seg(f"s{i}", float(i)) for i in range(n)]
    batches = llm._ollama_batches(segs, batch_minutes=15)
    assert [len(b) for b in batches] == [llm.OLLAMA_MAX_BATCH_ITEMS, 5]


def test_a_read_timeout_rejects_its_batch_and_the_run_continues(ollama):
    segs = [_seg("one", 0.0), _seg("two", 100.0)]
    ollama([httpx.ReadTimeout("slow"), _answer("TWO")])
    sink: list[dict] = []

    out = _run(segs, sink=sink)

    assert [s["text"] for s in out] == ["one", "TWO"]
    assert [b["status"] for b in sink] == ["rejected", "ok"]
    assert "did not answer" in sink[0]["reason"]


def test_an_unknown_model_stops_the_run_on_the_first_call(ollama):
    """A 404 fails every batch the same way; no retries, no second batch."""
    from ollama import ResponseError

    segs = [_seg("one", 0.0), _seg("two", 100.0)]
    fake = ollama([ResponseError("model not found", 404)] * 3)
    sink: list[dict] = []

    with pytest.raises(llm.LLMBatchError) as exc:
        _run(segs, sink=sink)

    assert exc.value.permanent
    assert "ollama pull" in str(exc.value)
    assert len(fake.chats) == 1
    assert [b["status"] for b in sink] == ["rejected"]


def test_a_server_error_is_retried(ollama):
    from ollama import ResponseError

    fake = ollama([ResponseError("overloaded", 503), _answer("ONE")])
    sink: list[dict] = []

    out = _run([_seg("one", 0.0)], sink=sink)

    assert [s["text"] for s in out] == ["ONE"]
    assert len(fake.chats) == 2


# ── length guard ─────────────────────────────────────────────────────────


def test_a_truncated_correction_keeps_the_original_and_flags_the_batch(ollama):
    original = "This is a fairly long sentence that the model cut short."
    ollama([_answer("Short.")])
    sink: list[dict] = []

    out = _run([_seg(original, 0.0)], sink=sink)

    assert out[0]["text"] == original
    assert sink[0]["status"] == "rejected"
    assert "kept their original text" in sink[0]["reason"]


def test_translation_to_a_compact_script_is_kept(ollama, monkeypatch):
    """English to Chinese is often a third of the characters; translation is
    exempt from the correction length guard, which would save it as the
    English source."""
    from podcodex.core import translate

    recorded: dict = {}
    monkeypatch.setattr(
        "podcodex.core.llm_failures.record_run",
        lambda *a, **kw: recorded.update(kw, step=a[2]),
    )
    english = "Hello, how are you doing today my friend?"
    chinese = "你好,我的朋友,你今天好吗?"
    ollama([_answer(chinese)])

    out = translate.translate_segments(
        [_seg(english, 0.0)],
        mode="ollama",
        source_lang="English",
        target_lang="Chinese",
        merge=False,
        output_dir="unused",
    )

    assert out[0]["text"] == chinese
    assert [b["status"] for b in recorded["records"]] == ["ok"]


def test_a_batch_answered_with_its_source_is_flagged_not_reverted(ollama, monkeypatch):
    """A small model sometimes returns the source verbatim; count and length
    checks both pass, so it must be flagged, not saved as a finished
    translation."""
    from podcodex.core import translate

    recorded: dict = {}
    monkeypatch.setattr(
        "podcodex.core.llm_failures.record_run",
        lambda *a, **kw: recorded.update(kw),
    )
    french = [
        "Il faut revenir un peu sur ce personnage.",
        "Oui.",
        "Oui, c'est tout le sujet du film, vraiment.",
    ]
    ollama([_answer(*french)])

    out = translate.translate_segments(
        [_seg(t, float(i)) for i, t in enumerate(french)],
        mode="ollama",
        source_lang="French",
        target_lang="Japanese",
        merge=False,
        output_dir="unused",
    )

    assert [s["text"] for s in out] == french
    (record,) = recorded["records"]
    assert record["status"] == "rejected"
    assert "2 of 2 segments came back unchanged" in record["reason"]


def test_short_segments_left_as_they_are_do_not_flag_a_translation():
    inputs = [
        _seg("OK.", 0.0),
        _seg("Stanley Kubrick", 1.0),
        _seg("Il a effectivement ce côté pantomime.", 2.0),
    ]
    outputs = [*inputs[:2], {**inputs[2], "text": "確かにパントマイムの側面がある。"}]
    assert llm._mostly_unchanged(inputs, outputs) is None


def test_correction_does_not_flag_unchanged_text(ollama):
    text = "This sentence was already correct and stays as it is."
    ollama([_answer(text)])
    sink: list[dict] = []

    _run([_seg(text, 0.0)], sink=sink)

    assert sink[0]["status"] == "ok"


@pytest.mark.parametrize(
    ("answer", "source", "expected"),
    [
        ("[86] [86] Il y a", "Il y a", "Il y a"),
        ("[3] Bonjour", "Bonjour", "Bonjour"),
        ("[3] note", "[3] note", "[3] note"),
        ("Voir [2]", "Voir", "Voir [2]"),
    ],
)
def test_echoed_position_markers_are_stripped(answer, source, expected):
    out = llm.apply_corrections([_seg(source, 0.0)], {0: {"text": answer}}, 0)
    assert out[0]["text"] == expected


def test_a_failed_run_leaves_the_failure_record_alone(ollama, monkeypatch):
    """No version is saved when the run raises, so llm_failures.json must
    keep describing the version on disk: the batch-fix flow patches that
    version by the recorded batch indices."""
    from ollama import ResponseError

    from podcodex.core import correct

    calls: list = []
    monkeypatch.setattr(
        "podcodex.core.llm_failures.record_run", lambda *a, **kw: calls.append(kw)
    )
    ollama([ResponseError("bad request", 400)])

    with pytest.raises(llm.LLMBatchError):
        correct.correct_segments(
            [_seg("one", 0.0)], mode="ollama", model="", merge=False
        )

    assert calls == []


def test_a_successful_run_records_the_model_that_ran(ollama, monkeypatch):
    from podcodex.core import correct
    from podcodex.core.constants import DEFAULT_OLLAMA_MODEL

    recorded: dict = {}
    monkeypatch.setattr(
        "podcodex.core.llm_failures.record_run",
        lambda *a, **kw: recorded.update(kw, step=a[2]),
    )
    ollama([_answer("ONE")])

    correct.correct_segments([_seg("one", 0.0)], mode="ollama", model="", merge=False)

    assert recorded["step"] == "corrected"
    assert recorded["model"] == DEFAULT_OLLAMA_MODEL


def test_effective_model_resolves_empty_picks():
    from podcodex.core.constants import DEFAULT_OLLAMA_MODEL, LLM_PROVIDER_DEFAULT_MODEL

    assert llm.effective_llm_model("ollama", "") == DEFAULT_OLLAMA_MODEL
    assert (
        llm.effective_llm_model("api", "", "openai")
        == LLM_PROVIDER_DEFAULT_MODEL["openai"]
    )
    assert llm.effective_llm_model("api", "", "custom") == ""
    assert llm.effective_llm_model("api", "chosen", "openai") == "chosen"


# ── parser ───────────────────────────────────────────────────────────────


def test_a_single_object_answer_is_one_item():
    """A lone object is one item, not enumerated: iterating a dict yields
    its keys, and a one-segment batch would save the literal text "text"."""
    assert llm.parse_llm_response('{"text": "Oui."}') == {0: {"text": "Oui."}}


def test_an_object_wrapping_one_array_is_that_array():
    raw = '{"segments": [{"text": "a"}, {"text": "b"}]}'
    assert llm.parse_llm_response(raw) == {0: {"text": "a"}, 1: {"text": "b"}}


@pytest.mark.parametrize("raw", ['"just a string"', "42", '{"a": 1, "b": 2}'])
def test_anything_else_is_a_parse_failure(raw):
    assert llm.parse_llm_response(raw) == {}


# ── Ollama pre-flight ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1:11434", "localhost", "https://x.example", "http://h/ollama/"],
)
def test_host_normalisation_matches_the_ollama_client(host):
    from ollama._client import _parse_host

    assert llm._normalize_ollama_host(host) == _parse_host(host)


def test_an_untagged_model_counts_as_latest():
    assert llm._ollama_model_key("llama3.1") == "llama3.1:latest"
    assert llm._ollama_model_key("qwen3:4b") == "qwen3:4b"


def test_probe_tells_a_taken_port_from_a_stopped_daemon(monkeypatch):
    """Another program on 11434 answers 404 to the Ollama API; the settings
    status must not call that "not running" while Ollama is open."""
    from ollama import ResponseError

    def taken(_host=None):
        raise ResponseError("", 404)

    def down(_host=None):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm, "list_pulled_ollama_models", taken)
    assert llm.probe_ollama("http://localhost:11434")["problem"] == "port_taken"
    monkeypatch.setattr(llm, "list_pulled_ollama_models", down)
    assert llm.probe_ollama("http://localhost:11434")["problem"] == "not_running"
    monkeypatch.setattr(llm, "list_pulled_ollama_models", lambda _h=None: ["m:latest"])
    assert llm.probe_ollama("http://localhost:11434") == {
        "reachable": True,
        "models": ["m:latest"],
        "problem": None,
        "error": None,
    }


# ── call_and_parse: one batch's answer, parsed or refused ────────────────


def make_call_fn(response: str):
    return lambda messages: response


def make_segments(*texts, speaker: str = "Alice", seg_duration: float = 10.0):
    return [
        {
            "speaker": speaker,
            "start": i * seg_duration,
            "end": (i + 1) * seg_duration,
            "text": t,
        }
        for i, t in enumerate(texts)
    ]


def test_call_and_parse_bad_json_keeps_original():
    batch = make_segments("Bonjour le monde")
    result = call_and_parse(batch, "sys", make_call_fn("not json at all"))
    assert len(result) == 1
    assert result[0]["text"] == "Bonjour le monde"


def test_call_and_parse_count_mismatch_rejects_whole_batch():
    """LLM returning fewer items than the batch = index-drift risk; reject everything."""
    batch = make_segments("Premier", "Deuxieme")
    response = json.dumps([{"index": 0, "text": "First"}])
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "Premier"
    assert result[1]["text"] == "Deuxieme"


def test_call_and_parse_strips_think_tags():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Hello"}])
    response = f"<think>some reasoning</think>\n{inner}"
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "Hello"


def test_call_and_parse_strips_markdown_fences():
    batch = make_segments("Bonjour")
    inner = json.dumps([{"index": 0, "text": "Hello"}])
    response = f"```json\n{inner}\n```"
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "Hello"


def test_call_and_parse_truncation_guard_disabled():
    """min_length_ratio=0 disables the guard."""
    batch = make_segments("This is a very long sentence that should not be truncated")
    response = json.dumps([{"index": 0, "text": "Short"}])
    result = call_and_parse(batch, "sys", make_call_fn(response), min_length_ratio=0)
    assert result[0]["text"] == "Short"


def test_call_and_parse_all_breaks_no_llm_call():
    batch = [
        {"speaker": "[BREAK]", "start": 0.0, "end": 5.0, "text": ""},
        {"speaker": "[BREAK]", "start": 10.0, "end": 15.0, "text": ""},
    ]

    def should_not_be_called(messages):
        raise AssertionError("LLM should not be called for all-break batches")

    result = call_and_parse(batch, "sys", should_not_be_called)
    assert len(result) == 2
    assert all(seg["speaker"] == "[BREAK]" for seg in result)


# ── batch_segments_by_duration ───────────────────────────────────────────


def test_batch_single_batch_when_under_limit():
    segments = make_segments("Short")
    batches = batch_segments_by_duration(segments, batch_minutes=15)
    assert len(batches) == 1


def test_batch_splits_by_duration():
    # Segments start every ~17 min → each falls in its own 15-min window.
    segments = [
        {"speaker": "A", "start": i * 1000, "end": i * 1000 + 300, "text": f"s{i}"}
        for i in range(3)
    ]
    batches = batch_segments_by_duration(segments, batch_minutes=15)
    assert len(batches) == 3


def test_batch_merges_tiny_overshoot_tail():
    # Segments run 5 s past the second 15-min cutoff. Expect 2 batches,
    # not 3 — the tiny tail is absorbed into the previous batch.
    segments = [
        {"speaker": "A", "start": float(s), "end": float(s) + 0.5, "text": f"s{s}"}
        for s in range(0, 1906, 100)
    ]
    batches = batch_segments_by_duration(segments, batch_minutes=15)
    assert len(batches) == 2
    assert sum(len(b) for b in batches) == len(segments)


def test_batch_groups_short_segments():
    # 4 × 5-minute segments → 15-minute limit fits 3 per batch, leaving 1.
    segments = [
        {"speaker": "A", "start": i * 300, "end": (i + 1) * 300, "text": f"s{i}"}
        for i in range(4)
    ]
    batches = batch_segments_by_duration(segments, batch_minutes=15)
    assert len(batches) == 2
    assert sum(len(b) for b in batches) == 4


def test_batch_empty():
    assert batch_segments_by_duration([], batch_minutes=15) == []


def test_batch_count_produces_exact_count():
    # 100 segments spanning ~50 min. Request 4 batches → exactly 4, even
    # though batch_minutes is left at the default. No spurious extra batch.
    segments = [
        {
            "speaker": "A",
            "start": float(i * 30),
            "end": float(i * 30) + 1,
            "text": f"s{i}",
        }
        for i in range(100)
    ]
    batches = batch_segments_by_duration(segments, batch_count=4)
    assert len(batches) == 4
    assert sum(len(b) for b in batches) == 100


def test_batch_count_overrides_minutes():
    # batch_minutes alone would split these into 10 batches; batch_count wins.
    segments = [
        {
            "speaker": "A",
            "start": float(i * 600),
            "end": float(i * 600) + 1,
            "text": f"s{i}",
        }
        for i in range(10)
    ]
    batches = batch_segments_by_duration(segments, batch_minutes=15, batch_count=2)
    assert len(batches) == 2
    assert sum(len(b) for b in batches) == 10


def test_batch_count_one_is_single_batch():
    segments = make_segments("a", "b", "c")
    assert len(batch_segments_by_duration(segments, batch_count=1)) == 1
