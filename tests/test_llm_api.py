"""Resilience of the hosted-API LLM path (`core/llm.run_api`).

A long episode is split into many batches and every one already completed is
lost when a call raises out of the loop, so the two things pinned here are
that a transient provider failure is retried rather than fatal, and that an
unusable response (empty `choices`, `content: null`) is recorded as one
rejected batch rather than an exception. The ollama sibling has had both for
a while; this is the paid path catching up.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import httpx
import pytest

from podcodex.core import llm


def _segments(n: int = 2) -> list[dict]:
    return [
        {"text": f"line {i}", "speaker": "A", "start": float(i), "end": float(i) + 1.0}
        for i in range(n)
    ]


def _completion(content: str | None, *, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ]
    )


def _good_response(segments: list[dict]):
    import json

    return _completion(
        json.dumps([{"index": i, "text": s["text"]} for i, s in enumerate(segments)])
    )


def _status_error(cls, status: int, headers: dict | None = None):
    request = httpx.Request("POST", "https://api.example/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("boom", response=response, body=None)


class _FakeCompletions:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes):
        self.completions = _FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture
def api(monkeypatch):
    """Patch the OpenAI client and sleep; returns a runner over one batch."""
    import openai

    slept: list[float] = []
    # run_api imports `time` inside the retry helper, so patch the module.
    monkeypatch.setattr(time, "sleep", slept.append)

    def run(outcomes, *, sink: list[dict] | None = None, client_kwargs=None, **kwargs):
        client = _FakeClient(outcomes)

        def make_client(**kw):
            if client_kwargs is not None:
                client_kwargs.update(kw)
            return client

        monkeypatch.setattr(openai, "OpenAI", make_client)
        segs = kwargs.pop("segments", _segments())
        sink = [] if sink is None else sink
        out = llm.run_api(
            segs,
            "system",
            "some-model",
            "https://api.example/v1",
            "key",
            batch_sink=sink,
            **kwargs,
        )
        return out, sink, client.completions, slept

    return run


def test_a_rate_limit_is_retried_and_the_run_survives(api):
    import openai

    segs = _segments()
    out, sink, calls, slept = api(
        [_status_error(openai.RateLimitError, 429), _good_response(segs)],
        segments=segs,
    )

    assert calls.calls == 2
    assert slept == [llm.API_BACKOFF_BASE_S**0]
    assert [s["text"] for s in out] == [s["text"] for s in segs]
    assert [b["status"] for b in sink] == ["ok"]


def test_retry_after_wins_over_the_backoff_ladder(api):
    import openai

    segs = _segments()
    _out, _sink, _calls, slept = api(
        [
            _status_error(openai.RateLimitError, 429, {"retry-after": "5"}),
            _good_response(segs),
        ],
        segments=segs,
    )

    assert slept == [5.0]


def test_a_retry_after_beyond_the_cap_is_clamped(api):
    import openai

    segs = _segments()
    _out, _sink, _calls, slept = api(
        [
            _status_error(openai.RateLimitError, 429, {"retry-after": "3600"}),
            _good_response(segs),
        ],
        segments=segs,
    )

    assert slept == [llm.API_RETRY_AFTER_MAX_S]


def test_retries_stop_at_the_attempt_ceiling(api):
    """A one-batch run whose only batch fails is an error, not a saved copy
    of the source, and the failed batch is still on record."""
    import openai

    sink: list[dict] = []
    with pytest.raises(llm.LLMBatchError) as exc:
        api(
            [_status_error(openai.RateLimitError, 429)] * llm.API_MAX_ATTEMPTS,
            sink=sink,
        )
    assert not exc.value.permanent
    assert [b["status"] for b in sink] == ["rejected"]
    assert sink[0]["reason"].startswith("provider error")


def test_a_bad_key_stops_the_run_on_the_first_attempt(api):
    """A 401 fails every batch the same way: no retries, no second batch."""
    import openai

    sink: list[dict] = []
    with pytest.raises(llm.LLMBatchError) as exc:
        api([_status_error(openai.AuthenticationError, 401)] * 3, sink=sink)
    assert exc.value.permanent
    assert isinstance(exc.value.__cause__, openai.AuthenticationError)
    assert len(sink) == 1


def test_a_content_400_rejects_only_its_batch(api):
    """Context length or a content filter is about one batch; stopping the
    run there threw away every batch already paid for."""
    import json

    import openai

    segs = [
        {"text": f"line {i}", "speaker": "A", "start": i * 100.0, "end": i * 100.0 + 1}
        for i in range(2)
    ]
    out, sink, calls, _slept = api(
        [
            _status_error(openai.BadRequestError, 400),
            _completion(json.dumps([{"text": "fixed 1"}])),
        ],
        segments=segs,
        batch_minutes=1,
    )

    assert [s["text"] for s in out] == ["line 0", "fixed 1"]
    assert [b["status"] for b in sink] == ["rejected", "ok"]
    assert calls.calls == 2


def test_a_failed_batch_keeps_the_batches_around_it(api):
    """Three batches, the middle one exhausts its retries: the run finishes
    with batches 1 and 3 processed and batch 2 kept as source, recorded as
    rejected so the manual-fix flow can pick it up."""
    import json

    import openai

    segs = [
        {"text": f"line {i}", "speaker": "A", "start": i * 100.0, "end": i * 100.0 + 1}
        for i in range(3)
    ]

    def reply(text: str):
        return _completion(json.dumps([{"text": text}]))

    outcomes = [
        reply("fixed 0"),
        *[_status_error(openai.InternalServerError, 503)] * llm.API_MAX_ATTEMPTS,
        reply("fixed 2"),
    ]
    out, sink, _calls, _slept = api(outcomes, segments=segs, batch_minutes=1)

    assert [s["text"] for s in out] == ["fixed 0", "line 1", "fixed 2"]
    assert [b["status"] for b in sink] == ["ok", "rejected", "ok"]


def test_the_sdk_does_not_retry_underneath_our_ladder(api):
    """SDK retries inside our attempts multiplied requests (3 x 3)."""
    kwargs: dict = {}
    api([_good_response(_segments())], client_kwargs=kwargs)
    assert kwargs["max_retries"] == 0
    assert kwargs["timeout"].read == llm.API_READ_TIMEOUT_S


def test_a_server_error_is_retried(api):
    import openai

    segs = _segments()
    _out, _sink, calls, _slept = api(
        [_status_error(openai.InternalServerError, 503), _good_response(segs)],
        segments=segs,
    )

    assert calls.calls == 2


def test_an_empty_choices_list_rejects_the_batch_instead_of_raising(api):
    segs = _segments()
    out, sink, _calls, _slept = api([SimpleNamespace(choices=[])], segments=segs)

    assert [s["text"] for s in out] == [s["text"] for s in segs]
    assert [b["status"] for b in sink] == ["rejected"]


def test_a_null_content_rejects_the_batch_instead_of_raising(api):
    """Refusals and reasoning length-stops answer with content: null."""
    segs = _segments()
    out, sink, _calls, _slept = api(
        [_completion(None, finish_reason="length")], segments=segs
    )

    assert [s["text"] for s in out] == [s["text"] for s in segs]
    assert [b["status"] for b in sink] == ["rejected"]


def test_a_missing_key_never_falls_back_to_the_environment(monkeypatch):
    """Keys come from the pool only: a generic `API_KEY` or a provider
    variable in the shell is never sent to whatever base_url the profile
    carries."""
    monkeypatch.setenv("API_KEY", "unrelated-shell-value")
    monkeypatch.setenv("OPENAI_API_KEY", "also-unrelated")

    with pytest.raises(ValueError, match="No API key"):
        llm.run_api(
            _segments(),
            "system",
            "m",
            "https://api.example/v1",
            None,
            provider="openai",
        )
