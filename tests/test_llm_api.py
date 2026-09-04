"""Resilience of the hosted-API LLM path (`core/_utils.run_api`).

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

from podcodex.core import _utils


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

    def run(outcomes, **kwargs):
        client = _FakeClient(outcomes)
        monkeypatch.setattr(openai, "OpenAI", lambda **_kw: client)
        segs = kwargs.pop("segments", _segments())
        sink: list[dict] = []
        out = _utils.run_api(
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
    assert slept == [_utils.API_BACKOFF_BASE_S**0]
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

    assert slept == [_utils.API_RETRY_AFTER_MAX_S]


def test_retries_stop_at_the_attempt_ceiling(api):
    import openai

    with pytest.raises(openai.RateLimitError):
        api([_status_error(openai.RateLimitError, 429)] * _utils.API_MAX_ATTEMPTS)


def test_a_client_error_raises_on_the_first_attempt(api):
    """A bad key or an unknown model fails the same way three times."""
    import openai

    with pytest.raises(openai.BadRequestError):
        api([_status_error(openai.BadRequestError, 400)] * 3)


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


def test_a_missing_key_names_the_provider_variables(monkeypatch):
    """No generic `API_KEY` fallback: a name that common is very likely
    already set in the shell for something else, and it was sent to whatever
    base_url the profile carried."""
    monkeypatch.setenv("API_KEY", "unrelated-shell-value")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError) as exc:
        _utils.run_api(_segments(), "system", "m", "https://api.example/v1", None)

    assert "OPENAI_API_KEY" in str(exc.value)
