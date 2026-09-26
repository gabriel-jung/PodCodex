"""A stand-in LLM run for tests that drive correct / translate without a model.

``fake_llm_run`` is what ``resolve_llm_run`` returns; ``stub_llm_resolver``
patches the resolver so a route or the batch runner never probes a live
Ollama (a test that reached it passed or failed by which models were pulled).
"""

from __future__ import annotations

from podcodex.core.llm_resolver import LLMRun


def fake_llm_run(model: str = "m", mode: str = "ollama") -> LLMRun:
    return LLMRun(mode=mode, model=model, provider=None, api_base_url="", api_key=None)


def stub_llm_resolver(monkeypatch) -> None:
    """``resolve_llm_run`` returns a fake run for whatever model was asked."""
    import podcodex.core.llm_resolver as resolver

    monkeypatch.setattr(
        resolver,
        "resolve_llm_run",
        lambda _mode, _profile, _key, model="", *_a, **_k: fake_llm_run(model or "m"),
    )
