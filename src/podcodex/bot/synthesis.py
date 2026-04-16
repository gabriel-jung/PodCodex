"""podcodex.bot.synthesis — LLM-synthesized Q&A answers for /ask."""

from __future__ import annotations

import os

from loguru import logger

from podcodex.bot.formatting import fmt_time
from podcodex.core.constants import LLM_PROVIDERS

# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a podcast research assistant. Answer the user's question using ONLY the \
transcript excerpts provided below. If the answer cannot be found in the excerpts, \
respond with exactly: "I couldn't find relevant information in the indexed transcripts."
Do not use external knowledge. Be concise and direct. \
Cite excerpt numbers inline where relevant (e.g. [1], [2])."""


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _format_chunks(chunks: list[dict]) -> str:
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        speaker = chunk.get("speaker") or chunk.get("dominant_speaker") or "Unknown"
        start = chunk.get("start", 0.0)
        episode = chunk.get("episode_title") or chunk.get("episode", "")
        show = chunk.get("show", "")
        text = chunk.get("text", "")

        parts: list[str] = [f"[{i}]"]
        if show:
            parts.append(show)
        if episode:
            parts.append(episode)
        parts.append(f"{speaker} @ {fmt_time(start)}")

        lines.append(" — ".join(parts) + f"\n{text}")
    return "\n\n".join(lines)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def synthesize_answer(
    question: str,
    chunks: list[dict],
    *,
    provider: str,
    model: str,
    api_base_url: str,
    api_key: str | None,
) -> str:
    """Synthesize a grounded answer from retrieved chunks.

    Uses an OpenAI-compatible API (same pattern as correct/translate pipeline).
    API key is resolved from the provider's env variable; the explicit *api_key*
    argument is used only when passed non-None (e.g. from CLI ``--ask-api-key``).

    Raises:
        ValueError: if no API key is available.
        openai.OpenAIError: on API failure.
    """
    from openai import OpenAI

    resolved_url = api_base_url
    resolved_model = model

    if provider and provider in LLM_PROVIDERS:
        spec = LLM_PROVIDERS[provider]
        resolved_url = resolved_url or spec["url"]
        resolved_model = resolved_model or spec["model"]
        env_var = spec.get("env_var", "")
        if not api_key and env_var:
            api_key = os.environ.get(env_var)

    api_key = api_key or os.environ.get("ASK_API_KEY")
    if not api_key:
        env_hint = ""
        if provider and provider in LLM_PROVIDERS:
            env_hint = f" Set {LLM_PROVIDERS[provider].get('env_var', 'ASK_API_KEY')}."
        raise ValueError(f"No API key found for provider {provider!r}.{env_hint}")

    context = _format_chunks(chunks)
    user_message = f"Transcript excerpts:\n\n{context}\n\nQuestion: {question}"

    logger.debug(
        f"ask: synthesizing via {provider!r} model={resolved_model!r} "
        f"chunks={len(chunks)}"
    )

    client = OpenAI(api_key=api_key, base_url=resolved_url or None)
    response = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
