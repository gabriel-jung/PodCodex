"""Probe the raw shape of Ollama output for correction prompts.

Reproduces exactly what `core.correct.correct_segments` sends to the daemon
and prints the raw response, then runs it through `parse_llm_response` so we
can see whether the model is emitting a top-level array (what we want), an
object with a wrapping key, a single concatenated string, or something else
entirely. The "1 items for 64 segments" rejection suggests the model is
likely returning `{"key": [...]}` or a single string, but until we see the
raw bytes it's a guess.

Run:  .venv/bin/python scripts/debug_ollama_output.py [model] [batch-size]
"""

from __future__ import annotations

import json
import sys

from podcodex.core._utils import (
    correction_schema,
    format_segments,
    ollama_host,
    parse_llm_response,
)
from podcodex.core.correct import _build_prompt


SAMPLE_SEGMENTS = [
    {"text": "euh ouais c est ca"},
    {"text": "[music]"},
    {"text": "donc on parlait de deep mind tout a l heure"},
    {"text": "ben en fait il faut comprendre que"},
    {"text": "voila c est tout"},
]


def main(model: str = "qwen3:0.6b", n_extra: int = 0) -> None:
    from ollama import Client

    segments = SAMPLE_SEGMENTS + [
        {"text": f"segment numero {i}"} for i in range(n_extra)
    ]
    system_prompt = _build_prompt(source_lang="French", engine="whisper")
    user_content = format_segments(segments, instruction="Correct", start_index=0)

    print(f"Host:      {ollama_host()}")
    print(f"Model:     {model}")
    print(f"Segments:  {len(segments)}")
    print("=" * 70)
    print("SYSTEM PROMPT")
    print("=" * 70)
    print(system_prompt)
    print()
    print("=" * 70)
    print("USER MESSAGE")
    print("=" * 70)
    print(user_content)
    print()

    client = Client(host=ollama_host())
    schema = correction_schema(len(segments))
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        options={"temperature": 0.1},
        format=schema,
        think=False,
    )
    raw = resp.message.content
    print(f"done_reason: {resp.done_reason!r}")
    print(f"eval_count: {resp.eval_count}")
    print(f"prompt_eval_count: {resp.prompt_eval_count}")
    if resp.message.thinking:
        print(
            f"thinking ({len(resp.message.thinking)} chars): {resp.message.thinking[:200]!r}"
        )

    print("=" * 70)
    print("RAW RESPONSE")
    print("=" * 70)
    print(raw)
    print()
    print("=" * 70)
    print("DECODED SHAPE")
    print("=" * 70)
    try:
        decoded = json.loads(raw)
        print(f"top-level type: {type(decoded).__name__}")
        if isinstance(decoded, list):
            print(f"length:         {len(decoded)}")
            if decoded:
                print(f"first item:     {decoded[0]!r}")
        elif isinstance(decoded, dict):
            print(f"keys:           {list(decoded.keys())}")
            for k, v in decoded.items():
                marker = f"list[{len(v)}]" if isinstance(v, list) else type(v).__name__
                print(f"  {k!r}: {marker}")
    except Exception as e:
        print(f"json.loads failed: {e}")

    print()
    print("=" * 70)
    print("parse_llm_response() RESULT")
    print("=" * 70)
    by_index = parse_llm_response(raw)
    print(f"items: {len(by_index)} / expected {len(segments)}")
    for i, item in by_index.items():
        print(f"  [{i}] {item!r}")


if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "qwen3:0.6b"
    n_extra = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(model_arg, n_extra)
