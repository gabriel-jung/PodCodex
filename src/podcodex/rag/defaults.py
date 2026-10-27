"""
podcodex.rag.defaults — Centralized model registry and default parameters.

All model specs, HF model IDs, and tunable defaults live here.
CLI, bot, and RAG modules import from this file — never redefine
constants elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass


# ──────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a supported embedding model.

    Attributes:
        label: Human-readable display name.
        hf_model: HuggingFace model ID for passage encoding.
        dim: Dense vector dimensionality.
        description: Human-readable summary, also used in Discord command choices.
        hf_query_model: Separate query model if different from ``hf_model``
            (used by Perplexity only; empty string otherwise).
        revision, query_revision: Commit sha of ``hf_model`` and
            ``hf_query_model``. Set for every model loaded with
            ``trust_remote_code``: that code runs in the app and on a bot
            host, so it is pinned to a known commit instead of whatever
            the repo holds at download time. Bump deliberately, and
            re-embed the collections that model built.
    """

    label: str
    hf_model: str
    dim: int
    description: str
    hf_query_model: str = ""
    revision: str | None = None
    query_revision: str | None = None


MODELS: dict[str, ModelSpec] = {
    "bge-m3": ModelSpec(
        label="BGE-M3",
        hf_model="BAAI/bge-m3",
        dim=1024,
        description="Default, good search quality",
    ),
    "e5-small": ModelSpec(
        label="E5 Small",
        hf_model="intfloat/multilingual-e5-small",
        dim=384,
        description="Very fast to run",
    ),
    "e5-large": ModelSpec(
        label="E5 Large",
        hf_model="intfloat/multilingual-e5-large",
        dim=1024,
        description="Better than E5 Small but slower",
    ),
    "pplx-0.6B": ModelSpec(
        label="Perplexity 0.6B",
        hf_model="perplexity-ai/pplx-embed-context-v1-0.6B",
        dim=1024,
        description="Context-aware, slow on CPU",
        hf_query_model="perplexity-ai/pplx-embed-v1-0.6B",
        # main on 2026-06-16 and 2026-06-02 (pinned 2026-10-27).
        revision="b42df969d4d78d1840769e45c68c4e4ce763768b",
        query_revision="2c4d510dd4a732063c31a0f70193e35067b51fd8",
    ),
    "pplx-4B": ModelSpec(
        label="Perplexity 4B",
        hf_model="perplexity-ai/pplx-embed-context-v1-4B",
        dim=2560,
        description="Context-aware, GPU strongly recommended",
        hf_query_model="perplexity-ai/pplx-embed-v1-4B",
        # main on 2026-06-16 and 2026-06-02 (pinned 2026-10-27).
        revision="679845b2236c653177fff8fa3ffc1b22f3090d30",
        query_revision="06456497a00540a582918fe8dcd3a5eabb207772",
    ),
}

DEFAULT_MODEL = "bge-m3"


# ──────────────────────────────────────────────
# Chunking strategy registry
# ──────────────────────────────────────────────

CHUNKING_STRATEGIES: dict[str, str] = {
    "semantic": "Semantic similarity (Chonkie) — recommended default",
    "speaker": "One chunk per speaker turn — fast, no extra deps",
}

DEFAULT_CHUNKING = "semantic"

# Chonkie's internal splitting model — not exposed to users
CHUNKER_MODEL_KEY = "e5-small"


# ──────────────────────────────────────────────
# Tunable defaults
# ──────────────────────────────────────────────

CHUNK_SIZE = 256
CHUNK_THRESHOLD = 0.5
ALPHA = 0.5
TOP_K = 5

# Context expansion: number of neighboring chunks fetched around each hit
# when building a "scene" (MCP get_context tool).
CONTEXT_WINDOW = 3
