"""
podcodex.rag.defaults — Centralized model registry and default parameters.

All model specs, HF model IDs, and tunable defaults live here.
CLI, bot, Streamlit app, and RAG modules import from this file — never
redefine constants elsewhere.
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
    """

    label: str
    hf_model: str
    dim: int
    description: str
    hf_query_model: str = ""


MODELS: dict[str, ModelSpec] = {
    "bge-m3": ModelSpec(
        label="BGE-M3",
        hf_model="BAAI/bge-m3",
        dim=1024,
        description="BAAI/bge-m3 — multilingual, dense + sparse (full hybrid)",
    ),
    "e5-small": ModelSpec(
        label="E5 Small",
        hf_model="intfloat/multilingual-e5-small",
        dim=384,
        description="intfloat/multilingual-e5-small — fast, dense only",
    ),
    "e5-large": ModelSpec(
        label="E5 Large",
        hf_model="intfloat/multilingual-e5-large",
        dim=1024,
        description="intfloat/multilingual-e5-large — accurate, dense only",
    ),
    "pplx": ModelSpec(
        label="Perplexity",
        hf_model="perplexity-ai/pplx-embed-context-v1-0.6B",
        dim=1024,
        description="perplexity-ai/pplx-embed — context-aware, dense only (experimental)",
        hf_query_model="perplexity-ai/pplx-embed-v1-0.6B",
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
CHUNKER_MODEL = MODELS["e5-small"].hf_model


# ──────────────────────────────────────────────
# Tunable defaults
# ──────────────────────────────────────────────

CHUNK_SIZE = 256
CHUNK_THRESHOLD = 0.5
ALPHA = 0.5
TOP_K = 5
