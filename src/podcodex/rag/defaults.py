"""
podcodex.rag.defaults — Centralized model registry and default parameters.

All model keys, HF model IDs, and tunable defaults live here.
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
    key: str  # identifier used in CLI args, collection names, bot commands
    label: str  # human-readable display name
    dim: int  # dense vector dimension
    sparse: bool  # True → native sparse vectors (full hybrid); False → dense only
    description: str


MODELS: dict[str, ModelSpec] = {
    "bge-m3": ModelSpec(
        key="bge-m3",
        label="BGE-M3",
        dim=1024,
        sparse=True,
        description="BAAI/bge-m3 — multilingual, dense + sparse (full hybrid)",
    ),
    "e5-small": ModelSpec(
        key="e5-small",
        label="E5 Small",
        dim=384,
        sparse=False,
        description="intfloat/multilingual-e5-small — fast, dense only",
    ),
    "e5-large": ModelSpec(
        key="e5-large",
        label="E5 Large",
        dim=1024,
        sparse=False,
        description="intfloat/multilingual-e5-large — accurate, dense only",
    ),
    "pplx": ModelSpec(
        key="pplx",
        label="Perplexity",
        dim=1024,
        sparse=False,
        description="perplexity-ai/pplx-embed — context-aware, dense only (experimental)",
    ),
}

DEFAULT_MODEL: str = "bge-m3"


# ──────────────────────────────────────────────
# Chunking strategy registry
# ──────────────────────────────────────────────

CHUNKING_STRATEGIES: dict[str, str] = {
    "semantic": "Semantic similarity (Chonkie) — recommended default",
    "speaker": "One chunk per speaker turn — fast, no extra deps",
}

DEFAULT_CHUNKING: str = "semantic"


# ──────────────────────────────────────────────
# HF model identifiers
# ──────────────────────────────────────────────

BGE_M3_MODEL: str = "BAAI/bge-m3"
E5_SMALL_MODEL: str = "intfloat/multilingual-e5-small"
E5_LARGE_MODEL: str = "intfloat/multilingual-e5-large"
PPLX_PASSAGE_MODEL: str = "perplexity-ai/pplx-embed-context-v1-0.6B"
PPLX_QUERY_MODEL: str = "perplexity-ai/pplx-embed-v1-0.6B"

# Chonkie's internal splitting model — not exposed to users
CHUNKER_MODEL: str = E5_SMALL_MODEL


# ──────────────────────────────────────────────
# Tunable defaults
# ──────────────────────────────────────────────

CHUNK_SIZE: int = 256
CHUNK_THRESHOLD: float = 0.5
ALPHA: float = 0.5
TOP_K: int = 5
