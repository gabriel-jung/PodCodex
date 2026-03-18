"""
podcodex.rag.embedder — Embedding models for podcast RAG.

All embedders expose the same interface:

    encode_passages(chunks: list[dict]) -> np.ndarray  shape (n, dim), float32
    encode_query(query: str)            -> np.ndarray  shape (dim,),   float32

Hybrid search (dense + BM25) is handled separately by the Retriever;
embedders only produce dense vectors.

Factory:
    get_embedder(model_key: str) -> embedder instance
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from podcodex.rag.defaults import MODELS


# ──────────────────────────────────────────────
# PplxEmbedder
# ──────────────────────────────────────────────

_PPLX_SPEC = MODELS["pplx"]


class PplxEmbedder:
    """
    Context-aware embedder using Perplexity's pplx-embed-context-v1-0.6B.

    Passages are encoded with full episode context: all chunks from the same
    episode are passed together so each embedding is influenced by its neighbors.
    Query encoding uses the separate pplx-embed-v1-0.6B query model.

    Dense vectors (dim from MODELS registry). Unnormalized — Qdrant handles
    cosine normalization at query time.
    """

    def __init__(self, device: str = "cpu"):
        from transformers import AutoModel
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading PplxEmbedder ({_PPLX_SPEC.hf_model}) on {device}")
        self._ctx_model = AutoModel.from_pretrained(
            _PPLX_SPEC.hf_model, trust_remote_code=True
        ).to(device)
        self._query_model = SentenceTransformer(
            _PPLX_SPEC.hf_query_model, trust_remote_code=True, device=device
        )
        self._dim = _PPLX_SPEC.dim

    def encode_passages(self, chunks: list[dict]) -> np.ndarray:
        """
        Encode passages with full episode context (grouped by episode).

        All chunks from the same episode are passed as a single batch so
        each embedding is influenced by its neighbors (context-aware).

        Returns:
            np.ndarray of shape (n, dim), dtype float32
        """
        episode_indices: dict[str, list[int]] = {}
        for i, chunk in enumerate(chunks):
            episode_indices.setdefault(chunk.get("episode", ""), []).append(i)

        result = np.empty((len(chunks), self._dim), dtype=np.float32)
        for episode, indices in episode_indices.items():
            texts = [chunks[i]["text"] for i in indices]
            episode_embs = self._ctx_model.encode([texts])[0]  # (n_chunks, dim)
            for pos, idx in enumerate(indices):
                result[idx] = episode_embs[pos].astype(np.float32)

        logger.info(f"PplxEmbedder: encoded {len(chunks)} passages")
        return result

    def encode_query(self, query: str) -> np.ndarray:
        """Returns float32 vector of dim from MODELS registry."""
        emb = self._query_model.encode(query)
        return np.array(emb, dtype=np.float32)


# ──────────────────────────────────────────────
# E5Embedder
# ──────────────────────────────────────────────


class E5Embedder:
    """
    Multilingual E5 embedder — small (384-dim) or large (1024-dim).

    Uses "passage: " / "query: " prefixes as required by the model.
    Returns L2-normalized dense float32 vectors.
    """

    def __init__(
        self,
        model_key: str = "e5-small",
        device: str = "cpu",
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        hf_model = MODELS[model_key].hf_model
        logger.info(f"Loading E5Embedder ({hf_model}) on {device}")
        self._model = SentenceTransformer(hf_model, device=device)
        self._batch_size = batch_size

    def encode_passages(self, chunks: list[dict]) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (n, dim), dtype float32, L2-normalized
        """
        texts = ["passage: " + c["text"] for c in chunks]
        embs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        logger.info(f"E5Embedder: encoded {len(chunks)} passages")
        return np.array(embs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Returns normalized float32 vector."""
        emb = self._model.encode("query: " + query, normalize_embeddings=True)
        return np.array(emb, dtype=np.float32).squeeze()


# ──────────────────────────────────────────────
# BGEEmbedder
# ──────────────────────────────────────────────


class BGEEmbedder:
    """
    BGE-M3 embedder (BAAI/bge-m3). Dense 1024-dim vectors.

    BGE-M3 also produces sparse lexical weights but those are not used here —
    BM25 is handled separately by the Retriever via bm25s.
    """

    MODEL = MODELS["bge-m3"].hf_model
    _ENCODE_OPTS = dict(
        return_dense=True, return_sparse=False, return_colbert_vecs=False
    )

    def __init__(
        self, device: str = "cpu", use_fp16: bool = True, batch_size: int = 32
    ):
        from FlagEmbedding import BGEM3FlagModel

        devices = [device] if device else None
        logger.info(f"Loading BGEEmbedder ({self.MODEL}) on {device}")
        self._model = BGEM3FlagModel(self.MODEL, use_fp16=use_fp16, devices=devices)
        self._batch_size = batch_size

    def encode_passages(self, chunks: list[dict]) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (n, 1024), dtype float32
        """
        texts = [c["text"] for c in chunks]
        output = self._model.encode(
            texts, batch_size=self._batch_size, max_length=512, **self._ENCODE_OPTS
        )
        logger.info(f"BGEEmbedder: encoded {len(chunks)} passages")
        return np.array(output["dense_vecs"], dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Returns float32 vector of dim 1024."""
        output = self._model.encode([query], **self._ENCODE_OPTS)
        return np.array(output["dense_vecs"][0], dtype=np.float32)


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────


def get_embedder(model_key: str, device: str = "cpu"):
    """Return the appropriate embedder instance for the given model key.

    Args:
        model_key: one of the keys in MODELS registry (e.g. "bge-m3", "e5-small").
        device: torch device string (e.g. "cpu", "cuda", "mps").

    Returns:
        An embedder instance with ``encode_passages()`` and ``encode_query()`` methods.

    Raises:
        ValueError: if model_key is not in the registry.
    """
    if model_key not in MODELS:
        valid = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model '{model_key}'. Valid: {valid}")

    if model_key == "bge-m3":
        return BGEEmbedder(device=device)
    if model_key in ("e5-small", "e5-large"):
        return E5Embedder(model_key=model_key, device=device)
    if model_key == "pplx":
        return PplxEmbedder(device=device)

    raise ValueError(f"No embedder class registered for '{model_key}'")
