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

from collections import defaultdict

import numpy as np
from loguru import logger

from podcodex.rag.defaults import (
    BGE_M3_MODEL,
    E5_LARGE_MODEL,
    E5_SMALL_MODEL,
    PPLX_PASSAGE_MODEL,
    PPLX_QUERY_MODEL,
)


# ──────────────────────────────────────────────
# PplxEmbedder  (experimental placeholder)
# ──────────────────────────────────────────────


class PplxEmbedder:
    """
    Context-aware embedder using Perplexity's pplx-embed-context-v1-0.6B.

    Passages are encoded with full episode context: all chunks from the same
    episode are passed together so each embedding is influenced by its neighbors.
    Query encoding uses the separate pplx-embed-v1-0.6B query model.

    Dense 1024-dim vectors.
    """

    PASSAGE_MODEL = PPLX_PASSAGE_MODEL
    QUERY_MODEL = PPLX_QUERY_MODEL

    def __init__(self, device: str = "cpu"):
        from transformers import AutoModel
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading PplxEmbedder ({self.PASSAGE_MODEL}) on {device}")
        self._ctx_model = AutoModel.from_pretrained(
            self.PASSAGE_MODEL, trust_remote_code=True
        )
        self._query_model = SentenceTransformer(
            self.QUERY_MODEL, trust_remote_code=True
        )

    def encode_passages(self, chunks: list[dict]) -> np.ndarray:
        """
        Encode passages with full episode context (grouped by episode).

        Returns:
            np.ndarray of shape (n, 1024), dtype float32
        """
        episode_indices: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            episode_indices[chunk.get("episode", "")].append(i)

        result = np.empty((len(chunks), 1024), dtype=np.float32)
        for episode, indices in episode_indices.items():
            texts = [chunks[i]["text"] for i in indices]
            episode_embs = self._ctx_model.encode([texts])[0]  # (n_chunks, 1024)
            for pos, idx in enumerate(indices):
                result[idx] = episode_embs[pos].astype(np.float32)

        logger.info(f"PplxEmbedder: encoded {len(chunks)} passages")
        return result

    def encode_query(self, query: str) -> np.ndarray:
        """Returns float32 vector of dim 1024."""
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

    _HF_MODELS: dict[str, str] = {
        "e5-small": E5_SMALL_MODEL,
        "e5-large": E5_LARGE_MODEL,
    }

    def __init__(
        self,
        model_key: str = "e5-small",
        device: str = "cpu",
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        hf_model = self._HF_MODELS.get(model_key, E5_SMALL_MODEL)
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

    MODEL = BGE_M3_MODEL

    def __init__(self, use_fp16: bool = True, batch_size: int = 32):
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading BGEEmbedder (BAAI/bge-m3)")
        self._model = BGEM3FlagModel(self.MODEL, use_fp16=use_fp16)
        self._batch_size = batch_size

    def encode_passages(self, chunks: list[dict]) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (n, 1024), dtype float32
        """
        texts = [c["text"] for c in chunks]
        output = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        logger.info(f"BGEEmbedder: encoded {len(chunks)} passages")
        return np.array(output["dense_vecs"], dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Returns float32 vector of dim 1024."""
        output = self._model.encode(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.array(output["dense_vecs"][0], dtype=np.float32)


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────


def get_embedder(model_key: str):
    """Return the appropriate embedder instance for the given model key."""
    from podcodex.rag.defaults import MODELS

    if model_key not in MODELS:
        valid = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model '{model_key}'. Valid: {valid}")

    if model_key == "bge-m3":
        return BGEEmbedder()
    if model_key in ("e5-small", "e5-large"):
        return E5Embedder(model_key=model_key)
    if model_key == "pplx":
        return PplxEmbedder()

    raise ValueError(f"No embedder class registered for '{model_key}'")
