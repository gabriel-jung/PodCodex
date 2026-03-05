"""
podcodex.rag.embedder — Embedding models for RAG strategies.

Three embedders covering the four indexing strategies:

    PplxEmbedder — perplexity-ai/pplx-embed-context-v1-0.6B (passages, context-aware)
                   + pplx-embed-v1-0.6B (queries). Dense 1024-dim.
    E5Embedder   — intfloat/multilingual-e5-small. Dense 384-dim, normalized.
    BGEEmbedder  — BAAI/bge-m3. Dense 1024-dim + sparse (Qdrant native hybrid).

All embedders expose:
    encode_passages(chunks: list[dict]) -> embeddings
    encode_query(query: str) -> embedding

For BGE, embeddings are dicts:
    {"dense": np.ndarray, "sparse": [{"indices": [...], "values": [...]}]}
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from loguru import logger


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _to_qdrant_sparse(lexical_weights: dict) -> dict:
    """Convert BGE-M3 {token_id_str: weight} to Qdrant sparse vector format."""
    indices = [int(k) for k in lexical_weights]
    values = [float(v) for v in lexical_weights.values()]
    return {"indices": indices, "values": values}


# ──────────────────────────────────────────────
# PplxEmbedder
# ──────────────────────────────────────────────


class PplxEmbedder:
    """
    Context-aware embedder using Perplexity's pplx-embed-context-v1-0.6B.

    Passages are encoded with full episode context: all chunks from the same
    episode are passed together so each embedding is influenced by its neighbors.
    Query encoding uses the separate pplx-embed-v1-0.6B model.

    Output: dense float32 vectors of dim 1024.
    """

    PASSAGE_MODEL = "perplexity-ai/pplx-embed-context-v1-0.6B"
    QUERY_MODEL = "perplexity-ai/pplx-embed-v1-0.6B"

    def __init__(self, device: str = "cpu"):
        from transformers import AutoModel
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading PplxEmbedder (passage={self.PASSAGE_MODEL}) on {device}")
        self._ctx_model = AutoModel.from_pretrained(
            self.PASSAGE_MODEL, trust_remote_code=True
        )
        self._query_model = SentenceTransformer(
            self.QUERY_MODEL, trust_remote_code=True
        )

    def encode_passages(self, chunks: list[dict]) -> np.ndarray:
        """
        Encode passages with full episode context.

        Chunks are grouped by episode and encoded together so each embedding
        benefits from surrounding context within its episode.
        Output order matches input order.

        Returns:
            np.ndarray of shape (n, 1024), dtype float32
        """
        # Map each episode to the original indices of its chunks
        episode_indices: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            episode_indices[chunk.get("episode", "")].append(i)

        result = np.empty((len(chunks), 1024), dtype=np.float32)
        for episode, indices in episode_indices.items():
            texts = [chunks[i]["text"] for i in indices]
            # encode() takes a list-of-episodes; each episode is a list of texts
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
    Multilingual E5 embedder (intfloat/multilingual-e5-small).

    Uses "passage: " / "query: " prefixes as required by the model.
    Output: normalized dense float32 vectors of dim 384.
    """

    MODEL = "intfloat/multilingual-e5-small"

    def __init__(self, device: str = "cpu", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading E5Embedder ({self.MODEL}) on {device}")
        self._model = SentenceTransformer(self.MODEL, device=device)
        self._batch_size = batch_size

    def encode_passages(self, chunks: list[dict]) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (n, 384), dtype float32, L2-normalized
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
        """Returns normalized float32 vector of dim 384."""
        emb = self._model.encode("query: " + query, normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)


# ──────────────────────────────────────────────
# BGEEmbedder
# ──────────────────────────────────────────────


class BGEEmbedder:
    """
    BGE-M3 embedder (BAAI/bge-m3) with dense + sparse output.

    Returns dense vectors (1024-dim) alongside sparse lexical weights
    suitable for Qdrant's native hybrid search.
    Uses FlagEmbedding's BGEM3FlagModel.
    """

    MODEL = "BAAI/bge-m3"

    def __init__(self, use_fp16: bool = True, batch_size: int = 32):
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading BGEEmbedder (BAAI/bge-m3)")
        self._model = BGEM3FlagModel(self.MODEL, use_fp16=use_fp16)
        self._batch_size = batch_size

    def encode_passages(self, chunks: list[dict]) -> dict:
        """
        Returns:
            {
                "dense":  np.ndarray of shape (n, 1024), dtype float32
                "sparse": list of {"indices": [int], "values": [float]}
            }
        """
        texts = [c["text"] for c in chunks]
        output = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.array(output["dense_vecs"], dtype=np.float32)
        sparse = [_to_qdrant_sparse(lw) for lw in output["lexical_weights"]]
        logger.info(f"BGEEmbedder: encoded {len(chunks)} passages")
        return {"dense": dense, "sparse": sparse}

    def encode_query(self, query: str) -> dict:
        """
        Returns:
            {
                "dense":  np.ndarray of shape (1024,), dtype float32
                "sparse": {"indices": [int], "values": [float]}
            }
        """
        output = self._model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.array(output["dense_vecs"][0], dtype=np.float32)
        sparse = _to_qdrant_sparse(output["lexical_weights"][0])
        return {"dense": dense, "sparse": sparse}
