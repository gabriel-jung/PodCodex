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

import threading
import time

import numpy as np
from loguru import logger

from podcodex.core.cache import load_cached
from podcodex.rag.defaults import MODELS


# ──────────────────────────────────────────────
# PplxEmbedder
# ──────────────────────────────────────────────


class PplxEmbedder:
    """
    Context-aware embedder using Perplexity's pplx-embed-context-v1 family.

    Passages are encoded with full episode context: all chunks from the same
    episode are passed together so each embedding is influenced by its neighbors.
    Query encoding uses the separate pplx-embed-v1 query model.

    Dense vectors (dim from MODELS registry), unnormalized: the vector search
    uses the cosine metric, which ignores magnitude.

    Both models load on first use: a search only encodes queries and an
    index job only encodes passages, and holding both doubled the memory of
    either (4B each for pplx-4B).
    """

    def __init__(self, model_key: str = "pplx-0.6B", device: str = "cpu"):
        """Initialize the Perplexity embedder; nothing is loaded yet.

        Args:
            model_key: Registry key (``"pplx-0.6B"`` or ``"pplx-4B"``).
            device: Torch device string (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
        """
        from podcodex.core.cache import get_hf_cache_dir

        self._spec = MODELS[model_key]
        self._device = device
        self._cache_dir = str(get_hf_cache_dir())
        self._loaded: dict[str, object] = {}
        self._load_lock = threading.Lock()
        self._dim = self._spec.dim

    def _load(self, which: str):
        """Load the ``"ctx"`` (passage) or ``"query"`` model once."""
        model = self._loaded.get(which)
        if model is not None:
            return model
        with self._load_lock:
            model = self._loaded.get(which)
            if model is not None:
                return model
            from podcodex.core._hf_logging import timed_load

            spec = self._spec
            if which == "ctx":
                from transformers import AutoModel

                with timed_load(f"PplxEmbedder ctx {spec.hf_model} on {self._device}"):
                    model = load_cached(
                        lambda source: AutoModel.from_pretrained(
                            source,
                            revision=spec.revision,
                            trust_remote_code=True,
                            cache_dir=self._cache_dir,
                        ),
                        spec.hf_model,
                        cache_dir=self._cache_dir,
                        revision=spec.revision,
                    ).to(self._device)
            else:
                from sentence_transformers import SentenceTransformer

                with timed_load(
                    f"PplxEmbedder query {spec.hf_query_model} on {self._device}"
                ):
                    model = load_cached(
                        lambda source: SentenceTransformer(
                            source,
                            revision=spec.query_revision,
                            trust_remote_code=True,
                            device=self._device,
                            cache_folder=self._cache_dir,
                        ),
                        spec.hf_query_model,
                        cache_dir=self._cache_dir,
                        revision=spec.query_revision,
                    )
            self._loaded[which] = model
            return model

    @property
    def _ctx_model(self):
        return self._load("ctx")

    @property
    def _query_model(self):
        return self._load("query")

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
            logger.info(
                f"PplxEmbedder: encoding {len(texts)} passages "
                f"with full-episode context for '{episode}' (single forward pass)"
            )
            t0 = time.perf_counter()
            episode_embs = self._ctx_model.encode([texts])[0]  # (n_chunks, dim)
            logger.info(
                f"PplxEmbedder: forward pass for '{episode}' done in "
                f"{time.perf_counter() - t0:.1f}s"
            )
            for pos, idx in enumerate(indices):
                result[idx] = episode_embs[pos].astype(np.float32)

        logger.info(f"PplxEmbedder: encoded {len(chunks)} passages")
        return result

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string using the separate query model.

        Args:
            query: Natural language query text.

        Returns:
            Float32 vector of shape ``(dim,)``.
        """
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
        """Initialize the E5 embedder.

        Args:
            model_key: Registry key (``"e5-small"`` or ``"e5-large"``).
            device: Torch device string.
            batch_size: Encoding batch size.
        """
        from sentence_transformers import SentenceTransformer

        hf_model = MODELS[model_key].hf_model
        from podcodex.core._hf_logging import timed_load
        from podcodex.core.cache import (
            cached_snapshot,
            get_hf_cache_dir,
            get_sentence_transformers_dir,
        )

        cache_dir = str(get_hf_cache_dir())
        # Up to 0.2.x the semantic chunker loaded e5-small by name, so
        # sentence-transformers cached it under SENTENCE_TRANSFORMERS_HOME.
        # Reuse that copy rather than download the model again here.
        legacy = get_sentence_transformers_dir()
        if cached_snapshot(hf_model, cache_dir=cache_dir) is None and cached_snapshot(
            hf_model, cache_dir=legacy
        ):
            cache_dir = str(legacy)
        with timed_load(f"E5Embedder {hf_model} on {device}"):
            self._model = load_cached(
                lambda source: SentenceTransformer(
                    source, device=device, cache_folder=cache_dir
                ),
                hf_model,
                cache_dir=cache_dir,
            )
        self._batch_size = batch_size

    def encode_passages(self, chunks: list[dict]) -> np.ndarray:
        """Encode passage chunks with the ``"passage: "`` prefix.

        Args:
            chunks: List of chunk dicts, each containing a ``"text"`` key.

        Returns:
            L2-normalized float32 array of shape ``(n, dim)``.
        """
        texts = ["passage: " + c["text"] for c in chunks]
        t0 = time.perf_counter()
        embs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        logger.info(
            f"E5Embedder: encoded {len(chunks)} passages "
            f"(batch_size={self._batch_size}) in {time.perf_counter() - t0:.1f}s"
        )
        return np.array(embs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string with the ``"query: "`` prefix.

        Args:
            query: Natural language query text.

        Returns:
            L2-normalized float32 vector of shape ``(dim,)``.
        """
        emb = self._model.encode("query: " + query, normalize_embeddings=True)
        return np.array(emb, dtype=np.float32).squeeze()


# ──────────────────────────────────────────────
# BGEEmbedder
# ──────────────────────────────────────────────


class BGEEmbedder:
    """
    BGE-M3 embedder (BAAI/bge-m3). Dense 1024-dim vectors.

    BGE-M3 also produces sparse lexical weights but those are not used here —
    keyword search is handled separately by LanceDB's FTS index.
    """

    MODEL = MODELS["bge-m3"].hf_model
    _ENCODE_OPTS = dict(
        return_dense=True, return_sparse=False, return_colbert_vecs=False
    )

    def __init__(
        self, device: str = "cpu", use_fp16: bool | None = None, batch_size: int = 32
    ):
        """Initialize the BGE-M3 embedder.

        Args:
            device: Torch device string.
            use_fp16: Half precision. Default: on a CUDA device whose
                ``core.device.torch_dtype`` is not float32, so Pascal (where
                FP16 matmul is unusably slow) and CPU run in full precision.
            batch_size: Encoding batch size.
        """
        if use_fp16 is None:
            use_fp16 = _half_precision(device)
        from FlagEmbedding import BGEM3FlagModel
        from podcodex.core._hf_logging import timed_load
        from podcodex.core.cache import get_hf_hub_dir

        devices = [device] if device else None
        # Explicit cache: holds even where no bootstrap wired the env (a
        # script, a test), and matches the hub dir the Dockerfile's
        # build-time download fills.
        cache_dir = str(get_hf_hub_dir())
        with timed_load(f"BGEEmbedder {self.MODEL} on {device}"):
            self._model = load_cached(
                lambda source: BGEM3FlagModel(
                    source, use_fp16=use_fp16, devices=devices, cache_dir=cache_dir
                ),
                self.MODEL,
                cache_dir=cache_dir,
            )
        self._batch_size = batch_size

    def encode_passages(self, chunks: list[dict]) -> np.ndarray:
        """Encode passage chunks using BGE-M3 dense encoding.

        Args:
            chunks: List of chunk dicts, each containing a ``"text"`` key.

        Returns:
            Float32 array of shape ``(n, 1024)``.
        """
        texts = [c["text"] for c in chunks]
        t0 = time.perf_counter()
        output = self._model.encode(
            texts, batch_size=self._batch_size, max_length=512, **self._ENCODE_OPTS
        )
        logger.info(
            f"BGEEmbedder: encoded {len(chunks)} passages "
            f"(batch_size={self._batch_size}) in {time.perf_counter() - t0:.1f}s"
        )
        return np.array(output["dense_vecs"], dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Args:
            query: Natural language query text.

        Returns:
            Float32 vector of shape ``(1024,)``.
        """
        output = self._model.encode([query], **self._ENCODE_OPTS)
        return np.array(output["dense_vecs"][0], dtype=np.float32)


def _half_precision(device: str) -> bool:
    """Whether a model on *device* should run in half precision."""
    if not (device or "").startswith("cuda"):
        return False
    import torch

    from podcodex.core.device import torch_dtype

    return torch_dtype() != torch.float32


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────


# Embedders, plus the chunker's sentence model (get_chunker_embeddings).
_embedder_cache: dict[str, object] = {}
_embedder_lock = threading.Lock()


def clear_embedder_cache() -> None:
    """Evict all cached embedders to free memory (e.g. before heavy pipeline tasks)."""
    with _embedder_lock:
        if _embedder_cache:
            logger.info("Clearing embedder cache ({} model(s))", len(_embedder_cache))
        _embedder_cache.clear()


def get_chunker_embeddings(device: str):
    """The semantic chunker's sentence model, wrapped for chonkie.

    The chunker's model is E5 Small (``defaults.CHUNKER_MODEL_KEY``), the same
    weights as that embedder, so it reuses the cached embedder's model: one
    copy in memory when E5 Small also indexes, freed by
    ``clear_embedder_cache`` like every other model.
    """
    from chonkie import SentenceTransformerEmbeddings

    from podcodex.rag.defaults import CHUNKER_MODEL_KEY

    key = f"chunker|{device}"
    cached = _embedder_cache.get(key)
    if cached is not None:
        return cached
    e5 = get_embedder(CHUNKER_MODEL_KEY, device=device)  # takes the lock itself
    with _embedder_lock:
        cached = _embedder_cache.get(key)
        if cached is None:
            cached = SentenceTransformerEmbeddings(e5._model)
            _embedder_cache[key] = cached
        return cached


def get_embedder(
    model_key: str, device: str = "cpu"
) -> BGEEmbedder | E5Embedder | PplxEmbedder:
    """Return a cached embedder instance for the given model key.

    The embedder is created on first call and reused on subsequent calls
    with the same model_key (the heavy model load only happens once).

    Args:
        model_key: one of the keys in MODELS registry (e.g. "bge-m3", "e5-small").
        device: torch device string (e.g. "cpu", "cuda", "mps").

    Returns:
        An embedder instance with ``encode_passages()`` and ``encode_query()`` methods.

    Raises:
        ValueError: if model_key is not in the registry.
    """
    # Cache key includes device — same model on cpu vs cuda are different
    # objects with different parameter tensors; collapsing them silently
    # gave the wrong-device embedder to the second caller.
    cache_key = f"{model_key}|{device}"

    # Single .get() avoids the contains/index race against clear_embedder_cache
    cached = _embedder_cache.get(cache_key)
    if cached is not None:
        return cached

    with _embedder_lock:
        cached = _embedder_cache.get(cache_key)
        if cached is not None:
            return cached

        if model_key not in MODELS:
            valid = ", ".join(MODELS.keys())
            raise ValueError(f"Unknown model '{model_key}'. Valid: {valid}")

        if model_key == "bge-m3":
            embedder = BGEEmbedder(device=device)
        elif model_key in ("e5-small", "e5-large"):
            embedder = E5Embedder(model_key=model_key, device=device)
        elif model_key in ("pplx-0.6B", "pplx-4B"):
            embedder = PplxEmbedder(model_key=model_key, device=device)
        else:
            raise ValueError(f"No embedder class registered for '{model_key}'")

        _embedder_cache[cache_key] = embedder
        return embedder
