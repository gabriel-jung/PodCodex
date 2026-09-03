FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md CHANGELOG.md ./
COPY src/ src/

RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    ".[bot,rag]"

# One data dir for the container: the index, the machine identity, the
# per-guild bot state and the model cache all hang off it, and compose mounts
# it as the podcodex_data volume. Pinning it here also makes the build-time
# download below land in the exact directory the bot reads at runtime; the
# build used to write to HF's default ~/.cache, which nothing ever read.
ENV PODCODEX_DATA_DIR=/root/.local/share/podcodex

# Pre-download BGE-M3 into that cache so the first search after a fresh
# `docker compose up` does not wait on a ~2.5 GB fetch. get_hf_cache_dir() is
# the same facility the embedder calls at runtime, so build and run cannot
# disagree on the path; it has to run before sentence_transformers imports
# huggingface_hub, which reads the cache env vars into module constants.
RUN python -c "from podcodex.core.cache import get_hf_cache_dir; get_hf_cache_dir(); from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

ENTRYPOINT ["podcodex-bot"]
