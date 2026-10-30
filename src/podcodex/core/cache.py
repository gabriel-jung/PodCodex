"""
podcodex.core.cache — Model cache management.

Centralises all ML model downloads into a PodCodex-controlled directory
instead of scattering across ~/.cache/huggingface/.

Resolution order:
  1. ``PODCODEX_CACHE_DIR`` env var (explicit override)
  2. ``app_paths.data_dir() / "models"`` (the one definition of the data
     dir: ``PODCODEX_DATA_DIR`` when the Tauri shell sets it, otherwise
     the platform app-data directory)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from podcodex.core.app_paths import data_dir


def get_cache_dir() -> Path:
    """Return (and create) the PodCodex model cache directory.

    Everything but the explicit override goes through
    ``app_paths.data_dir()``, so the models tree sits next to the index
    and the logs no matter who started the process. The bundled sidecar
    inherits ``PODCODEX_DATA_DIR`` from the Tauri shell; the bot, the
    MCP server and dev checkouts get the platform app-data dir instead
    of a separate ``~/.podcodex``. A second definition of the data dir
    here is what used to send the bot's HF cache to a path on no compose
    volume, so BGE-M3 was re-downloaded on every container recreate.
    """
    explicit = os.environ.get("PODCODEX_CACHE_DIR", "").strip()
    path = Path(explicit) if explicit else data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_hf_cache_dir() -> Path:
    """Return (and create) the HuggingFace root inside the PodCodex cache.

    ``HF_HOME`` points here; see :func:`wire_model_caches` for the layout.
    WhisperX, E5 and Pplx have always cached here (flat ``models--*``
    layout) and keep doing so, so existing downloads stay valid; pyannote and
    BGE-M3 use :func:`get_hf_hub_dir`. The model list and delete read both.
    """
    hf_dir = get_cache_dir() / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    return hf_dir


def get_sentence_transformers_dir() -> Path:
    """``SENTENCE_TRANSFORMERS_HOME``: set by the user, else under the model cache.

    Where sentence-transformers caches a model loaded by name, which is how the
    semantic chunker fetched e5-small up to 0.2.x.
    """
    override = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "").strip()
    return Path(override) if override else get_cache_dir() / "sentence-transformers"


def get_hf_hub_dir() -> Path:
    """The hub snapshot store (``HF_HUB_CACHE``), for loaders taking a cache dir.

    Passing it explicitly holds even in a process whose huggingface_hub was
    imported before the env vars were set (a script, a test).
    """
    return get_hf_cache_dir() / "hub"


def cached_snapshot(
    repo_id: str, *, cache_dir: Path | str | None = None, revision: str | None = None
) -> str | None:
    """Local folder of *repo_id*'s cached snapshot, or ``None`` if not cached.

    For loaders that accept a path. Given a repo id they ask the Hub for the
    latest revision first: with no network that request is retried with
    backoff (about 12s for BGE-M3), and a refused connection fails the load
    outright instead of falling back to the cache. Loading from the folder
    skips the Hub, and keeps an index's embedding model from changing under
    it when the repo is updated. *cache_dir* is the one the loader itself is
    given (default: the hub store), since loaders disagree on it.
    """
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id,
            revision=revision,
            cache_dir=str(cache_dir or get_hf_hub_dir()),
            local_files_only=True,
        )
    except Exception:
        return None


def _snapshot_incomplete(local: str, exc: Exception) -> bool:
    """Whether a failed load from *local* looks like a partial download.

    Transformers reports a missing file as ``OSError``; otherwise the sign is
    a download cut short, whose unfinished blobs stay as ``*.incomplete``
    beside the snapshots.
    """
    if isinstance(exc, OSError):
        return True
    snapshot = Path(local)
    if len(snapshot.parents) < 2:
        return False
    return any((snapshot.parents[1] / "blobs").glob("*.incomplete"))


def _out_of_memory(exc: Exception) -> bool:
    """torch, ctranslate2 and the allocator all say so in the message."""
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()


def load_cached(
    load,
    repo_id: str,
    *,
    cache_dir: Path | str | None = None,
    revision: str | None = None,
):
    """``load(source)`` from the cached snapshot of *repo_id* when there is one.

    See `cached_snapshot`: given the repo id, the loaders ask the Hub first,
    which offline stalls on retries or fails outright. When the cached load
    fails, it is retried by repo id so the Hub can complete a partial
    download: loaders report a missing file every which way (``OSError``
    from transformers, ``RuntimeError`` from ctranslate2, ``ValueError``
    from whisperx's alignment loader), so any failure is retried except
    running out of memory, which a second load would only repeat while the
    first one's memory is still held. If the retry fails too, the error
    raised is the one that names the problem: the Hub's (gated model,
    rejected token, unreachable) when the snapshot looked incomplete, the
    cached load's own otherwise.
    """
    from loguru import logger

    local = cached_snapshot(repo_id, cache_dir=cache_dir, revision=revision)
    if local is None:
        return load(repo_id)
    try:
        return load(local)
    except Exception as exc:
        if _out_of_memory(exc):
            raise
        # The traceback's frames hold whatever the failed load built; log it
        # and let it go, so the retry doesn't load beside it.
        logger.opt(exception=exc).debug(f"Cached {repo_id} load failed")
        first = exc.with_traceback(None)
    logger.warning(
        f"Cached {repo_id} snapshot did not load ({first}), retrying from the Hub"
    )
    try:
        return load(repo_id)
    except Exception:
        if _snapshot_incomplete(local, first):
            raise
        raise first from None


def wire_model_caches() -> None:
    """Point every ML cache env var at the PodCodex model cache.

    The single setter for the cache layout, run first by every
    ``bootstrap_for_*()`` so the vars are in place before torch,
    transformers or huggingface_hub is imported (they read the vars once, at
    import). Every process shares one tree, dev and bot included, so the
    in-app model list and delete see every download. Values already in the
    environment win: the Tauri shell presets the hub and torch vars.

    ``HF_HOME`` is ``<hf>``; ``HF_HUB_CACHE`` and ``TRANSFORMERS_CACHE`` are
    both ``<hf>/hub``. ``transformers.utils.hub.cached_file`` and qwen_tts's
    ``from_pretrained`` look up via ``TRANSFORMERS_CACHE`` and fall back to
    ``HF_HOME/transformers`` (a stub dir distinct from where
    ``snapshot_download`` writes), so the hub and transformers vars must
    agree or the loader and downloader halves split-brain. The symptom is
    ``OSError: Can't load feature extractor for <path>/preprocessor_config.json``;
    any snapshot left under ``<hf>/transformers/`` is dead weight. Models
    re-downloading or missing from the in-app list mean something imported
    huggingface_hub before this ran, so the weights went to
    ``~/.cache/huggingface``.
    """
    models_dir = get_cache_dir()
    hub = str(get_hf_hub_dir())
    os.environ.setdefault("HF_HOME", str(get_hf_cache_dir()))
    os.environ.setdefault("HF_HUB_CACHE", hub)
    os.environ.setdefault("TRANSFORMERS_CACHE", hub)
    os.environ.setdefault("TORCH_HOME", str(models_dir / "torch"))
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME", str(get_sentence_transformers_dir())
    )

    # Cap HF Hub network calls so a flaky uplink (VPN, captive portal,
    # huggingface.co outage) can't stall a cached-model load past 10s. The
    # default urllib3 read-timeout is unset, so a half-broken TCP can hang
    # the whole pipeline at startup.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")

    # Hard offline opt-in: when the user knows every model is cached, this
    # bypasses the etag round-trip entirely. Maps to both HF Hub and the
    # transformers-side flag because the two libraries gate on different vars.
    if os.environ.get("PODCODEX_HF_OFFLINE", "").strip() in {"1", "true", "yes"}:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _model_cache_parents() -> list[Path]:
    """Every directory a loader may put a ``models--<org>--<name>`` tree in.

    - ``<hf_root>/hub``: ``HF_HUB_CACHE``, where pyannote, Qwen3-TTS and
      ``snapshot_download`` write
    - ``<hf_root>``: where a loader given ``cache_dir=<hf_root>`` writes
      (faster-whisper, the E5 embedder)
    - ``SENTENCE_TRANSFORMERS_HOME`` (`get_sentence_transformers_dir`),
      where the semantic chunker's e5-small went up to 0.2.x

    A model fetched through two of them has a copy in each. A
    ``SENTENCE_TRANSFORMERS_HOME`` set outside the model cache is left out:
    it may be shared with other tools, and this list feeds a delete.
    """
    hf_root = get_hf_cache_dir()
    parents = [hf_root / "hub", hf_root]
    st_home = get_sentence_transformers_dir()
    if st_home.resolve().is_relative_to(get_cache_dir().resolve()):
        parents.append(st_home)
    # Resolved once each: an override pointing at <hf> or <hf>/hub would
    # otherwise count every model in it twice.
    unique: dict[Path, Path] = {}
    for p in parents:
        unique.setdefault(p.resolve(), p)
    return list(unique.values())


def _disk_usage(root: Path) -> int:
    """Bytes *root* takes on disk, each file counted once.

    Symlinks are skipped and hard links deduplicated by inode, so a POSIX hub
    cache (snapshots link to blobs) counts its blobs once, while on Windows
    without symlink rights, where every snapshot file is a full copy of its
    blob, both copies count: they both take space.
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    for f in root.rglob("*"):
        try:
            if f.is_symlink() or not f.is_file():
                continue
            st = f.stat()
        except OSError:
            continue
        ident = (st.st_dev, st.st_ino)
        if st.st_ino and ident in seen:
            continue
        seen.add(ident)
        total += st.st_size
    return total


def list_cached_models() -> list[dict]:
    """List cached models with the disk space they take.

    One entry per model, its copies in every layout of
    `_model_cache_parents` summed; ``path`` is the largest copy, so an empty
    stub left by one loader does not hide the copy another one downloaded.
    """
    by_name: dict[str, dict] = {}
    for parent in _model_cache_parents():
        if not parent.is_dir():
            continue
        for entry in sorted(parent.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            size = _disk_usage(entry)
            cur = by_name.get(entry.name)
            if cur is None:
                by_name[entry.name] = cur = {
                    "id": entry.name,
                    "name": entry.name.replace("models--", "").replace("--", "/"),
                    "size_bytes": 0,
                    "path": str(entry),
                    "_largest": -1,
                }
            cur["size_bytes"] += size
            if size > cur["_largest"]:
                cur["_largest"] = size
                cur["path"] = str(entry)
    models = []
    for m in by_name.values():
        del m["_largest"]
        m["size_mb"] = round(m["size_bytes"] / (1024 * 1024), 1)
        models.append(m)
    return models


def delete_cached_model(model_id: str) -> bool:
    """Delete a model from the cache by its directory name. Returns True if deleted.

    *model_id* comes from a request, and ``parent / ".."`` passed the old
    ``target.parent == parent`` check, which would have removed the whole
    HuggingFace cache. Only a single ``models--*`` component is accepted.
    """
    from podcodex.core._utils import bad_path_component

    if bad_path_component(model_id) or not model_id.startswith("models--"):
        return False
    deleted = False
    # Every layout: a model fetched once through each kind of loader has a
    # copy in each, and removing one left the others on disk.
    for parent in _model_cache_parents():
        target = parent / model_id
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
            deleted = True
    return deleted


def get_vram_status() -> dict | None:
    """GPU VRAM info for the Settings widget, or None without CUDA.

    Runs in the API process, so it reads device properties and this
    process's allocator counters only: probing free memory would create a
    CUDA context and hold VRAM for the server's lifetime. ``free_mb`` is
    therefore total minus what this process reserved, not device-wide.
    """
    from podcodex.core.device import vram_total_bytes

    total = vram_total_bytes()
    if total is None:
        return None
    try:
        import torch

        used = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        return {
            "total_mb": round(total / (1024 * 1024)),
            "used_mb": round(used / (1024 * 1024)),
            "reserved_mb": round(reserved / (1024 * 1024)),
            "free_mb": round((total - reserved) / (1024 * 1024)),
            "device": torch.cuda.get_device_name(0),
        }
    except Exception:  # noqa: BLE001
        return None
