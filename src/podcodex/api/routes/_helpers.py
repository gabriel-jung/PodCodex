"""Shared helpers for API route modules."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import HTTPException
from loguru import logger

from pydantic import BaseModel, field_validator

from podcodex.api.schemas import TaskResponse
from podcodex.core._utils import bad_path_component  # noqa: F401
from podcodex.core._utils import (
    BREAK_SPEAKER,
    REMOVE_SPEAKER,
    UNKNOWN_SPEAKERS,
    AudioPaths,
    _separate_breaks,
)
from podcodex.ingest.rss import RSSEpisode, episode_stem

# Domain helpers now live in core/ so that core and rag never import the API
# package to reach them (see the module docstrings there). Re-exported here
# because every route module imports them from this one.
from podcodex.core.provenance import (  # noqa: F401
    _build_source_chain,
    build_edit_provenance,
    build_provenance,
    enrich_correct_kwargs,
    llm_prov_params,
    transcribe_prov_params,
)
from podcodex.core.source import (  # noqa: F401
    AUDIO_EXTS,
    _extract_broadcast_number,
    _resolve_source_segments,
    apply_broadcast_pattern,
    build_index_transcript,
    is_downloaded,
    list_show_stems,
    load_best_source,
    scan_show_stems,
)

__all__ = ["get_index_store"]


def get_index_store():
    """Lazy re-export of ``podcodex.rag.index_store.get_index_store``.

    A wrapper, not a plain re-export, because importing ``index_store``
    pulls pyarrow and numpy (~150 ms) and this module is the first thing
    the API's route package loads. A ``from _helpers import
    get_index_store`` at another module's top level would resolve a plain
    re-export — or a module ``__getattr__`` — at import time and pay it
    anyway; a wrapper defers to the first actual call, which only happens
    inside a request handler.

    The store is opened at startup regardless, by the lifespan's warmup
    thread (``app._warmup_caches_sync``), so no request waits on this.
    """
    from podcodex.rag.index_store import get_index_store as _get_index_store

    return _get_index_store()


def batch_progress(progress_cb, start: float = 0.1, end: float = 0.9):
    """Return a callback for reporting batch progress to the task manager."""

    def on_batch(batch_num: int, total: int) -> None:
        """Report progress for a single completed batch."""
        frac = start + (end - start) * (batch_num / total)
        progress_cb(frac, f"Batch {batch_num} of {total}")

    return on_batch


def counted_progress(progress_cb, total: int):
    """Return a `(index, message="", *, frac=None)` reporter that emits the
    canonical ``[i+1/total] message`` format consumed by the frontend's
    `parseProgressCount` regex. Standardizes the prefix so individual routes
    can't drift from the contract (a missing bracket silently kills the
    ``1/N`` counter in the TaskBar progress strip).

    Defaults the fraction to ``index / total`` (ticks while the item is in
    flight); pass ``frac`` to override (e.g. ``(i + 1) / total`` for "done"
    ticks, or any custom interpolation).
    """

    def report(index: int, message: str = "", *, frac: float | None = None) -> None:
        body = f" {message}" if message else ""
        progress_cb(
            index / total if frac is None else frac,
            f"[{index + 1}/{total}]{body}",
        )

    return report


# ── Path helpers ────────────────────────────────


def require_show_folder(show_folder: str) -> Path:
    """Resolve a show folder path, raising 404 if it doesn't exist."""
    path = Path(show_folder)
    if not path.is_dir():
        raise HTTPException(404, f"Show folder not found: {show_folder}")
    return path


def require_registered_show(show_folder: str) -> Path:
    """Resolve a show folder that must be a *registered* show.

    ``require_show_folder`` only checks the path is a directory, which is
    fine for read routes but dangerous for destructive ones: it would let a
    caller point ``delete``/``move`` at any directory on disk. This gate adds
    the missing check that the folder is actually tracked in the app config,
    confining ``rmtree``/``move`` to real shows.
    """
    path = require_show_folder(show_folder)
    from podcodex.api.routes.config import _load as _load_cfg

    # samefile() compares device+inode, so it is correct across case-insensitive
    # filesystems (macOS/Windows) and symlinks, where a resolved-string compare
    # would false-negative and 403 a legitimately registered show.
    for folder in _load_cfg().show_folders:
        try:
            if path.samefile(Path(folder)):
                return path
        except OSError:
            continue  # a registered root that no longer exists on disk
    raise HTTPException(403, "Not a registered show folder")


def require_audio_or_output(audio_path: str | None, output_dir: str | None) -> None:
    """Raise 422 unless at least one of ``audio_path`` / ``output_dir`` is set.

    Most pipeline routes accept either: an audio file (real episode) or just
    an output dir (e.g. YouTube subtitle imports without audio). Backend's
    ``AudioPaths.from_audio`` already raises ``ValueError`` when both are
    None, but it surfaces as a 500 — this gives a proper 422 to clients.
    """
    if not audio_path and not output_dir:
        raise HTTPException(status_code=422, detail="audio_path or output_dir required")


def resolve_inside_show_root(path: str) -> Path:
    """Defend ``?path=`` query params against arbitrary-file read/delete by
    requiring the resolved path to live under a registered show folder."""
    from podcodex.api.routes.config import _load as _load_cfg

    p = Path(path).expanduser().resolve()
    cfg = _load_cfg()
    roots = [Path(f).resolve() for f in cfg.show_folders]
    if not any(p == r or p.is_relative_to(r) for r in roots):
        raise HTTPException(403, "Path is not inside a registered show folder")
    return p


def rss_episode_to_out(
    ep: RSSEpisode,
    show_folder: Path,
    *,
    existing_stems: frozenset[str] | None = None,
    audio_stems: frozenset[str] | None = None,
) -> dict:
    """Convert an RSSEpisode to an RSSEpisodeOut dict.

    Loop callers should take both sets from :func:`scan_show_stems` once and
    pass them in: without ``existing_stems`` every call does its own
    ``os.scandir`` inside ``episode_stem``, and without ``audio_stems`` the
    downloaded flag relists and stats the entire show folder per episode.
    """
    stem = episode_stem(ep, show_folder, existing_stems=existing_stems)
    downloaded = (
        stem in audio_stems
        if audio_stems is not None
        else is_downloaded(show_folder, stem)
    )
    return {
        **asdict(ep),
        "local_stem": stem,
        "downloaded": downloaded,
    }


# ── Task submission ─────────────────────────────


_GPU_STEPS = frozenset({"transcribe", "index", "batch", "generate_tts"})


def submit_task(step: str, audio_path: str, fn, *args) -> TaskResponse:
    """Submit a background task.

    If a task is already running on this audio_path, return its task_id
    instead of raising an error — lets the UI reconnect after navigation.
    """
    from podcodex.api.tasks import task_manager

    if step in _GPU_STEPS:
        from podcodex.rag.embedder import clear_embedder_cache

        clear_embedder_cache()
    try:
        info = task_manager.submit(step, audio_path, fn, *args)
    except ValueError:
        # Return existing running task so the UI can reconnect
        existing = task_manager.get_active(audio_path)
        if existing:
            return TaskResponse(task_id=existing.task_id)
        raise HTTPException(409, "A task is already running on this file") from None
    return TaskResponse(task_id=info.task_id)


def submit_subprocess_task(
    step: str,
    audio_path: str,
    entry_path: str,
    kwargs: dict,
    req,
    on_result=None,
) -> TaskResponse:
    """Submit a background task whose work runs in a spawned subprocess.

    Centralises the boilerplate that would otherwise be copy-pasted in every
    route handler that delegates to ``subprocess_runner``: builds the inner
    closure, extracts the cancel_event attached by the task manager, and
    forwards the progress callback. ``on_result`` runs in the server process
    after a successful subprocess exit with the child's result dict — for
    cache upkeep that needs to happen where the caches live; its failure
    must not fail the task.
    """
    from podcodex.api.subprocess_runner import run_in_subprocess

    def _run(progress_cb, _req):
        result = run_in_subprocess(
            entry_path=entry_path,
            kwargs=kwargs,
            on_progress=progress_cb,
            on_log=getattr(progress_cb, "log_cb", None),
            cancel_event=getattr(progress_cb, "cancel_event", None),
        )
        if on_result is not None:
            try:
                on_result(result)
            except Exception:
                logger.opt(exception=True).warning(
                    "on_result hook failed for {} task", step
                )
        return result

    return submit_task(step, audio_path, _run, req)


def is_flagged(seg: dict) -> bool:
    """Determine whether a segment should be flagged for review."""
    speaker = seg.get("speaker", "")
    if speaker == BREAK_SPEAKER:
        return False
    if speaker in UNKNOWN_SPEAKERS:
        return True
    if speaker == REMOVE_SPEAKER:
        return True
    # Low speech density: < 2 chars/s
    dur = seg.get("end", 0) - seg.get("start", 0)
    if dur > 0 and len(seg.get("text", "")) / dur < 2:
        return True
    return False


def annotate_flags(segments: list[dict]) -> list[dict]:
    """Add a ``flagged`` field to each segment."""
    for seg in segments:
        seg["flagged"] = is_flagged(seg)
    return segments


# ── Shared request models ──────────────────────


class LLMRequest(BaseModel):
    """Base request for LLM pipeline steps (correct & translate).

    The frontend sends a profile name + key name; the route resolves
    them via ``llm_resolver.resolve_llm`` before invoking core. Ollama
    profiles need no key.
    """

    audio_path: str
    output_dir: str | None = None
    mode: str = "ollama"
    provider_profile: str | None = None
    key_name: str | None = None
    model: str = ""
    context: str = ""
    source_lang: str = "English"
    batch_minutes: float = 15.0
    source_version_id: str | None = None

    @field_validator("batch_minutes")
    @classmethod
    def batch_minutes_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("batch_minutes must be positive")
        return v


class ManualPromptsRequest(BaseModel):
    """Request for generating manual LLM prompts (shared by correct & translate)."""

    audio_path: str | None = None
    output_dir: str | None = None
    context: str = ""
    source_lang: str = "English"
    target_lang: str = "French"
    batch_minutes: float = 15.0
    # When set, overrides batch_minutes and produces exactly this many
    # batches. The frontend sends this whenever episode.duration is known.
    batch_count: int | None = None
    source_version_id: str | None = None


class ApplyManualRequest(BaseModel):
    """Request for applying manual LLM corrections (shared by correct & translate)."""

    audio_path: str | None = None
    output_dir: str | None = None
    corrections: list[dict]
    lang: str = ""
    # Pin the same source the prompts were built from (ManualPromptsRequest
    # carries it too); validating against a different default-pick is exactly
    # the case where the entry counts disagree.
    source_version_id: str | None = None


class BatchFix(BaseModel):
    """One hand-reconciled batch: `batch` selects the recorded batch in
    ``llm_failures.json``, `corrections` is the reconciled response in batch
    order, one entry per input segment."""

    batch: int
    corrections: list[dict]


class ApplyBatchesRequest(BaseModel):
    """Request for applying hand-reconciled batches from a failed auto run.

    All fixes are patched into one new version. Shared by correct & translate.
    """

    audio_path: str | None = None
    output_dir: str | None = None
    fixes: list[BatchFix]
    lang: str = ""


def reconcile_batches(
    req: ApplyBatchesRequest, step: str
) -> tuple[AudioPaths, list[dict], dict]:
    """Patch every fix's batch into the latest version of *step*.

    Looks each batch up in ``llm_failures.json``, checks its correction count,
    loads the latest version, and applies all fixes in one pass. Returns
    ``(paths, patched_segments, failures_section)``; raises HTTPException on a
    missing episode, missing batch, count mismatch, or missing version.
    """
    from podcodex.core._utils import apply_corrections
    from podcodex.core.llm_failures import get_step
    from podcodex.core.versions import load_latest

    require_audio_or_output(req.audio_path, req.output_dir)
    p = AudioPaths.from_audio(req.audio_path, output_dir=req.output_dir)

    section = get_step(req.audio_path, req.output_dir, step)
    if not section:
        raise HTTPException(404, "No recorded batch failures for this episode")
    if not req.fixes:
        raise HTTPException(400, "No fixes provided")

    records = {b.get("batch"): b for b in section.get("batches", [])}
    # Flattened across all fixes — batches never share a segment index.
    by_index: dict[int, dict] = {}
    for fix in req.fixes:
        record = records.get(fix.batch)
        if record is None:
            raise HTTPException(404, f"Batch {fix.batch} not found")
        indices = [s["index"] for s in record.get("input", [])]
        if len(fix.corrections) != len(indices):
            raise HTTPException(
                400,
                f"Batch {fix.batch} expects {len(indices)} entries, "
                f"got {len(fix.corrections)}",
            )
        for i, idx in enumerate(indices):
            by_index[idx] = fix.corrections[i]

    segments = load_latest(p.base, step)
    if segments is None:
        raise HTTPException(404, "No segments found for this step")

    patched = apply_corrections(segments, by_index, min_length_ratio=0)
    return p, patched, section


def format_prompt_batches(batches: list) -> list[dict]:
    """Format build_manual_prompts_batched output into API response dicts.

    ``segment_count`` is the real (non-[BREAK]) segment count, matching the
    prompt's "Output MUST contain exactly N entries" line and the apply-path
    count check (validate_manual). Counting [BREAK] markers here would make
    the per-batch validation reject a correct LLM response by the number of
    breaks in the batch.
    """
    return [
        {
            "batch_index": i,
            "prompt": prompt,
            "segment_count": len(_separate_breaks(batch_segs)[1]),
        }
        for i, (batch_segs, prompt) in enumerate(batches)
    ]


def resolve_collection_for_show(
    show: str, model: str, chunking: str, store=None
) -> str | None:
    """Collection for a show given its display name, or None when not indexed.

    The API receives display names (from the frontend, MCP, and the bot), so
    this is the one place that goes label -> id -> collection. Nothing
    reconstructs a collection name from a show name any more: that is what a
    rename used to orphan.

    Args:
        store: The store to query. Pass the one the caller already resolved,
            so a route that swaps its store (tests do) is not bypassed by a
            second lookup in here.
    """
    from podcodex.ingest.show_registry import show_id_for_label

    store = store if store is not None else get_index_store()
    return store.resolve_collection(
        show_id_for_label(show), model, chunking, show_label=show
    )


def collections_for_show_name(show: str, store=None) -> list[str]:
    """Every collection of the show with this display name.

    Companion to ``resolve_collection_for_show`` for callers that want all of
    a show's collections rather than one specific combination.

    An empty *show* means "no filter" and returns every collection, matching
    the ``list_collections(show="")`` behaviour these call sites had before
    identity moved off the display name.

    Args:
        store: The store to query; see ``resolve_collection_for_show``.
    """
    from podcodex.ingest.show_registry import show_id_for_label

    store = store if store is not None else get_index_store()
    if not (show or "").strip():
        return store.list_collections()
    return store.collections_for_show(show_id_for_label(show), show_label=show)
