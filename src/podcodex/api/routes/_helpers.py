"""Shared helpers for API route modules."""

from __future__ import annotations

import re
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
from podcodex.core.constants import AUDIO_EXTENSIONS
from podcodex.ingest.rss import RSSEpisode, episode_stem
from podcodex.rag.index_store import get_index_store  # re-export

__all__ = ["get_index_store"]

# Single source of truth — keeping this aligned with the scanner's set
# avoids "is_downloaded says yes, scanner says no" mismatches that hid
# yt-dlp output behind missing-ffmpeg failures.
AUDIO_EXTS = AUDIO_EXTENSIONS


def list_show_stems(show_folder: Path) -> frozenset[str]:
    """One-shot listing of stems on disk in a show folder.

    Pass into :func:`episode_stem` / :func:`rss_episode_to_out` from any
    loop that processes many episodes — without it, each call inside the
    loop would do its own ``os.scandir``. Frozen so episode_stem's suffix
    lookup can memoize an index keyed on it.
    """
    import os

    stems: set[str] = set()
    try:
        with os.scandir(show_folder) as it:
            for entry in it:
                name = entry.name
                if entry.is_dir(follow_symlinks=False):
                    if not name.startswith("."):
                        stems.add(name)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                dot = name.rfind(".")
                if dot > 0 and name[dot:].lower() in AUDIO_EXTENSIONS:
                    stems.add(name[:dot])
    except OSError:
        pass
    return frozenset(stems)


def _build_source_chain(
    audio_path: str | None,
    output_dir: str | None,
    step: str,
    model: str | None,
    mode: str | None,
) -> list[str] | None:
    """Build a source chain by looking up the input version's chain and appending this step.

    Returns e.g. ["youtube-subtitles", "ollama/qwen3:4b", "openai/gpt-4"].
    """
    try:
        from podcodex.core._utils import AudioPaths
        from podcodex.core.versions import get_latest_provenance

        p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

        # Find the input version — walk backwards through the pipeline
        input_prov = None
        if step == "corrected":
            input_prov = get_latest_provenance(p.base, "transcript")
        else:
            # Translate and others: try corrected first, then transcript
            input_prov = get_latest_provenance(
                p.base, "corrected"
            ) or get_latest_provenance(p.base, "transcript")

        # Get existing chain or start from the input's source
        prev_chain: list[str] = []
        if input_prov:
            input_params = input_prov.get("params") or {}
            prev_chain = list(input_params.get("source_chain", []))
            if not prev_chain:
                # Legacy: build chain from source field
                source = input_params.get("source")
                if source:
                    prev_chain = [source]

        # Append this step's identifier
        step_id = model or mode or step
        return prev_chain + [step_id] if prev_chain else None
    except Exception:
        logger.opt(exception=True).debug("source chain build failed for {}", audio_path)
        return None


def transcribe_prov_params(
    diarize: bool, source: str = "whisper", model: str | None = None, **extra: object
) -> dict:
    """Build provenance params for a transcribe step.

    Also builds a source_chain entry like ``"whisper/large-v3-turbo, diarized"``.
    """
    d: dict = {"diarize": diarize, "source": source}
    # Build a descriptive source chain entry for downstream steps
    label = f"{source}/{model}" if model else source
    if diarize:
        label += ", diarized"
    d["source_chain"] = [label]
    d.update(extra)
    return d


def llm_prov_params(
    mode: str,
    provider_profile: str | None = None,
    key_name: str | None = None,
    **extra: object,
) -> dict:
    """Build the LLM portion of provenance params."""
    d: dict = {"llm_mode": mode}
    if provider_profile:
        d["llm_provider_profile"] = provider_profile
    if key_name:
        d["llm_key_name"] = key_name
    d.update(extra)
    return d


def build_provenance(
    step: str,
    ptype: str = "raw",
    model: str | None = None,
    params: dict | None = None,
    manual_edit: bool = False,
    audio_path: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Build a standard provenance dict for version tracking.

    When *audio_path* or *output_dir* is provided and the step is not
    ``transcript``, a ``source_chain`` is built by looking up the input
    version's chain and appending this step's model/mode identifier.
    """
    params = dict(params) if params else {}
    # A hand-edited version is "validated" by definition, and the two flags
    # must agree: `is_edited` reads either, but only the type reaches the
    # filename, so a manual edit typed "raw" is indistinguishable from model
    # output once the DB is rebuilt from disk. Enforced here rather than at
    # each caller, which is how /translate/save-manual drifted.
    if manual_edit:
        ptype = "validated"
    if (
        step != "transcript"
        and "source_chain" not in params
        and (audio_path or output_dir)
    ):
        chain = _build_source_chain(
            audio_path, output_dir, step, model, params.get("llm_mode")
        )
        if chain:
            params["source_chain"] = chain
    return {
        "step": step,
        "type": ptype,
        "model": model,
        "params": params,
        "manual_edit": manual_edit,
    }


def build_edit_provenance(
    step: str,
    audio_path: str | None,
    output_dir: str | None,
) -> dict:
    """Build provenance for a manual edit by inheriting from the latest version of the same step.

    Edited versions keep the same model/params/source_chain as their parent
    so their label reflects the pipeline that produced them, just marked as
    ``type=validated`` + ``manual_edit=True``.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.core.versions import get_latest_provenance

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    parent = get_latest_provenance(p.base, step) or {}
    return {
        "step": step,
        "type": "validated",
        "model": parent.get("model"),
        "params": dict(parent.get("params") or {}),
        "manual_edit": True,
    }


def enrich_correct_kwargs(
    audio_path: str | None,
    output_dir: str | None,
    fallback_source_lang: str,
) -> dict:
    """Look up transcript provenance and return kwargs for correct_segments.

    Returns dict with ``source_lang``, ``engine``, ``engine_model``.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.core.correct import transcript_provenance_info
    from podcodex.core.versions import get_latest_provenance

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    tc_prov = get_latest_provenance(p.base, "transcript")
    tc_info = transcript_provenance_info(tc_prov)
    return {
        "source_lang": tc_info["language"] or fallback_source_lang,
        "engine": tc_info["source"],
        "engine_model": tc_info["model"],
    }


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


def is_downloaded(show_folder: Path, stem: str) -> bool:
    """Check if an audio file with the given stem exists in the show folder."""
    return any((show_folder / f"{stem}{ext}").exists() for ext in AUDIO_EXTS)


def rss_episode_to_out(
    ep: RSSEpisode,
    show_folder: Path,
    *,
    existing_stems: frozenset[str] | None = None,
) -> dict:
    """Convert an RSSEpisode to an RSSEpisodeOut dict.

    Loop callers should pre-compute ``existing_stems`` once via
    ``_list_show_stems`` (or equivalent) and pass it in to avoid an
    ``os.scandir`` per episode inside ``episode_stem``.
    """
    stem = episode_stem(ep, show_folder, existing_stems=existing_stems)
    return {
        **asdict(ep),
        "local_stem": stem,
        "downloaded": is_downloaded(show_folder, stem),
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


def _resolve_source_segments(p, source: str) -> tuple[list[dict], str]:
    """Resolve source segments from the version DB.

    Returns (segments, source_label).  Priority for 'auto':
    verified pointer → corrected → transcript.  Raises ValueError if
    nothing found.
    """
    from podcodex.core._utils import normalize_lang
    from podcodex.core.transcribe import load_transcript
    from podcodex.core.versions import (
        load_latest,
        load_version,
        resolve_verified_source,
    )

    if source == "auto":
        # Verified pointer wins when present; downstream consumers honor the
        # user's canonical pick over the freshest output.
        verified = resolve_verified_source(p.base)
        if verified is not None:
            v_step, v_id, _ = verified
            try:
                segs = load_version(p.base, v_step, v_id)
                if segs:
                    return segs, v_step
            except Exception:
                pass  # stale pointer; reconcile pass clears it asynchronously
        segs = load_latest(p.base, "corrected")
        if segs:
            return segs, "corrected"
        segs = load_transcript(str(p.audio_path))
        if segs:
            return segs, "transcript"
        raise ValueError("No transcript found — transcribe first")

    if source == "transcript":
        segs = load_transcript(str(p.audio_path))
        if segs:
            return segs, "transcript"
        raise ValueError("No transcript found — transcribe first")

    if source == "corrected":
        segs = load_latest(p.base, "corrected")
        if segs:
            return segs, "corrected"
        raise ValueError("No corrected segments found")

    # Language code
    lang_norm = normalize_lang(source)
    segs = load_latest(p.base, lang_norm)
    if segs:
        return segs, lang_norm
    raise ValueError(f"No translation found for '{source}'")


def load_best_source(
    audio_path: str | None = None, output_dir: str | None = None
) -> list[dict]:
    """Load the best available source segments (corrected → transcript fallback).

    Raises ValueError if no source segments are found.
    """
    from podcodex.core._utils import AudioPaths

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    segments, _ = _resolve_source_segments(p, "auto")
    return segments


def build_index_transcript(
    audio_path: str,
    show_name: str,
    stem: str,
    segments: list[dict] | None = None,
    source: str = "auto",
    output_dir: str | None = None,
) -> dict:
    """Build the transcript dict expected by vectorize_batch.

    If *segments* are provided directly (e.g. from version DB), wraps them.
    Otherwise resolves from the version DB (corrected > transcript fallback).
    Injects RSS metadata (title, pub_date, episode_number) when available.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.ingest.rss import load_episode_meta

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    if segments is None:
        segments, source = _resolve_source_segments(p, source)

    transcript: dict = {
        "meta": {"show": show_name, "episode": stem, "source": source},
        "segments": segments,
    }

    # Inject RSS metadata
    ep_meta = load_episode_meta(p.base.parent)
    if ep_meta:
        if ep_meta.title:
            transcript["meta"].setdefault("rss_title", ep_meta.title)
        if ep_meta.pub_date:
            transcript["meta"].setdefault("rss_pub_date", ep_meta.pub_date)
        if ep_meta.episode_number is not None:
            transcript["meta"].setdefault("episode_number", ep_meta.episode_number)
        if ep_meta.description:
            transcript["meta"].setdefault("rss_description", ep_meta.description)
        # Media pointers for the Discord bot (index-only): episode artwork, the
        # RSS enclosure to link, and the explicit YouTube video id so the bot
        # can build a timestamped watch link.
        if ep_meta.artwork_url:
            transcript["meta"].setdefault("rss_artwork_url", ep_meta.artwork_url)
        if ep_meta.audio_url:
            transcript["meta"].setdefault("rss_audio_url", ep_meta.audio_url)
        if ep_meta.youtube_id:
            transcript["meta"].setdefault("youtube_id", ep_meta.youtube_id)

    # Broadcast (airing) number: extracted from the episode title using the
    # show's configured regex, when set. Distinct from the per-season
    # episode_number. Absent for shows with no pattern.
    bnum = _extract_broadcast_number(p.show_dir, ep_meta.title if ep_meta else "")
    if bnum is not None:
        transcript["meta"].setdefault("broadcast_number", bnum)

    return transcript


def apply_broadcast_pattern(pattern: str, title: str) -> int | None:
    """Apply *pattern* to *title*, returning the first capture group as an int.

    Returns ``None`` when the pattern or title is empty, the pattern has no
    capture group, the pattern does not match, or the captured group is not an
    integer. Raises ``re.error`` when the pattern itself is invalid, even for
    an empty title, so callers can always surface a bad regex (e.g. the live
    preview must not silently accept one on a show with no titled episode).
    """
    if not pattern:
        return None
    compiled = re.compile(pattern)  # raises re.error on a bad pattern
    if not title:
        return None
    m = compiled.search(title)
    if not m or not m.lastindex:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_broadcast_number(show_dir: Path, title: str) -> int | None:
    """Apply the show's ``broadcast_number_pattern`` to *title*, if configured.

    Returns the first captured group as an int, or ``None`` when the show has
    no pattern, the title is empty, or the pattern does not match. Invalid
    patterns are swallowed (indexing must never crash on a bad regex).
    """
    if not title:
        return None
    try:
        from podcodex.ingest.show import load_show_meta

        meta = load_show_meta(show_dir)
    except Exception:
        return None
    pattern = meta.broadcast_number_pattern if meta else ""
    try:
        return apply_broadcast_pattern(pattern, title)
    except re.error:
        return None


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
