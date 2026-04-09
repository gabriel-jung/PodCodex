"""Shared helpers for API route modules."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi import HTTPException
from loguru import logger

from pydantic import BaseModel, field_validator

from podcodex.api.schemas import TaskResponse
from podcodex.core._utils import UNKNOWN_SPEAKERS
from podcodex.ingest.rss import RSSEpisode, episode_stem

# ── Shared constants ────────────────────────────

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".opus", ".wma"}


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


def llm_prov_params(mode: str, provider: str | None = None, **extra: object) -> dict:
    """Build the LLM portion of provenance params."""
    d: dict = {"llm_mode": mode}
    if provider:
        d["llm_provider"] = provider
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


# ── Path helpers ────────────────────────────────


def require_show_folder(show_folder: str) -> Path:
    """Resolve a show folder path, raising 404 if it doesn't exist."""
    path = Path(show_folder)
    if not path.is_dir():
        raise HTTPException(404, f"Show folder not found: {show_folder}")
    return path


def is_downloaded(show_folder: Path, stem: str) -> bool:
    """Check if an audio file with the given stem exists in the show folder."""
    return any((show_folder / f"{stem}{ext}").exists() for ext in AUDIO_EXTS)


def rss_episode_to_out(ep: RSSEpisode, show_folder: Path) -> dict:
    """Convert an RSSEpisode to an RSSEpisodeOut dict."""
    stem = episode_stem(ep)
    return {
        **asdict(ep),
        "local_stem": stem,
        "downloaded": is_downloaded(show_folder, stem),
    }


# ── Task submission ─────────────────────────────


def submit_task(step: str, audio_path: str, fn, *args) -> TaskResponse:
    """Submit a background task.

    If a task is already running on this audio_path, return its task_id
    instead of raising an error — lets the UI reconnect after navigation.
    """
    from podcodex.api.tasks import task_manager
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


# Extend the core set with empty string (relevant for flagging UI segments).
_UNKNOWN_SPEAKERS = UNKNOWN_SPEAKERS | {""}


def read_segments(path: Path) -> list[dict] | None:
    """Read segments from a JSON file.

    Handles both formats:
    - Plain array: [{speaker, text, start, end}, ...]
    - Wrapped: {meta: {...}, segments: [...]}
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read segments from {}: {}", path, exc)
        return None

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    return None


def is_flagged(seg: dict) -> bool:
    """Determine whether a segment should be flagged for review."""
    speaker = seg.get("speaker", "")
    if speaker == "[BREAK]":
        return False
    if speaker in _UNKNOWN_SPEAKERS:
        return True
    if speaker == "[remove]":
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
    corrected → transcript.  Raises ValueError if nothing found.
    """
    from podcodex.core._utils import normalize_lang
    from podcodex.core.transcribe import load_transcript
    from podcodex.core.versions import load_latest

    if source == "auto":
        segs = load_latest(p.base, "corrected")
        if segs:
            return segs, "corrected"
        segs = load_latest(p.base, "transcript")
        if segs:
            return segs, "transcript"
        # Legacy transcript file fallback
        segs = load_transcript(str(p.audio_path))
        if segs:
            return segs, "transcript"
        raise ValueError("No transcript found — transcribe first")

    if source == "transcript":
        segs = load_latest(p.base, "transcript")
        if segs:
            return segs, "transcript"
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

    return transcript


# ── Shared request models ──────────────────────


class LLMRequest(BaseModel):
    """Base request for LLM pipeline steps (correct & translate)."""

    audio_path: str
    output_dir: str | None = None
    mode: str = "ollama"
    provider: str | None = None
    model: str = ""
    context: str = ""
    source_lang: str = "French"
    batch_minutes: float = 15.0
    api_base_url: str = ""
    api_key: str | None = None

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
    source_lang: str = "French"
    target_lang: str = "English"
    batch_minutes: float = 15.0
    source_version_id: str | None = None


class ApplyManualRequest(BaseModel):
    """Request for applying manual LLM corrections (shared by correct & translate)."""

    audio_path: str | None = None
    output_dir: str | None = None
    corrections: list[dict]
    lang: str = ""


def format_prompt_batches(batches: list) -> list[dict]:
    """Format build_manual_prompts_batched output into API response dicts."""
    return [
        {"batch_index": i, "prompt": prompt, "segment_count": len(batch_segs)}
        for i, (batch_segs, prompt) in enumerate(batches)
    ]
