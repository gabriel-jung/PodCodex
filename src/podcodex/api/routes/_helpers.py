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


def build_provenance(
    step: str,
    ptype: str = "raw",
    model: str | None = None,
    params: dict | None = None,
    manual_edit: bool = False,
) -> dict:
    """Build a standard provenance dict for version tracking."""
    return {
        "step": step,
        "type": ptype,
        "model": model,
        "params": params or {},
        "manual_edit": manual_edit,
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
    polished → transcript.  Raises ValueError if nothing found.
    """
    from podcodex.core._utils import normalize_lang
    from podcodex.core.transcribe import load_transcript
    from podcodex.core.versions import load_latest

    if source == "auto":
        segs = load_latest(p.base, "polished")
        if segs:
            return segs, "polished"
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

    if source == "polished":
        segs = load_latest(p.base, "polished")
        if segs:
            return segs, "polished"
        raise ValueError("No polished segments found")

    # Language code
    lang_norm = normalize_lang(source)
    segs = load_latest(p.base, lang_norm)
    if segs:
        return segs, lang_norm
    raise ValueError(f"No translation found for '{source}'")


def load_best_source(audio_path: str, output_dir: str | None = None) -> list[dict]:
    """Load the best available source segments (polished → transcript fallback).

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
    Otherwise resolves from the version DB (polished > transcript fallback).
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
    """Base request for LLM pipeline steps (polish & translate)."""

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
    """Request for generating manual LLM prompts (shared by polish & translate)."""

    audio_path: str
    output_dir: str | None = None
    context: str = ""
    source_lang: str = "French"
    target_lang: str = "English"
    batch_minutes: float = 15.0
    engine: str = "Whisper"


class ApplyManualRequest(BaseModel):
    """Request for applying manual LLM corrections (shared by polish & translate)."""

    audio_path: str
    output_dir: str | None = None
    corrections: list[dict]
    lang: str = ""
