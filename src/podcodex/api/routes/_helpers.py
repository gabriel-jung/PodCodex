"""Shared helpers for API route modules."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi import HTTPException
from loguru import logger

from pydantic import BaseModel

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


def load_segments_or_404(path: Path, label: str = "segments") -> list[dict]:
    """Load and annotate segments from a path, raising 404 if not found."""
    data = read_segments(path)
    if data is None:
        raise HTTPException(404, f"No {label} found")
    return annotate_flags(data)


def load_best_source(audio_path: str, output_dir: str | None = None) -> list[dict]:
    """Load the best available source segments (polished → transcript fallback).

    Raises ValueError if no source segments are found.
    """
    from podcodex.core.polish import load_polished
    from podcodex.core.transcribe import load_transcript

    try:
        segments = load_polished(audio_path, output_dir=output_dir)
    except (FileNotFoundError, ValueError):
        segments = None
    if not segments:
        segments = load_transcript(audio_path, output_dir=output_dir)
    if not segments:
        raise ValueError("No source segments found (need transcript or polished)")
    return segments


# ── Index transcript builder ───────────────────


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
    Otherwise resolves from filesystem (polished > transcript fallback).
    Injects RSS metadata (title, pub_date, episode_number) when available.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.ingest.rss import load_episode_meta

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    if segments is not None:
        transcript: dict = {
            "meta": {"show": show_name, "episode": stem, "source": "version"},
            "segments": segments,
        }
    else:
        from podcodex.cli import _resolve_source, _source_label

        transcript_path = p.transcript_best
        if not transcript_path.exists():
            raise ValueError("No transcript found — transcribe first")

        source_path = _resolve_source(transcript_path, source)
        source_label = _source_label(source_path, transcript_path)
        data = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            transcript = {
                "meta": {"show": show_name, "episode": stem, "source": source_label},
                "segments": data,
            }
        else:
            transcript = data
            transcript.setdefault("meta", {})
            transcript["meta"].setdefault("show", show_name)
            transcript["meta"].setdefault("episode", stem)
            transcript["meta"].setdefault("source", source_label)

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
