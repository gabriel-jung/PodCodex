"""
podcodex.core._utils — Shared utilities for the core pipeline.

Heavy libraries (torch, pandas) are imported lazily inside functions
so this module stays cheap to import at the top level.
"""

import gc
import json
import os
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Container
from typing import TYPE_CHECKING, Self

from loguru import logger

if TYPE_CHECKING:
    from podcodex.rag.hit import SpeakerTurn


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# Knobs for the schema-constrained correction call. See `run_ollama`.
OLLAMA_READ_TIMEOUT_S = 600.0
OLLAMA_CONNECT_TIMEOUT_S = 10.0
OLLAMA_KEEP_ALIVE = "10m"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT_MAX = 8192
OLLAMA_NUM_CTX_MAX = 16384


def ollama_host() -> str:
    """Return the Ollama daemon URL, honoring the ``OLLAMA_HOST`` env var.

    Centralizes host resolution so both the pipeline (``run_ollama``) and
    the API health check route resolve the same target.
    """
    return os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST


def list_pulled_ollama_models(host: str | None = None) -> list[str]:
    """Return sorted list of model tags pulled into the local Ollama daemon.

    Shared by the API health route and the pipeline's pre-flight check so
    both stay in lock-step on how the daemon's model list is fetched.
    """
    from ollama import Client

    resp = Client(host=host or ollama_host()).list()
    return sorted(m.model for m in resp.models if m.model)


def correction_schema(n_items: int) -> dict:
    """JSON Schema for one batch of N correction items.

    Used by ``run_ollama`` and the ``scripts/debug_ollama_output.py`` probe.
    Constrains output to a JSON array of exactly N objects, each with a
    single ``text`` field, which is what stops small models from emitting
    a wrapping object or looping garbage when given ``format=<schema>``.
    """
    return {
        "type": "array",
        "minItems": n_items,
        "maxItems": n_items,
        "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    }


def _ollama_token_budget(n_items: int) -> tuple[int, int]:
    """Pick ``num_predict`` and ``num_ctx`` for a batch of N correction items.

    Schema-constrained sampling can spin forever without a num_predict cap.
    Ollama's default ``num_ctx`` is 4096 regardless of the model card, so
    we round the (prompt + output) budget up to the next power of two and
    cap to keep KV cache within typical consumer VRAM.
    """
    num_predict = min(OLLAMA_NUM_PREDICT_MAX, n_items * 120 + 64)
    rough_budget = 800 + n_items * 140 + num_predict
    num_ctx = max(4096, 1 << max(0, rough_budget - 1).bit_length())
    return num_predict, min(num_ctx, OLLAMA_NUM_CTX_MAX)


def _warn_if_ollama_too_old(host: str) -> None:
    """Structured-output ``format=<schema>`` requires Ollama >= 0.5; older
    daemons silently ignore it and the model emits whatever shape it likes."""
    import httpx

    try:
        version = (
            httpx.get(f"{host}/api/version", timeout=5.0).json().get("version", "")
        )
        # Strip prerelease suffix (e.g. "0.5.0-rc1") before int-parsing.
        major, minor = (int(p.split("-")[0]) for p in version.split(".")[:2])
        if (major, minor) < (0, 5):
            logger.warning(
                f"Ollama daemon version {version} predates structured-output "
                "support (>=0.5). Schema-constrained decoding will be ignored."
            )
    except Exception as e:
        logger.debug(f"Could not read Ollama version: {e}")


def _warn_if_model_unpulled(client, model: str) -> None:
    """Pre-flight check so a typo'd model name fails before the first batch
    instead of mid-pipeline with a confusing 404."""
    try:
        pulled = {m.model for m in client.list().models if m.model}
        if model not in pulled:
            logger.warning(
                f"Model {model!r} not in pulled list {sorted(pulled)}; "
                "first chat call will likely 404."
            )
    except Exception as e:
        logger.debug(f"Could not list pulled models: {e}")


# ──────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────


VOICE_SAMPLES_DIR = "voice_samples"
TTS_SEGMENTS_DIR = "tts_segments"

# Fake filename for an episode that has an output dir but no audio on disk
# (subtitle-only YouTube imports). ``AudioPaths.from_audio`` resolves it to the
# real episode base because the fake name's stem *is* the episode stem, which
# is what lets such an episode be batched and locked like any other.
#
# Owner of the suffix on the Python side. The frontend mints the same string in
# ``frontend/src/lib/episodeRef.ts:getEpisodeBatchPath`` and the two must agree:
# it is the key space of ``BatchRequest.audio_paths``, of its
# ``source_version_ids`` map, and of ``task_manager``'s per-episode locks.
# ``tests/test_batch_version_picker.py`` pins the pair.
VIRTUAL_AUDIO_SUFFIX = ".virtual"


def virtual_audio_path(output_dir: Path | str) -> str:
    """The batch/lock key for an episode with an output dir but no audio."""
    return f"{str(output_dir).rstrip('/\\')}{VIRTUAL_AUDIO_SUFFIX}"


@dataclass
class AudioPaths:
    """All derived file paths for a given audio file.

    Centralises path logic for the entire pipeline (transcribe, correct,
    translate, synthesize).  Create via the ``from_audio`` classmethod::

        p = AudioPaths.from_audio("episode.mp3")
        p.voice_samples_dir # → …/episode/voice_samples/
        p.show_dir          # → …/{show}/
    """

    audio_path: Path  # resolved source audio file
    base: Path  # output_root / stem — no extension

    @staticmethod
    def output_dir(
        audio_path: str | Path, output_dir: str | Path | None = None
    ) -> Path:
        """Resolve the output directory for files related to a given episode.

        Args:
            audio_path : source audio file
            output_dir :
                None (default) — per-episode subfolder next to the audio:
                                 {audio.parent}/{audio.stem}/   (matches UI behaviour)
                ""             — flat: files land directly next to the audio file
                relative path  — resolved relative to audio_path.parent
                absolute path  — used as-is

        Returns:
            Resolved output directory Path (not yet created).
        """
        audio_path = Path(audio_path)
        if output_dir is None:
            return audio_path.parent / audio_path.stem
        p = Path(output_dir)
        if not str(output_dir):  # empty string
            return audio_path.parent
        return p if p.is_absolute() else audio_path.parent / p

    @classmethod
    def from_audio(
        cls,
        audio_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> Self:
        if audio_path:
            audio_path = Path(audio_path)
            root = cls.output_dir(audio_path, output_dir)
            base = root / audio_path.stem
        elif output_dir:
            root = Path(output_dir)
            base = root / root.name
        else:
            raise ValueError("Either audio_path or output_dir must be provided")
        base.parent.mkdir(parents=True, exist_ok=True)
        return cls(audio_path=audio_path or base, base=base)

    # — RAG —

    @property
    def show_dir(self) -> Path:
        """Show-level directory (parent of the episode output dir)."""
        return self.base.parent.parent

    # — Synthesis —

    @property
    def voice_samples_dir(self) -> Path:
        return self.base.parent / VOICE_SAMPLES_DIR

    @property
    def tts_segments_dir(self) -> Path:
        return self.base.parent / TTS_SEGMENTS_DIR

    def ensure_voice_samples_dir(self) -> Path:
        d = self.voice_samples_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ensure_tts_segments_dir(self) -> Path:
        d = self.tts_segments_dir
        d.mkdir(parents=True, exist_ok=True)
        return d


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────


# Speaker labels that don't represent a real person (unresolved diarization placeholders).
# Used by transcribe.py (filtering) and synthesize.py (voice sample extraction).
UNKNOWN_SPEAKERS = frozenset({"UNKNOWN", "UNK", "None", "none", ""})

# Placeholder speaker written when diarization is skipped. Deliberately not a
# plausible human name: it doubles as the voice-sample filename key and used to
# be "Narrator", which a documentary could legitimately call someone, making an
# identified speaker indistinguishable from "nobody identified anyone".
NARRATOR_SPEAKER = "NoDiarization"

# The value NARRATOR_SPEAKER had before 0.2.10. Still recognised as a
# placeholder so libraries written by older versions, and RAG indexes built
# from them, keep reading as unattributed without a reindex.
LEGACY_NARRATOR_SPEAKER = "Narrator"

# Segment inserted by merge_consecutive_segments when gap > max_gap.
BREAK_SPEAKER = "[BREAK]"


def is_unattributed(speaker: str | None, declared: Container[str] = ()) -> bool:
    """True when a speaker label names nobody.

    NARRATOR_SPEAKER is a storage placeholder, not an identification: it is
    what a transcript gets with diarization off, and it doubles as the voice
    sample key on disk (see synthesize.fill_narrator_speaker), which is why
    it stays in the data. Output boundaries should treat it exactly like the
    empty label and attribute nothing.

    Pass *declared* (a show's known speakers) wherever it is available: a
    documentary can legitimately have someone called "Narrator", and once the
    user has declared that name it is an identification like any other. The
    empty and UNKNOWN labels are never names, declared or not.
    """
    if not speaker or speaker in UNKNOWN_SPEAKERS:
        return True
    if speaker not in (NARRATOR_SPEAKER, LEGACY_NARRATOR_SPEAKER):
        return False
    # The legacy value is a plausible name, so a show that declares it means a
    # person. The current one cannot be a name, so declaring it changes nothing.
    return speaker == NARRATOR_SPEAKER or speaker not in declared


# ── mtime-based caching ──────────────────────────────────────────────────────

# A cached mtime younger than this is not trusted. Filesystems with coarse
# timestamps (FAT32 rounds to 2s, exFAT to 10ms) can land a write in the same
# tick as the stat that cached it, leaving the cached value looking current;
# external drives are a normal home for a podcast library.
MTIME_SETTLE_SECONDS = 2.0


def mtime_settled(mtime: float, now: float | None = None) -> bool:
    """True when *mtime* is old enough that a same-tick write can't hide.

    Gate *storing* a cache entry on this, not reading one: two writes inside
    the same coarse bucket share an mtime, so an entry recorded between them
    matches forever and hides the second.
    """
    import time as _time

    return (now if now is not None else _time.time()) - mtime >= MTIME_SETTLE_SECONDS


# Sentinel for segments the user marked for removal in the editor.
REMOVE_SPEAKER = "[remove]"

# Audio sample rate used by Whisper / TTS pipeline (16 kHz mono).
SAMPLE_RATE = 16000


def normalize_lang(lang: str) -> str:
    """Normalize a language name: lowercase, strip, collapse spaces to underscores.

    Used everywhere a language becomes a file-path component or version step name.
    """
    return lang.strip().lower().replace(" ", "_")


_ISO_TO_NAME: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "el": "Greek",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "uk": "Ukrainian",
    "ca": "Catalan",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "pl": "Polish",
}


def iso_to_language(code: str) -> str:
    """Convert an ISO 639-1 code to a language name. Returns the code as-is if unknown."""
    return _ISO_TO_NAME.get(code.lower().strip(), code)


# Default time-based thresholds shared across pipeline modules.
DEFAULT_MAX_GAP = 10.0
DEFAULT_BATCH_MINUTES = 15.0

# LLM temperature for deterministic output in correct / translate.
DEFAULT_TEMPERATURE = 0


# ──────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────


def read_parquet(path: Path) -> list[dict]:
    """Read a parquet file and return a list of dicts."""
    import pandas as pd

    return pd.read_parquet(path).to_dict("records")


def write_parquet(path: Path, records: list[dict]) -> None:
    """Write a list of dicts to a parquet file atomically."""
    import pandas as pd

    atomic_write(path, lambda p: pd.DataFrame(records).to_parquet(p, index=False))


def write_json(path: Path, data) -> None:
    """Write data as formatted JSON atomically."""
    write_json_atomic(path, data)


# ──────────────────────────────────────────────
# Episode title display
# ──────────────────────────────────────────────

_STEM_PREFIX_RE = re.compile(r"^\d+_(?:episode_\d+_)?", re.IGNORECASE)


def humanize_stem(stem: str) -> str:
    """Convert an episode file stem to a readable fallback title.

    Strips the numeric prefix used for sort stability (``"0027_"``, also
    matches ``"0027_episode_3_..."``), replaces underscores with spaces
    and capitalises the first letter. Used when an RSS title is not
    available in the chunk metadata.
    """
    s = _STEM_PREFIX_RE.sub("", stem).replace("_", " ").strip()
    return (s[:1].upper() + s[1:]) if s else stem


def resolve_episode_title(episode_title: str, stem: str) -> str:
    """Canonical episode-title resolution: RSS title, else humanized stem.

    The single owner of the fallback rule. ``episode_display`` (dict-shaped
    episode records) and ``Hit.display_title`` (typed search hits) both
    delegate here so every consumer cites the same title.
    """
    return episode_title or humanize_stem(stem)


def episode_display(chunk: dict) -> str:
    """Best human-readable episode title for a dict-shaped episode record."""
    return resolve_episode_title(
        chunk.get("episode_title") or "", chunk.get("episode", "")
    )


_PUB_DATE_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_PUB_DATE_COMPACT_RE = re.compile(r"^\d{8}$")


def normalize_pub_date(raw) -> str | None:
    """Normalize a publication date to ``YYYY-MM-DD``.

    Accepts ISO 8601 (``2024-01-15``, ``2024-01-15T12:00:00Z``), RFC 2822
    (``Mon, 15 Jan 2024 12:00:00 GMT``), and YouTube's compact
    ``YYYYMMDD``. Returns ``None`` if *raw* is falsy or unparseable.
    Idempotent on already-normalized input.
    """
    if not raw:
        return None
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.strip()
    if not s:
        return None
    if _PUB_DATE_ISO_RE.match(s):
        return s[:10]
    if _PUB_DATE_COMPACT_RE.match(s):
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    try:
        dt = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        dt = None
    if dt is not None:
        return dt.date().isoformat()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


_HMS_RE = re.compile(r"^(?:(\d+)h)?(\d+)m(\d{1,2})$")


def format_hms(seconds: float) -> str:
    """Format a timestamp for citation.

    ``< 3600 s`` gives ``"9m38"`` (minutes unpadded, seconds 2-digit).
    ``>= 3600 s`` gives ``"1h09m46"`` (minutes and seconds 2-digit within
    the hour). Negative inputs clamp to zero.

    Truncates (floors) rather than rounding: a start timestamp must never
    point past the passage it marks, and the wiki convention treats these
    strings as identifiers (strict-equality lint, cross-page citations), so
    a rounded ``,5+`` fraction would break the canonical floor form.
    """
    total = int(float(seconds))
    if total < 0:
        total = 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}"
    return f"{minutes}m{secs:02d}"


def parse_time(value: str | int | float) -> float:
    """Parse a timestamp into float seconds.

    Accepts raw seconds (``4186`` / ``4186.0`` / ``"4186"``) and the clock
    forms ``"1h09m46"`` / ``"69m46"`` / ``"9m38"``. Raises ``ValueError`` if
    a minutes-within-hour or seconds field is >= 60.
    """
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass
    m = _HMS_RE.match(text)
    if not m:
        raise ValueError(f"unrecognized time format: {value!r}")
    hours = int(m.group(1)) if m.group(1) else 0
    minutes = int(m.group(2))
    secs = int(m.group(3))
    if secs >= 60:
        raise ValueError(f"seconds field must be < 60: {value!r}")
    if hours and minutes >= 60:
        raise ValueError(f"minutes-within-hour field must be < 60: {value!r}")
    return float(hours * 3600 + minutes * 60 + secs)


def bad_path_component(name: str) -> bool:
    """True when *name* is unusable as a single path component: empty,
    traversal (".", ".."), or carrying a separator. Single facility for
    every folder/file-name safety check (API routes, bundle import)."""
    return not name or "/" in name or "\\" in name or name in {".", ".."}


def atomic_write(
    path: Path,
    writer_fn,
    *,
    prefix: str = ".tmp_",
    suffix: str = "",
) -> None:
    """Write to ``path`` atomically via same-dir temp file + ``os.replace``.

    ``writer_fn`` receives the temp Path and must fully write it. On any
    exception the temp file is removed so the destination is never a
    half-written or zero-filled stub visible to readers (cloud-sync
    clients, other processes).
    """
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=suffix)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer_fn(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_json_atomic(path: Path, data, *, prefix: str = ".tmp_") -> None:
    """Write ``data`` as formatted JSON atomically."""

    def _write(p: Path) -> None:
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    atomic_write(path, _write, prefix=prefix, suffix=".tmp")


def wav_duration(path: Path) -> float:
    """Return WAV duration in seconds, or 0.0 on error."""
    import soundfile as sf

    try:
        return sf.info(str(path)).duration
    except (OSError, RuntimeError):
        return 0.0


def default_batch_size() -> int:
    """Return 16 if total VRAM > 10 GB, else 8."""
    from podcodex.core.device import cuda_available

    try:
        if cuda_available():
            import torch

            _, total = torch.cuda.mem_get_info()
            if total > 10 * 1024 * 1024 * 1024:
                return 16
    except Exception:
        pass
    return 8


def free_vram() -> None:
    """Flush VRAM — call after ``del model`` in the caller's scope."""
    from podcodex.core.device import cuda_available

    gc.collect()
    if cuda_available():
        import torch

        torch.cuda.empty_cache()


def check_vram(label: str = "model", min_mb: int = 512) -> None:
    """Flush caches then raise if free VRAM is below *min_mb*.

    Call this on CUDA devices before loading a heavy model.  On CPU or
    when CUDA is unavailable, this is a no-op.
    """
    from podcodex.core.device import cuda_available

    if not cuda_available():
        return
    import torch

    # flush first so the reading is accurate
    gc.collect()
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_mb = free_bytes // (1024 * 1024)
    total_mb = total_bytes // (1024 * 1024)
    logger.info(f"VRAM before {label}: {free_mb} MB free / {total_mb} MB total")
    if free_mb < min_mb:
        raise RuntimeError(
            f"Not enough VRAM to load {label}: {free_mb} MB free, "
            f"need at least {min_mb} MB. "
            f"Try closing other GPU processes or restarting the backend."
        )


# ──────────────────────────────────────────────
# Segment helpers
# ──────────────────────────────────────────────


def tts_segment_filename(seg: dict) -> str:
    """Stable on-disk filename for a TTS segment's .wav.

    Keyed by start/end timestamps (millisecond integers) only — independent
    of the segment's position in the source list AND of the speaker label.
    That way the file survives:
      * narrowing the source picker (indices shift after filtering),
      * the narrator-speaker fallback (speakers can be remapped at synth
        time but timestamps don't move),
      * later re-runs that touch only a subset of speakers / segments.
    Filenames sort chronologically, so a plain ``sorted(glob("*.wav"))``
    yields playback order.
    """
    start_ms = int(round((seg.get("start") or 0) * 1000))
    end_ms = int(round((seg.get("end") or 0) * 1000))
    return f"{start_ms:010d}_{end_ms:010d}.wav"


def seg_key(seg: dict) -> str:
    """Canonical segment identity shared with the frontend.

    Mirrors ``frontend/src/lib/segKey.ts::segKey`` byte-for-byte.
    Timestamps are rounded to integer milliseconds so the key is stable
    across Python / JS float stringification: JS ``(1).toString()`` yields
    ``"1"`` while Python ``f"{1.0}"`` yields ``"1.0"``, and the mismatch
    would silently drop segments on integer-second boundaries.
    """
    speaker = seg.get("speaker") or ""
    start = seg.get("start") or 0
    end = seg.get("end") or 0
    return f"{speaker}:{round(start * 1000)}:{round(end * 1000)}"


def real_speakers(segments: list[dict]) -> list[str]:
    """Return the sorted set of real speaker labels in ``segments``.

    Drops ``[BREAK]``, empty-string placeholders, and any diarization
    placeholders in :data:`UNKNOWN_SPEAKERS`. Single source of truth for
    "which speakers count" across synthesis / voice-sample extraction.
    """
    skip = UNKNOWN_SPEAKERS | {BREAK_SPEAKER}
    return sorted({s.get("speaker", "") for s in segments} - skip)


def fill_narrator_speaker(segments: list[dict]) -> list[dict]:
    """Return ``segments`` with empty/unknown speakers relabelled to :data:`NARRATOR_SPEAKER`.

    Used at synth time so legacy transcripts (e.g. YouTube subtitle imports
    saved before the parser default landed) can still feed a single
    voice-clone bucket without forcing the user to rename through the editor.

    Returns the input list unchanged when no segment needs remapping; only
    segments that actually change get a shallow-copied dict, others share refs.
    """
    if not any(seg.get("speaker", "") in UNKNOWN_SPEAKERS for seg in segments):
        return segments
    return [
        {**seg, "speaker": NARRATOR_SPEAKER}
        if seg.get("speaker", "") in UNKNOWN_SPEAKERS
        else seg
        for seg in segments
    ]


def group_by_speaker(segments: list[dict]) -> dict[str, list[dict]]:
    """Group segments by speaker label.

    Args:
        segments : list of segment dicts with at least a ``speaker`` field

    Returns:
        ``{speaker: [seg, …]}`` preserving original order within each group.
    """
    by_speaker: dict[str, list[dict]] = {}
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        by_speaker.setdefault(speaker, []).append(seg)
    return by_speaker


def speaker_airtime(
    segments: list[dict], declared: Container[str] = ()
) -> dict[str, dict]:
    """Per-speaker airtime from a seglist.

    Returns ``{speaker: {"segment_count": int, "total_seconds": float}}``,
    summing ``end - start`` (clamped at 0) over each speaker's segments.
    Break markers and labels that name nobody are skipped, so the result holds
    only real, attributable speakers. That includes NARRATOR_SPEAKER: an
    episode transcribed without diarization carries it on every segment, and
    counting it would put a speaker nobody identified in the show roster and
    the per-episode airtime line. Shared by both of those endpoints, so the
    rule lives here rather than in each surface.
    """
    out: dict[str, dict] = {}
    for spk, segs in group_by_speaker(segments).items():
        if spk == BREAK_SPEAKER or is_unattributed(spk, declared):
            continue
        secs = sum(
            max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
            for s in segs
        )
        out[spk] = {"segment_count": len(segs), "total_seconds": secs}
    return out


def build_batched_manual_prompts(
    segments: list[dict],
    build_prompt_fn,
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    batch_count: int | None = None,
) -> list[tuple[list[dict], str]]:
    """Split *segments* into batches and build one prompt per batch.

    *build_prompt_fn* receives ``(batch, start_index)`` and returns the
    prompt string. ``start_index`` is the absolute count of real segments
    consumed by prior batches, so each batch's [N] markers are unique
    across the whole transcript — concatenated LLM responses then keep
    distinct positions.

    When *batch_count* is given it overrides *batch_minutes* and produces
    exactly that many batches (see batch_segments_by_duration).
    """
    batches = batch_segments_by_duration(segments, batch_minutes, batch_count)
    out: list[tuple[list[dict], str]] = []
    offset = 0
    for batch in batches:
        out.append((batch, build_prompt_fn(batch, offset)))
        _, real = _separate_breaks(batch)
        offset += len(real)
    return out


def _segment_end(seg: dict) -> float:
    """Best-effort end timestamp for a segment (falls back to start, then 0)."""
    return float(seg.get("end", seg.get("start", 0)) or 0)


def batch_segments_by_duration(
    segments: list[dict],
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    batch_count: int | None = None,
) -> list[list[dict]]:
    """Split segments into time-based batches.

    Splits by absolute segment start timestamp against fixed cutoffs, so
    the batch count tracks the audio's elapsed duration (not the sum of
    per-segment durations, which can under-count silence and produce fewer
    batches than the user requested).

    Args:
        segments      : transcript segments to batch
        batch_minutes : maximum duration per batch in minutes (default 15)
        batch_count   : when set, overrides batch_minutes and sizes batches
                        off the transcript's real span so the count tracks
                        the request without overshooting (a large silence
                        gap straddling a cutoff can still yield one fewer).

    Returns:
        List of non-empty segment batches (each batch is a list of segment dicts).
    """
    if not segments:
        return []
    if batch_count is not None and batch_count >= 1:
        if batch_count == 1:
            return [list(segments)]
        max_seconds = max(_segment_end(s) for s in segments) / batch_count
    else:
        max_seconds = batch_minutes * 60
    if max_seconds <= 0:
        return [list(segments)]

    batches: list[list[dict]] = []
    current: list[dict] = []
    cutoff = max_seconds

    for seg in segments:
        start = float(seg.get("start", 0))
        while start >= cutoff:
            if current:
                batches.append(current)
                current = []
            cutoff += max_seconds
        current.append(seg)

    if current:
        batches.append(current)

    # Merge a tiny overshoot tail into the previous batch. When segments
    # extend just past the final cutoff (e.g. episode.duration under-reports
    # the real transcript span), the user's chosen batch count would otherwise
    # gain a spurious extra batch containing only the last few seconds.
    if len(batches) >= 2:
        last_batch_start = cutoff - max_seconds
        span = _segment_end(batches[-1][-1]) - last_batch_start
        if 0 <= span < max_seconds * 0.15:
            batches[-2].extend(batches.pop())

    return batches


def segments_to_text(
    segments: list[dict], text_field: str = "text", declared: Container[str] = ()
) -> str:
    """Format segments as plain readable text.

    Args:
        segments   : list of segment dicts with speaker, start, end, and text fields
        text_field : which field to use for the text content (default "text")
    """
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        if is_unattributed(speaker, declared):
            speaker = ""
        start = seg.get("start")
        end = seg.get("end")
        if start is not None and end is not None:
            header = f"[{start:.3f}s - {end:.3f}s] {speaker}".rstrip()
        else:
            header = speaker
        text = seg.get(text_field) or "[empty]"
        # An untimed, unattributed segment has no header at all; emitting an
        # empty one would open the block with a blank line.
        lines.append(f"{header}\n{text}" if header else text)
    return "\n\n".join(lines)


def segments_to_srt(
    segments: list[dict], text_field: str = "text", declared: Container[str] = ()
) -> str:
    """Format segments as SRT subtitles.

    Args:
        segments   : list of segment dicts with speaker, start, end, and text fields
        text_field : which field to use for the text content (default "text")
    """
    lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        speaker = seg.get("speaker", "")
        text = seg.get(text_field) or "[empty]"
        prefix = f"{speaker}: " if not is_unattributed(speaker, declared) else ""
        lines.append(str(i))
        lines.append(f"{_srt_ts(start)} --> {_srt_ts(end)}")
        lines.append(f"{prefix}{text}")
        lines.append("")
    return "\n".join(lines)


def _srt_ts(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_vtt(
    segments: list[dict], text_field: str = "text", declared: Container[str] = ()
) -> str:
    """Format segments as WebVTT subtitles.

    Args:
        segments   : list of segment dicts with speaker, start, end, and text fields
        text_field : which field to use for the text content (default "text")
    """
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        speaker = seg.get("speaker", "")
        text = seg.get(text_field) or "[empty]"
        prefix = f"<v {speaker}>" if not is_unattributed(speaker, declared) else ""
        lines.append(f"{_vtt_ts(start)} --> {_vtt_ts(end)}")
        lines.append(f"{prefix}{text}")
        lines.append("")
    return "\n".join(lines)


def _vtt_ts(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ── Subtitle parsing (inverse of segments_to_srt / segments_to_vtt) ────


def _parse_srt_ts(ts: str) -> float:
    """Parse an SRT timestamp (``HH:MM:SS,mmm``) to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _parse_vtt_ts(ts: str) -> float:
    """Parse a VTT timestamp (``HH:MM:SS.mmm`` or ``MM:SS.mmm``) to seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


_VTT_SPEAKER_RE = re.compile(r"<v\s+([^>]+)>")


def _merge_parsed_cues(cues: list[dict]) -> list[dict]:
    """Deduplicate subtitle cues, preserving original timing.

    YouTube auto-generated subtitles often produce overlapping cues with
    repeated text.  This pass deduplicates consecutive identical lines and
    cleans HTML entities, but does NOT merge distinct cues — the original
    subtitle timing is kept as-is.
    """
    if not cues:
        return []

    # Clean HTML entities from all cues
    for cue in cues:
        cue["text"] = (
            cue["text"]
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
        )
        # Collapse multiple spaces
        cue["text"] = re.sub(r"  +", " ", cue["text"]).strip()

    # Deduplicate consecutive identical text
    deduped: list[dict] = [cues[0]]
    for cue in cues[1:]:
        prev = deduped[-1]
        if cue["text"] == prev["text"] and cue["speaker"] == prev["speaker"]:
            # Extend end time of previous cue
            prev["end"] = max(prev["end"], cue["end"])
        else:
            deduped.append(cue)

    # Detect and collapse rolling/progressive subtitles.
    # YouTube auto-generated VTTs display text progressively: each cue shows
    # the previously completed line plus new words.  Between the rolling cues
    # there are brief "flash" cues (< 0.05s) that just repeat completed text.
    # We strip the flash cues, detect rolling overlap, and extract only the
    # new text from each cue.
    if len(deduped) >= 4:
        # Remove near-zero-duration "flash" cues
        no_flash: list[dict] = []
        for cue in deduped:
            if cue["end"] - cue["start"] >= 0.05:
                no_flash.append(cue)

        # Detect rolling pattern: suffix of cue[i] == prefix of cue[i+1]
        if len(no_flash) >= 4:
            overlap_count = 0
            for i in range(len(no_flash) - 1):
                t1, t2 = no_flash[i]["text"], no_flash[i + 1]["text"]
                # Check if any suffix of t1 (>10 chars) is a prefix of t2
                min_overlap = min(10, len(t1) // 2)
                for length in range(len(t1), min_overlap - 1, -1):
                    if t2.startswith(t1[-length:]):
                        overlap_count += 1
                        break

            if overlap_count > len(no_flash) * 0.3:
                collapsed: list[dict] = []
                for i, cue in enumerate(no_flash):
                    if i == 0:
                        collapsed.append(cue)
                        continue
                    prev_text = no_flash[i - 1]["text"]
                    cur_text = cue["text"]
                    # Find longest suffix of prev that is a prefix of cur
                    best_overlap = 0
                    for length in range(len(prev_text), 0, -1):
                        if cur_text.startswith(prev_text[-length:]):
                            best_overlap = length
                            break
                    if best_overlap > 0:
                        new_text = cur_text[best_overlap:].strip()
                        if new_text:
                            collapsed.append(
                                {
                                    **cue,
                                    "text": new_text,
                                }
                            )
                    else:
                        collapsed.append(cue)
                deduped = collapsed

    return deduped


def srt_to_segments(srt_text: str) -> list[dict]:
    """Parse SRT subtitle text into segment dicts.

    Returns a list of ``{"speaker": str, "text": str, "start": float,
    "end": float}`` dicts.  Speaker is extracted from a ``Speaker: ``
    prefix if present.

    Args:
        srt_text: Full SRT file content.
    """
    cues: list[dict] = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # Find the timestamp line (skip the index line)
        ts_line = None
        text_start = 0
        for idx, line in enumerate(lines):
            if "-->" in line:
                ts_line = line
                text_start = idx + 1
                break
        if ts_line is None:
            continue
        parts = ts_line.split("-->")
        if len(parts) != 2:
            continue
        start = _parse_srt_ts(parts[0])
        end = _parse_srt_ts(parts[1])
        text = " ".join(lines[text_start:]).strip()
        # Extract speaker from "Speaker: text" prefix
        speaker = ""
        if ": " in text:
            maybe_speaker, rest = text.split(": ", 1)
            if maybe_speaker and not any(c in maybe_speaker for c in ".,!?"):
                speaker = maybe_speaker
                text = rest
        if text:
            cues.append(
                {
                    "speaker": speaker or NARRATOR_SPEAKER,
                    "text": text,
                    "start": start,
                    "end": end,
                }
            )

    return _merge_parsed_cues(cues)


def vtt_to_segments(vtt_text: str) -> list[dict]:
    """Parse WebVTT subtitle text into segment dicts.

    Handles YouTube's auto-generated format with overlapping/duplicate cues
    and ``<v SpeakerName>`` voice tags.

    Returns a list of ``{"speaker": str, "text": str, "start": float,
    "end": float}`` dicts.

    Args:
        vtt_text: Full WebVTT file content.
    """
    cues: list[dict] = []
    blocks = re.split(r"\n\s*\n", vtt_text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        # Find timestamp line
        ts_line = None
        text_start = 0
        for idx, line in enumerate(lines):
            if "-->" in line:
                ts_line = line
                text_start = idx + 1
                break
        if ts_line is None:
            continue
        # Strip position/alignment metadata after timestamp
        ts_part = ts_line.split("-->")
        if len(ts_part) != 2:
            continue
        start = _parse_vtt_ts(ts_part[0].split()[0] if ts_part[0].strip() else "0")
        end_raw = ts_part[1].strip().split()
        end = _parse_vtt_ts(end_raw[0]) if end_raw else start

        text = " ".join(lines[text_start:]).strip()
        if not text:
            continue
        # Extract speaker from <v SpeakerName> tags
        speaker = ""
        m = _VTT_SPEAKER_RE.search(text)
        if m:
            speaker = m.group(1).strip()
            text = _VTT_SPEAKER_RE.sub("", text).strip()
        # Strip remaining HTML-like tags
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            cues.append(
                {
                    "speaker": speaker or NARRATOR_SPEAKER,
                    "text": text,
                    "start": start,
                    "end": end,
                }
            )

    return _merge_parsed_cues(cues)


def merge_display_turns(turns: "list[SpeakerTurn]") -> list[dict]:
    """Collapse consecutive same-speaker turns for search-result display.

    One speaker label / one text block per contiguous speaker run —
    unlike :func:`merge_consecutive_segments`, there are no gap caps,
    duration caps, or break sentinels. Intended for rendering a single
    search-result chunk (``Hit.speakers``) where readers just want a clean
    paragraph per speaker. Output entries are plain display dicts.
    """
    out: list[dict] = []
    for t in turns:
        speaker = t.speaker or "Unknown"
        text = t.text.strip()
        if not text:
            continue
        if out and out[-1]["speaker"] == speaker:
            out[-1]["text"] += " " + text
            # A turn with no timing (legacy rows default end to 0.0) must not
            # drag the merged run's end backwards.
            if t.end:
                out[-1]["end"] = t.end
        else:
            out.append(
                {"speaker": speaker, "text": text, "start": t.start, "end": t.end}
            )
    return out


def merge_consecutive_segments(
    segments: list[dict],
    max_gap: float = DEFAULT_MAX_GAP,
    max_duration: float = 15.0,
) -> list[dict]:
    """
    Merge consecutive segments from the same speaker into single entries.
    Segments are only merged if the gap between them is <= max_gap seconds,
    preventing merges across music breaks or long silences.

    Args:
        segments     : raw diarized segments
        max_gap      : maximum silence gap (seconds) to merge across (default 10s);
                       0 disables merging
        max_duration : maximum duration (seconds) for a merged segment (default 15s);
                       keeps segments subtitle-sized for readability

    Returns:
        List of simplified segments [{speaker, start, end, text}]
    """
    n_input = len(segments)
    result = []
    for seg in segments:
        speaker = seg.get("speaker_name") or seg.get("speaker") or "UNKNOWN"
        raw_start = seg.get("start")
        raw_end = seg.get("end")
        has_times = raw_start is not None and raw_end is not None
        entry: dict = {
            "speaker": speaker,
            "text": str(seg.get("text", "")).strip(),
        }
        if has_times:
            entry["start"] = round(float(raw_start), 3)
            entry["end"] = round(float(raw_end), 3)

        prev = result[-1] if result else None
        if prev and prev["speaker"] == entry["speaker"]:
            # With timestamps: merge only if gap <= max_gap and duration <= max_duration
            # Without timestamps: always merge consecutive same-speaker
            if has_times and "start" in prev:
                gap = entry["start"] - prev["end"]
                merged_duration = entry["end"] - prev["start"]
                if gap <= max_gap and merged_duration <= max_duration:
                    prev["end"] = entry["end"]
                    prev["text"] += " " + entry["text"]
                elif gap > max_gap:
                    result.append(
                        {
                            "speaker": BREAK_SPEAKER,
                            "start": prev["end"],
                            "end": entry["start"],
                            "text": "",
                        }
                    )
                    result.append(entry)
                else:
                    # Duration cap hit — start a new segment, no break
                    result.append(entry)
            else:
                prev["text"] += " " + entry["text"]
                if has_times:
                    prev["end"] = entry["end"]
        else:
            # Different speaker — check for break insertion (only with timestamps)
            if prev and has_times and "end" in prev:
                if entry["start"] - prev["end"] > max_gap:
                    result.append(
                        {
                            "speaker": BREAK_SPEAKER,
                            "start": prev["end"],
                            "end": entry["start"],
                            "text": "",
                        }
                    )
            result.append(entry)
    n_breaks = sum(1 for s in result if s["speaker"] == BREAK_SPEAKER)
    logger.debug(
        f"merge_consecutive_segments: {n_input} → {len(result)} segments "
        f"({n_breaks} breaks, max_gap={max_gap}s, max_duration={max_duration}s)"
    )
    return result


# ──────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────


def build_llm_prompt(
    role: str,
    task: str,
    output: str,
    context: str = "",
    context_extra: str = "",
) -> str:
    """Assemble a system prompt from standard sections.

    Args:
        role          : opening role sentence
        task          : bullet-list of task instructions
        output        : output format instructions
        context       : optional podcast context; omitted when empty
        context_extra : additional sentence appended to the context block
    """
    context_section = (
        f"Context about this podcast: {context}\n"
        "Any names, titles, brands, or terms mentioned in the context above are the CORRECT spellings."
        + (f" {context_extra}" if context_extra else "")
        if context
        else ""
    )
    sections = [role, context_section, task, output]
    return "\n\n".join(s for s in sections if s)


# ──────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────


def _is_break(seg: dict) -> bool:
    """Return True for [BREAK] segments (music/jingle markers)."""
    return seg.get("speaker") == BREAK_SPEAKER


def _separate_breaks(
    segments: list[dict],
) -> tuple[list[int], list[dict]]:
    """Split segments into real content and [BREAK] markers.

    Returns:
        (real_indices, real_segments) — positions and segments that are
        not ``[BREAK]`` markers.
    """
    real_indices: list[int] = []
    real_segs: list[dict] = []
    for i, seg in enumerate(segments):
        if not _is_break(seg):
            real_indices.append(i)
            real_segs.append(seg)
    return real_indices, real_segs


def _reassemble_breaks(
    segments: list[dict],
    real_indices: list[int],
    processed: list[dict],
) -> list[dict]:
    """Merge processed results back with [BREAK] segments in original order."""
    real_set = set(real_indices)
    results: list[dict] = []
    proc_iter = iter(processed)
    for i, seg in enumerate(segments):
        if i in real_set:
            results.append(next(proc_iter))
        else:
            results.append(seg)
    return results


def format_segments(
    segments: list[dict],
    instruction: str = "Process",
    start_index: int = 0,
) -> str:
    """Format segments as a numbered user message for the LLM.

    Produces the same ``[i] text`` format used by all three modes
    (ollama, api, manual).  ``[BREAK]`` segments are excluded.

    Args:
        segments    : transcript segments (breaks are filtered out)
        instruction : verb for the closing instruction line
        start_index : first absolute index for numbering (used by manual mode
                      so concatenated batch responses keep unique indices)
    """
    _, real = _separate_breaks(segments)
    n = len(real)
    lines = [f"[{start_index + i}] {seg['text']}" for i, seg in enumerate(real)]
    first = start_index
    last = start_index + n - 1 if n > 0 else start_index
    lines.append(
        f"\n{instruction} all {n} segments above. "
        f"Output MUST contain exactly {n} entries with indices {first}..{last}, "
        "no gaps, no extras, no renumbering. Verify the count before responding."
    )
    return "\n\n".join(lines)


def parse_llm_response(raw: str) -> dict[int, dict]:
    """Parse a raw LLM response string into a dict keyed by segment position.

    Keys are positional (0..N-1) — the LLM's own ``index`` field is treated
    as advisory only. Position-based mapping is safer because callers verify
    ``len(parsed) == len(input)`` before applying, so position equals the
    intended target index regardless of any renumbering by the LLM.

    Strips ``<think>`` tags and markdown fences before parsing JSON. Falls
    back through tiered repair (trim trailing junk, fix invalid ``\\'``
    escape, regex-extract orphan ``"text": "..."`` pairs) so a single
    schema slip from a small model doesn't drop the whole batch.

    Returns:
        ``{position: {"text": "...", ...}}`` dict.  Empty dict on parse failure.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    def _to_index(parsed: list) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for i, item in enumerate(parsed):
            if isinstance(item, dict):
                out[i] = item
            elif isinstance(item, str):
                out[i] = {"text": item}
        return out

    try:
        return _to_index(json.loads(raw))
    except Exception as first_err:
        # Cross-model repair: trailing junk after the array close, and the
        # `\'` escape (invalid in JSON, valid in Python/JS). Both seen on
        # multiple model sizes, not just one run.
        repaired = raw
        last_bracket = repaired.rfind("]")
        if last_bracket != -1:
            repaired = repaired[: last_bracket + 1]
        repaired = repaired.replace("\\'", "'")
        try:
            return _to_index(json.loads(repaired))
        except Exception:
            pass

        logger.warning(f"Parse error: {first_err}, batch will keep original text")
        logger.warning(f"Raw response (first 600 chars): {raw[:600]}")
        return {}


def apply_corrections(
    batch: list[dict],
    by_index: dict[int, dict],
    min_length_ratio: float = 0.7,
) -> list[dict]:
    """Apply LLM corrections to a batch of segments.

    Merges corrected text from *by_index* into the original segments.
    ``[BREAK]`` segments are passed through unchanged.  Segments whose
    corrected text is suspiciously short (below *min_length_ratio* of the
    original) keep their original text.

    Args:
        batch            : original segments (may include ``[BREAK]``s)
        by_index         : ``{index: {"text": "..."}}`` from the LLM
        min_length_ratio : minimum corrected/original length ratio (0 to disable)

    Returns:
        List of segments with text field updated.
    """
    real_indices, real_segs = _separate_breaks(batch)

    corrected_segs: list[dict] = []
    changed = 0
    for i, seg in enumerate(real_segs):
        item = by_index.get(i, {})
        original_text = seg["text"]
        corrected_text = item.get("text", original_text)

        if not corrected_text:
            logger.warning(f"Segment [{i}] has no corrected text — keeping original")
            corrected_text = original_text

        if (
            min_length_ratio
            and original_text
            and len(corrected_text) < len(original_text) * min_length_ratio
        ):
            logger.warning(
                f"Segment [{i}] truncated by LLM "
                f"({len(corrected_text)} vs {len(original_text)} chars), keeping original. "
                f"original={original_text!r:.120} corrected={corrected_text!r:.120}"
            )
            corrected_text = original_text

        if corrected_text != original_text:
            changed += 1
        entry = {**seg, "text": corrected_text}
        entry.pop("index", None)
        corrected_segs.append(entry)

    logger.debug(f"Batch: {changed}/{len(real_segs)} segments modified")
    return _reassemble_breaks(batch, real_indices, corrected_segs)


def call_and_parse(
    batch: list[dict],
    system_prompt: str,
    call_fn,
    instruction: str = "Process",
    min_length_ratio: float = 0.7,
    start_index: int = 0,
    on_outcome: Callable[[str, int, int, str, str], None] | None = None,
) -> list[dict]:
    """Call the LLM for one batch and parse the response.

    Uses :func:`format_segments`, :func:`parse_llm_response`, and
    :func:`apply_corrections` — the same pipeline that manual mode uses.
    ``[BREAK]`` segments are passed through unchanged.

    ``start_index`` shifts the displayed ``[N]`` markers in the prompt so
    log lines and the LLM see absolute positions across batches.

    ``on_outcome``, when given, is called once with
    ``(raw, expected, got, status, reason)`` — ``status`` is ``"ok"`` or
    ``"rejected"`` — so a caller can record per-batch results.
    """
    _, real_segs = _separate_breaks(batch)
    if not real_segs:
        return list(batch)

    user_content = format_segments(
        batch, instruction=instruction, start_index=start_index
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw = call_fn(messages)
    logger.debug(f"LLM response: {len(raw)} chars")
    by_index = parse_llm_response(raw)

    expected = len(real_segs)
    got = len(by_index)
    status, reason = "ok", ""

    if not by_index:
        # parse_llm_response already logged the parse error; flag the batch
        # so the run's failure record shows it produced no corrections.
        status, reason = "rejected", "parse failure"
    elif got != expected:
        # LLM count drift: a response with fewer/more items than the input
        # batch means indices got renumbered, which would silently misalign
        # corrections. Reject the whole batch and keep originals.
        logger.warning(
            f"LLM returned {got} items for {expected} segments, "
            "rejecting batch to avoid index drift; keeping original text."
        )
        # Surface enough of the raw response to diagnose the shape mismatch
        # (top-level object vs array, wrapping key, single concatenated
        # string). Without this the rejection is opaque.
        logger.warning(f"Rejected raw response (first 600 chars): {raw[:600]}")
        sample = next(iter(by_index.values()), None)
        logger.warning(
            f"Rejected first item shape: {type(sample).__name__}={sample!r:.300}"
        )
        status, reason = "rejected", "count drift"
        by_index = {}

    if on_outcome is not None:
        on_outcome(raw, expected, got, status, reason)

    return apply_corrections(batch, by_index, min_length_ratio=min_length_ratio)


def _append_batch_record(
    batch_sink: list[dict] | None,
    *,
    batch_num: int,
    real_segs: list[dict],
    offset: int,
    raw: str,
    expected: int,
    got: int,
    status: str,
    reason: str,
) -> None:
    """Append one batch's LLM outcome to *batch_sink* (no-op when None)."""
    if batch_sink is None:
        return
    batch_sink.append(
        {
            "batch": batch_num,
            "status": status,
            "reason": reason,
            "expected": expected,
            "got": got,
            "raw": raw,
            "input": [
                {"index": offset + i, "text": s.get("text", "")}
                for i, s in enumerate(real_segs)
            ],
        }
    )


def _outcome_recorder(
    batch_sink: list[dict] | None,
    batch_num: int,
    real_segs: list[dict],
    offset: int,
) -> Callable[[str, int, int, str, str], None]:
    """Build a ``call_and_parse`` on_outcome callback bound to one batch."""

    def record(raw: str, expected: int, got: int, status: str, reason: str) -> None:
        _append_batch_record(
            batch_sink,
            batch_num=batch_num,
            real_segs=real_segs,
            offset=offset,
            raw=raw,
            expected=expected,
            got=got,
            status=status,
            reason=reason,
        )

    return record


def run_ollama(
    segments: list[dict],
    system_prompt: str,
    model: str,
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    instruction: str = "Process",
    min_length_ratio: float = 0.7,
    label: str = "",
    on_batch: Callable[[int, int], None] | None = None,
    batch_sink: list[dict] | None = None,
) -> list[dict]:
    """Run segments through a local Ollama model.

    Args:
        segments: source segments to process.
        system_prompt: system prompt for the LLM.
        model: Ollama model name.
        batch_minutes: max audio duration per batch in minutes.
        instruction: verb for user-message formatting (e.g. "Correct", "Translate").
        min_length_ratio: minimum output/input length ratio before flagging.
        label: human-readable label for log messages.
        on_batch: optional callback(batch_num, total_batches) for progress.

    Returns:
        Processed segments with updated text fields.
    """
    import time

    import httpx
    from ollama import Client, ResponseError

    host = ollama_host()
    client = Client(
        host=host,
        timeout=httpx.Timeout(OLLAMA_READ_TIMEOUT_S, connect=OLLAMA_CONNECT_TIMEOUT_S),
    )
    _warn_if_ollama_too_old(host)
    _warn_if_model_unpulled(client, model)

    results = []
    batches = batch_segments_by_duration(segments, batch_minutes)
    n_batches = len(batches)
    offset = 0

    for batch_num, batch in enumerate(batches, 1):
        logger.info(f"{label} batch {batch_num}/{n_batches} via Ollama ({model})")
        _, real_segs = _separate_breaks(batch)
        n_items = len(real_segs)
        schema = correction_schema(n_items)
        num_predict, num_ctx = _ollama_token_budget(n_items)

        def call_fn(messages):
            for attempt in range(3):
                try:
                    response = client.chat(
                        model=model,
                        messages=messages,
                        options={
                            "temperature": OLLAMA_TEMPERATURE,
                            "num_predict": num_predict,
                            "num_ctx": num_ctx,
                        },
                        format=schema,
                        # Reasoning models (Qwen3, DeepSeek-R1) emit
                        # `<think>...</think>` before the answer, which
                        # conflicts with schema-constrained decoding and
                        # burns the whole num_predict budget producing zero
                        # JSON output.
                        think=False,
                        keep_alive=OLLAMA_KEEP_ALIVE,
                    )
                    break
                except (httpx.HTTPError, ResponseError, ConnectionError) as e:
                    if attempt == 2:
                        raise
                    backoff = 2**attempt
                    logger.warning(
                        f"Ollama call failed (attempt {attempt + 1}/3): {e}. "
                        f"Retrying in {backoff}s."
                    )
                    time.sleep(backoff)

            content = response.message.content.strip()
            pec = response.prompt_eval_count or 0
            # Ollama silently chops the prompt when it exceeds num_ctx; the
            # system prompt is what gets cut first, so the model ends up
            # following a partial instruction.
            if pec > num_ctx * 0.9:
                logger.warning(
                    f"Prompt used {pec}/{num_ctx} tokens (>=90%); Ollama may "
                    "have truncated the system prompt. Reduce batch size."
                )
            if not content:
                logger.warning(
                    f"Empty response from {model}. done_reason="
                    f"{response.done_reason!r} eval_count={response.eval_count} "
                    f"prompt_eval_count={pec} done={response.done}"
                )
            return content

        results.extend(
            call_and_parse(
                batch,
                system_prompt,
                call_fn,
                instruction=instruction,
                min_length_ratio=min_length_ratio,
                start_index=offset,
                on_outcome=_outcome_recorder(batch_sink, batch_num, real_segs, offset),
            )
        )
        offset += n_items
        if on_batch:
            on_batch(batch_num, n_batches)

    return results


def run_api(
    segments: list[dict],
    system_prompt: str,
    model: str,
    api_base_url: str,
    api_key: str | None,
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    provider: str | None = None,
    instruction: str = "Process",
    min_length_ratio: float = 0.7,
    label: str = "",
    on_batch: Callable[[int, int], None] | None = None,
    batch_sink: list[dict] | None = None,
) -> list[dict]:
    """Run segments through an OpenAI-compatible API.

    Args:
        segments: source segments to process.
        system_prompt: system prompt for the LLM.
        model: model name (auto-detected from provider if empty).
        api_base_url: base URL (auto-detected from provider if empty).
        api_key: API key (None reads from provider's env variable).
        batch_minutes: max audio duration per batch in minutes.
        provider: provider shorthand ("openai", "anthropic", "mistral").
        instruction: verb for user-message formatting.
        min_length_ratio: minimum output/input length ratio before flagging.
        label: human-readable label for log messages.
        on_batch: optional callback(batch_num, total_batches) for progress.

    Returns:
        Processed segments with updated text fields.
    """
    import os

    from openai import OpenAI

    from podcodex.core.constants import LLM_PROVIDER_DEFAULTS

    if provider and provider in LLM_PROVIDER_DEFAULTS:
        spec = LLM_PROVIDER_DEFAULTS[provider]
        model = model or spec["model"]
        api_key = api_key or os.environ.get(spec["env_var"])

    key = api_key or os.environ.get("API_KEY")
    if not key:
        raise ValueError(
            "No API key found. Set the provider's API key env variable or pass api_key=."
        )

    logger.debug(
        f"API config: base_url={api_base_url}, model={model}, provider={provider}"
    )
    client = OpenAI(api_key=key, base_url=api_base_url)
    results = []
    batches = batch_segments_by_duration(segments, batch_minutes)
    n_batches = len(batches)
    offset = 0

    for batch_num, batch in enumerate(batches, 1):
        logger.info(f"{label} batch {batch_num}/{n_batches} via API ({model})")
        _, real_segs = _separate_breaks(batch)

        def call_fn(messages):
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=DEFAULT_TEMPERATURE
            )
            return response.choices[0].message.content.strip()

        results.extend(
            call_and_parse(
                batch,
                system_prompt,
                call_fn,
                instruction=instruction,
                min_length_ratio=min_length_ratio,
                start_index=offset,
                on_outcome=_outcome_recorder(batch_sink, batch_num, real_segs, offset),
            )
        )
        offset += len(real_segs)
        if on_batch:
            on_batch(batch_num, n_batches)

    return results


def validate_manual(
    corrections: list[dict], original_segments: list[dict]
) -> list[dict]:
    """Merge LLM-returned corrections with original source segments.

    Uses position-based mapping — corrections must be in the same order as
    the (non-[BREAK]) source segments. The LLM-supplied ``index`` field, if
    present, is ignored. Count must match exactly or the whole batch is
    rejected and originals are kept.

    Args:
        corrections       : list of {"text": "..."} entries from the LLM, in order
        original_segments : source segments (speaker, start, end, text, ...)

    Returns:
        List of segments with text field updated from corrections.
    """
    if not isinstance(corrections, list) or not corrections:
        raise ValueError("Expected a non-empty JSON array from the LLM.")
    if "text" not in corrections[0]:
        raise ValueError(
            f"Expected 'text' field in each entry. "
            f"Fields found: {sorted(corrections[0].keys())}"
        )

    _, real_segs = _separate_breaks(original_segments)
    if len(corrections) != len(real_segs):
        logger.warning(
            f"Correction count mismatch: {len(corrections)} corrections "
            f"vs {len(real_segs)} segments (excluding "
            f"{len(original_segments) - len(real_segs)} breaks) — "
            "rejecting to avoid index drift; keeping original text."
        )
        by_index: dict[int, dict] = {}
    else:
        # Position-based mapping (LLM's index field is advisory only).
        by_index = {i: item for i, item in enumerate(corrections)}
    results = apply_corrections(original_segments, by_index, min_length_ratio=0)

    logger.info(f"Manual corrections validated — {len(results)} segments")
    return results


def run_llm_pipeline(
    segments: list[dict],
    system_prompt: str,
    *,
    mode: str = "ollama",
    model: str = "",
    api_base_url: str = "",
    api_key: str | None = None,
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    provider: str | None = None,
    instruction: str = "Process",
    label: str = "",
    original_segments: list[dict] | None = None,
    merge: bool = True,
    max_gap: float = DEFAULT_MAX_GAP,
    on_batch: Callable[[int, int], None] | None = None,
    batch_sink: list[dict] | None = None,
) -> list[dict]:
    """Run an LLM pipeline (correct or translate) on segments.

    Handles manual/ollama/api modes, optional merge, and progress callbacks.
    When *batch_sink* is given, each batch's LLM outcome is appended to it
    (ollama/api modes only).
    """
    if mode == "manual":
        orig = original_segments if original_segments is not None else segments
        return validate_manual(segments, orig)

    if merge:
        segments = merge_consecutive_segments(segments, max_gap=max_gap)
        logger.info(f"After merge: {len(segments)} segments")

    if mode == "ollama":
        from podcodex.core.constants import DEFAULT_OLLAMA_MODEL

        return run_ollama(
            segments,
            system_prompt,
            model=model or DEFAULT_OLLAMA_MODEL,
            batch_minutes=batch_minutes,
            instruction=instruction,
            label=label,
            on_batch=on_batch,
            batch_sink=batch_sink,
        )
    elif mode == "api":
        return run_api(
            segments,
            system_prompt,
            model=model,
            api_base_url=api_base_url,
            api_key=api_key,
            batch_minutes=batch_minutes,
            provider=provider,
            instruction=instruction,
            label=label,
            on_batch=on_batch,
            batch_sink=batch_sink,
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Choose from 'manual', 'ollama', 'api'."
        )
