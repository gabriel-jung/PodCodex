"""
podcodex.core._utils — Shared utilities for the core pipeline.

Heavy libraries (torch, pandas) are imported lazily inside functions
so this module stays cheap to import at the top level.
"""

import functools
import gc
import json
import os
import re
import stat
from datetime import datetime
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Container
from typing import Self

from loguru import logger

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


def episode_base(show_dir: Path, stem: str) -> Path:
    """The version root of an episode: ``{show}/{stem}/{stem}``.

    With :func:`show_dir_of`, the one definition of the episode layout for
    code that holds a show folder and a stem rather than an audio path.
    """
    return Path(show_dir) / stem / stem


def show_dir_of(base: Path) -> Path:
    """The show folder of an episode version root (see :func:`episode_base`)."""
    return base.parent.parent


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
            base = episode_base(root.parent, root.name)
        else:
            raise ValueError("Either audio_path or output_dir must be provided")
        base.parent.mkdir(parents=True, exist_ok=True)
        return cls(audio_path=audio_path or base, base=base)

    # — RAG —

    @property
    def show_dir(self) -> Path:
        """Show-level directory (parent of the episode output dir)."""
        return show_dir_of(self.base)

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
    sample key on disk (see fill_narrator_speaker), which is why
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
    traversal (".", ".."), carrying a separator, or a Windows drive prefix.
    Single facility for every folder/file-name safety check (API routes,
    bundle import).

    The drive check matters on Windows only, but is applied everywhere since
    a bundle is written on one machine and imported on another: pathlib
    joins a drive-relative name like ``C:..`` or ``D:foo`` by replacing or
    climbing out of the base (``shows / "C:.."`` is the shows folder's
    parent), which no separator check sees. A colon anywhere else (a title
    like "Episode 3: the return") stays legal.
    """
    from pathlib import PureWindowsPath

    return (
        not name
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
        or bool(PureWindowsPath(name).drive)
    )


_UNSAFE_FILENAME_CHARS = re.compile(r'[/\\:*?"<>|\[\]\x00-\x1f]')


def speaker_file_slug(speaker: str) -> str:
    """Make a speaker label safe to use as one filename component.

    Speaker labels reach disk verbatim (``{speaker}_00.wav`` under
    ``voice_samples/``, plus the glob that clears old samples). Subtitle
    imports keep whatever sits inside a ``<v ...>`` tag, so a label can carry
    path separators or glob metacharacters: ``../../x`` wrote the clips
    outside the samples directory and let the cleanup glob follow the ``..``
    segments into ancestor directories.

    Single facility for every site that turns a label into a path
    (extraction, the upload route, the loader that globs them back); they
    must agree or the samples go missing. Only the dangerous characters are
    replaced, so ordinary names (spaces, dots, accents) keep the filenames
    they already have on disk. The label always stays a *prefix* of the
    filename, so a leading "." or ".." cannot become a path component.
    """
    return _UNSAFE_FILENAME_CHARS.sub("_", speaker or "")


# Every PodCodex temp file starts with this, and startup recovery reaps
# exactly this pattern (core/recovery.py). Write temp files through
# atomic_write or name them with temp_sibling so none escapes the reaper.
TEMP_PREFIX = ".tmp_"


def temp_sibling(path: Path) -> Path:
    """A fixed temp name next to *path* that startup recovery will reap."""
    return path.with_name(f"{TEMP_PREFIX}{path.name}")


@functools.cache
def _default_file_mode() -> int:
    """The mode a plain ``open(path, "w")`` would give a new file.

    ``os.umask`` can only be read by setting it, so it is read once. The
    first call happens on the first write, long after startup has settled.
    """
    mask = os.umask(0o022)
    os.umask(mask)
    return 0o666 & ~mask


def atomic_write(
    path: Path,
    writer_fn,
    *,
    suffix: str = "",
    tag: str = "",
    durable: bool = True,
    exclusive: bool = False,
) -> None:
    """Write to ``path`` atomically via same-dir temp file + ``os.replace``.

    ``writer_fn`` receives the temp Path and must fully write it. On any
    exception the temp file is removed so the destination is never a
    half-written or zero-filled stub visible to readers (cloud-sync
    clients, other processes).

    Args:
        suffix:    temp file suffix (some writers pick the format from it).
        tag:       recognisable part of the temp name, after ``TEMP_PREFIX``.
        durable:   fsync the temp file before the rename. Without it a power
                   loss can persist the rename ahead of the data and leave
                   exactly the zero-filled file this exists to prevent. Off
                   only for output that is cheap to regenerate.
        exclusive: publish only if *path* does not exist yet (hard link
                   instead of replace); raises FileExistsError otherwise.
    """
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f"{TEMP_PREFIX}{tag}", suffix=suffix
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        # mkstemp creates the file 0600, and os.replace publishes that mode:
        # every episode file would become owner-only, unreadable from a
        # Samba share or a container running as another user. Keep the mode
        # the file already has (0600 secrets stay 0600), else the umask
        # default. A writer may still chmod the temp file itself.
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            mode = _default_file_mode()
        os.chmod(tmp_path, mode)
        writer_fn(tmp_path)
        if durable:
            with tmp_path.open("rb+") as f:
                os.fsync(f.fileno())
        if not exclusive:
            os.replace(tmp_path, path)
            return
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            raise
        except OSError:
            # No hard links on this filesystem (FAT, some network shares).
            if path.exists():
                raise FileExistsError(path) from None
            os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_json_atomic(
    path: Path, data, *, tag: str = "", sort_keys: bool = False
) -> None:
    """Write ``data`` as formatted JSON atomically."""

    def _write(p: Path) -> None:
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=sort_keys)
            f.write("\n")

    atomic_write(path, _write, suffix=".tmp", tag=tag)


def wav_duration(path: Path) -> float:
    """Return WAV duration in seconds, or 0.0 on error."""
    import soundfile as sf

    try:
        return sf.info(str(path)).duration
    except (OSError, RuntimeError):
        return 0.0


def default_batch_size() -> int:
    """Return 16 if total VRAM > 10 GB, else 8."""
    from podcodex.core.device import vram_bytes

    vram = vram_bytes()
    return 16 if vram and vram[1] > 10 * 1024 * 1024 * 1024 else 8


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
    from podcodex.core.device import cuda_available, vram_bytes

    if not cuda_available():
        return
    # flush first so the reading is accurate
    free_vram()
    vram = vram_bytes()
    if vram is None:
        return
    free_bytes, total_bytes = vram
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
