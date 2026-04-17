"""
podcodex.core._utils — Shared utilities for the core pipeline.

Heavy libraries (torch, pandas) are imported lazily inside functions
so this module stays cheap to import at the top level.
"""

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Self

from loguru import logger


# ──────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────


VOICE_SAMPLES_DIR = "voice_samples"
TTS_SEGMENTS_DIR = "tts_segments"


@dataclass
class AudioPaths:
    """All derived file paths for a given audio file.

    Centralises path logic for the entire pipeline (transcribe, correct,
    translate, synthesize).  Create via the ``from_audio`` classmethod::

        p = AudioPaths.from_audio("episode.mp3")
        p.transcript        # → …/episode/episode.transcript.json
        p.synthesized       # → …/episode/episode.synthesized.wav
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

    # — Transcription —

    @property
    def transcript_raw(self) -> Path:
        return self.base.with_suffix(".transcript.raw.json")

    @property
    def transcript(self) -> Path:
        return self.base.with_suffix(".transcript.json")

    @property
    def transcript_best(self) -> Path:
        """Validated transcript if it exists, else raw."""
        return self.transcript if self.transcript.exists() else self.transcript_raw

    # — Synthesis —

    @property
    def voice_samples_dir(self) -> Path:
        d = self.base.parent / VOICE_SAMPLES_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def tts_segments_dir(self) -> Path:
        d = self.base.parent / TTS_SEGMENTS_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def synthesized(self) -> Path:
        return self.base.with_suffix(".synthesized.wav")


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────


# Speaker labels that don't represent a real person (unresolved diarization placeholders).
# Used by transcribe.py (filtering) and synthesize.py (voice sample extraction).
UNKNOWN_SPEAKERS = frozenset({"UNKNOWN", "UNK", "None", "none"})

# Default speaker label when diarization is skipped.
NARRATOR_SPEAKER = "Narrator"

# Segment inserted by merge_consecutive_segments when gap > max_gap.
BREAK_SPEAKER = "[BREAK]"

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
    """Write a list of dicts to a parquet file."""
    import pandas as pd

    pd.DataFrame(records).to_parquet(path, index=False)


def read_json(path: Path):
    """Read and parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    """Write data as formatted JSON."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


def episode_display(chunk: dict) -> str:
    """Best human-readable episode title for a chunk.

    Canonical resolution order:
      1. ``chunk["episode_title"]`` — RSS title injected at index time.
      2. humanised ``chunk["episode"]`` stem.

    Used by the bot, MCP server, and the desktop API so every consumer
    cites the same title for the same episode.
    """
    return chunk.get("episode_title") or humanize_stem(chunk.get("episode", ""))


def write_json_atomic(path: Path, data, *, prefix: str = ".tmp_") -> None:
    """Write ``data`` as formatted JSON atomically.

    Uses a same-directory temp file + ``os.replace`` so a crash mid-write
    can never leave a half-written config visible to readers (this matters
    for files other tools — Claude Desktop, the bot — may read concurrently).
    """
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=prefix,
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def wav_duration(path: Path) -> float:
    """Return WAV duration in seconds, or 0.0 on error."""
    import soundfile as sf

    try:
        return sf.info(str(path)).duration
    except (OSError, RuntimeError):
        return 0.0


def default_batch_size() -> int:
    """Return 16 if total VRAM > 10 GB, else 8."""
    try:
        import torch

        if torch.cuda.is_available():
            _, total = torch.cuda.mem_get_info()
            if total > 10 * 1024 * 1024 * 1024:
                return 16
    except Exception:
        pass
    return 8


def free_vram() -> None:
    """Flush VRAM — call after ``del model`` in the caller's scope."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_vram(label: str = "model", min_mb: int = 512) -> None:
    """Flush caches then raise if free VRAM is below *min_mb*.

    Call this on CUDA devices before loading a heavy model.  On CPU or
    when CUDA is unavailable, this is a no-op.
    """
    import torch

    if not torch.cuda.is_available():
        return
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


def batch_segments_by_duration(
    segments: list[dict], batch_minutes: float = DEFAULT_BATCH_MINUTES
) -> list[list[dict]]:
    """Split segments into time-based batches.

    Args:
        segments      : transcript segments to batch
        batch_minutes : maximum duration per batch in minutes (default 15)

    Returns:
        List of segment batches (each batch is a list of segment dicts).
    """
    max_seconds = batch_minutes * 60
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_duration = 0.0

    for seg in segments:
        seg_duration = seg.get("end", 0) - seg.get("start", 0)
        if current and current_duration + seg_duration > max_seconds:
            batches.append(current)
            current = []
            current_duration = 0.0
        current.append(seg)
        current_duration += seg_duration

    if current:
        batches.append(current)

    return batches


def segments_to_text(segments: list[dict], text_field: str = "text") -> str:
    """Format segments as plain readable text.

    Args:
        segments   : list of segment dicts with speaker, start, end, and text fields
        text_field : which field to use for the text content (default "text")
    """
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start")
        end = seg.get("end")
        if start is not None and end is not None:
            header = f"[{start:.3f}s - {end:.3f}s] {speaker}"
        else:
            header = speaker
        text = seg.get(text_field) or "[empty]"
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def segments_to_srt(segments: list[dict], text_field: str = "text") -> str:
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
        prefix = f"{speaker}: " if speaker else ""
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


def segments_to_vtt(segments: list[dict], text_field: str = "text") -> str:
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
        prefix = f"<v {speaker}>" if speaker else ""
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
            cues.append({"speaker": speaker, "text": text, "start": start, "end": end})

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
            cues.append({"speaker": speaker, "text": text, "start": start, "end": end})

    return _merge_parsed_cues(cues)


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
    return seg.get("speaker") == "[BREAK]"


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
    results: list[dict] = []
    proc_iter = iter(processed)
    for i, seg in enumerate(segments):
        if i in real_indices:
            results.append(next(proc_iter))
        else:
            results.append(seg)
    return results


def format_segments(segments: list[dict], instruction: str = "Process") -> str:
    """Format segments as a numbered user message for the LLM.

    Produces the same ``[i] text`` format used by all three modes
    (ollama, api, manual).  ``[BREAK]`` segments are excluded.

    Args:
        segments    : transcript segments (breaks are filtered out)
        instruction : verb for the closing instruction line
    """
    _, real = _separate_breaks(segments)
    lines = [f"[{i}] {seg['text']}" for i, seg in enumerate(real)]
    lines.append(f"\n{instruction} all {len(real)} numbered segments above.")
    return "\n\n".join(lines)


def parse_llm_response(raw: str) -> dict[int, dict]:
    """Parse a raw LLM response string into a dict keyed by segment index.

    Strips ``<think>`` tags and markdown fences before parsing JSON.

    Returns:
        ``{index: {"text": "...", ...}}`` dict.  Empty dict on parse failure.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
        by_index = {item.get("index", i): item for i, item in enumerate(parsed)}
        logger.debug(f"Parsed {len(parsed)} items from LLM response")
        return by_index
    except Exception as e:
        logger.warning(f"Parse error: {e} — batch will keep original text")
        logger.debug(f"Raw response (first 500 chars): {raw[:500]}")
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
                f"({len(corrected_text)} vs {len(original_text)} chars) — keeping original"
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
) -> list[dict]:
    """Call the LLM for one batch and parse the response.

    Uses :func:`format_segments`, :func:`parse_llm_response`, and
    :func:`apply_corrections` — the same pipeline that manual mode uses.
    ``[BREAK]`` segments are passed through unchanged.
    """
    _, real_segs = _separate_breaks(batch)
    if not real_segs:
        return list(batch)

    user_content = format_segments(batch, instruction=instruction)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw = call_fn(messages)
    logger.debug(f"LLM response: {len(raw)} chars")
    by_index = parse_llm_response(raw)
    return apply_corrections(batch, by_index, min_length_ratio=min_length_ratio)


def run_ollama(
    segments: list[dict],
    system_prompt: str,
    model: str,
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    instruction: str = "Process",
    min_length_ratio: float = 0.7,
    label: str = "",
    on_batch: Callable[[int, int], None] | None = None,
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
    from ollama import Client

    client = Client()
    results = []
    batches = batch_segments_by_duration(segments, batch_minutes)
    n_batches = len(batches)

    for batch_num, batch in enumerate(batches, 1):
        logger.info(f"{label} batch {batch_num}/{n_batches} via Ollama ({model})")

        def call_fn(messages):
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": DEFAULT_TEMPERATURE},
                format="json",
            )
            return response.message.content.strip()

        results.extend(
            call_and_parse(
                batch,
                system_prompt,
                call_fn,
                instruction=instruction,
                min_length_ratio=min_length_ratio,
            )
        )
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

    from podcodex.core.constants import LLM_PROVIDERS

    if provider and provider in LLM_PROVIDERS:
        spec = LLM_PROVIDERS[provider]
        api_base_url = api_base_url or spec["url"]
        model = model or spec["model"]
        api_key = api_key or os.environ.get(spec.get("env_var", ""))

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

    for batch_num, batch in enumerate(batches, 1):
        logger.info(f"{label} batch {batch_num}/{n_batches} via API ({model})")

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
            )
        )
        if on_batch:
            on_batch(batch_num, n_batches)

    return results


def validate_manual(
    corrections: list[dict], original_segments: list[dict]
) -> list[dict]:
    """Merge LLM-returned corrections with original source segments.

    Uses :func:`parse_llm_response` (via raw JSON) and
    :func:`apply_corrections` — the same pipeline that ollama/api use.
    ``[BREAK]`` segments are passed through unchanged.

    Args:
        corrections       : list of {"index": i, "text": "corrected text"} from LLM
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
            f"{len(original_segments) - len(real_segs)} breaks)"
        )

    by_index = {item.get("index", i): item for i, item in enumerate(corrections)}
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
) -> list[dict]:
    """Run an LLM pipeline (correct or translate) on segments.

    Handles manual/ollama/api modes, optional merge, and progress callbacks.
    """
    if mode == "manual":
        orig = original_segments if original_segments is not None else segments
        return validate_manual(segments, orig)

    if merge:
        segments = merge_consecutive_segments(segments, max_gap=max_gap)
        logger.info(f"After merge: {len(segments)} segments")

    if mode == "ollama":
        return run_ollama(
            segments,
            system_prompt,
            model=model or "qwen3:4b",
            batch_minutes=batch_minutes,
            instruction=instruction,
            label=label,
            on_batch=on_batch,
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
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Choose from 'manual', 'ollama', 'api'."
        )
