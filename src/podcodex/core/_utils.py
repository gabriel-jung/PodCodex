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


@dataclass
class AudioPaths:
    """All derived file paths for a given audio file.

    Centralises path logic for the entire pipeline (transcribe, polish,
    translate, synthesize).  Create via the ``from_audio`` classmethod::

        p = AudioPaths.from_audio("episode.mp3")
        p.transcript        # → …/episode/episode.transcript.json
        p.polished_raw      # → …/episode/episode.polished.raw.json
        p.translation("en") # → …/episode/episode.en.json
        p.synthesized       # → …/episode/episode.synthesized.wav
    """

    audio_path: Path  # resolved source audio file
    base: Path  # output_root / stem — no extension
    nodiar: bool = False  # True when diarization was skipped

    def _suffix(self, name: str) -> str:
        """Insert '.nodiar' before the logical suffix when diarization is skipped."""
        return f".nodiar.{name}" if self.nodiar else f".{name}"

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
        audio_path: str | Path,
        output_dir: str | Path | None = None,
        nodiar: bool = False,
    ) -> Self:
        audio_path = Path(audio_path)
        root = cls.output_dir(audio_path, output_dir)
        base = root / audio_path.stem
        base.parent.mkdir(parents=True, exist_ok=True)
        return cls(audio_path=audio_path, base=base, nodiar=nodiar)

    # — RAG —

    @property
    def show_dir(self) -> Path:
        """Show-level directory (parent of the episode output dir)."""
        return self.base.parent.parent

    @property
    def vectors_db(self) -> Path:
        """Show-level SQLite vector store."""
        return self.show_dir / "vectors.db"

    # — Transcription —

    @property
    def segments(self) -> Path:
        return self.base.with_suffix(".segments.parquet")

    @property
    def segments_meta(self) -> Path:
        return self.base.with_suffix(".segments.meta.json")

    @property
    def diarization(self) -> Path:
        return self.base.with_suffix(".diarization.parquet")

    @property
    def diarization_meta(self) -> Path:
        return self.base.with_suffix(".diarization.meta.json")

    @property
    def diarized_segments(self) -> Path:
        return self.base.with_suffix(".diarized_segments.parquet")

    @property
    def speaker_map(self) -> Path:
        return self.base.with_suffix(".speaker_map.json")

    @property
    def transcript_raw(self) -> Path:
        return self.base.with_suffix(self._suffix("transcript.raw.json"))

    @property
    def transcript(self) -> Path:
        return self.base.with_suffix(self._suffix("transcript.json"))

    @property
    def transcript_best(self) -> Path:
        """Validated transcript if it exists, else raw."""
        return self.transcript if self.transcript.exists() else self.transcript_raw

    # — Polish —

    @property
    def polished(self) -> Path:
        return self.base.with_suffix(self._suffix("polished.json"))

    @property
    def polished_raw(self) -> Path:
        return self.base.with_suffix(self._suffix("polished.raw.json"))

    @property
    def polished_best(self) -> Path:
        """Validated polished if it exists, else raw."""
        return self.polished if self.polished.exists() else self.polished_raw

    def has_polished(self) -> bool:
        """True if either validated or raw polished file exists."""
        return self.polished.exists() or self.polished_raw.exists()

    def has_raw_polished(self) -> bool:
        """True if polished.raw.json exists but polished.json (validated) does not."""
        return self.polished_raw.exists() and not self.polished.exists()

    def has_validated_polished(self) -> bool:
        """True if the validated polished.json exists."""
        return self.polished.exists()

    def polished_raw_exists(self) -> bool:
        """True if polished.raw.json exists (regardless of validated state)."""
        return self.polished_raw.exists()

    # — Translation —

    def translation(self, lang: str) -> Path:
        lang = lang.lower().strip().replace(" ", "_")
        prefix = "nodiar." if self.nodiar else ""
        return self.base.parent / f"{self.base.name}.{prefix}{lang}.json"

    def translation_raw(self, lang: str) -> Path:
        lang = lang.lower().strip().replace(" ", "_")
        prefix = "nodiar." if self.nodiar else ""
        return self.base.parent / f"{self.base.name}.{prefix}{lang}.raw.json"

    def translation_best(self, lang: str) -> Path:
        """Validated translation if it exists, else raw."""
        v = self.translation(lang)
        return v if v.exists() else self.translation_raw(lang)

    def has_translation(self, lang: str) -> bool:
        """True if either validated or raw translation file exists."""
        return self.translation(lang).exists() or self.translation_raw(lang).exists()

    def has_raw_translation(self, lang: str) -> bool:
        """True if {lang}.raw.json exists but {lang}.json (validated) does not."""
        return (
            self.translation_raw(lang).exists() and not self.translation(lang).exists()
        )

    def has_validated_translation(self, lang: str) -> bool:
        """True if the validated {lang}.json exists."""
        return self.translation(lang).exists()

    def translation_raw_exists(self, lang: str) -> bool:
        """True if {lang}.raw.json exists (regardless of validated state)."""
        return self.translation_raw(lang).exists()

    # — Synthesis —

    @property
    def voice_samples_dir(self) -> Path:
        d = self.base.parent / "voice_samples"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def tts_segments_dir(self) -> Path:
        d = self.base.parent / "tts_segments"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def synthesized(self) -> Path:
        return self.base.with_suffix(".synthesized.wav")


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────


# LLM API provider presets: name → (base_url, env_var for key, default model).
API_PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": (
        "https://api.anthropic.com/v1/",
        "ANTHROPIC_API_KEY",
        "claude-sonnet-4-20250514",
    ),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", "mistral-small-latest"),
    "groq": (
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "llama-3.3-70b-versatile",
    ),
}


# Speaker labels that don't represent a real person (unresolved diarization placeholders).
# Used by transcribe.py (filtering) and synthesize.py (voice sample extraction).
UNKNOWN_SPEAKERS = frozenset({"UNKNOWN", "UNK", "None", "none"})

# Default speaker label when diarization is skipped.
NARRATOR_SPEAKER = "Narrator"

# Segment inserted by merge_consecutive_segments when gap > max_gap.
BREAK_SPEAKER = "[BREAK]"

# Audio sample rate used by Whisper / TTS pipeline (16 kHz mono).
SAMPLE_RATE = 16000

# Default time-based thresholds shared across pipeline modules.
DEFAULT_MAX_GAP = 10.0
DEFAULT_BATCH_MINUTES = 15.0

# LLM temperature for deterministic output in polish / translate.
DEFAULT_TEMPERATURE = 0


# Internal file suffixes that are never translation language names.
# Used by translate.py (to detect languages) and ingest/folder.py (to scan episodes).
INTERNAL_SUFFIXES = frozenset(
    {
        "transcript",
        "transcript.raw",
        "nodiar.transcript",
        "nodiar.transcript.raw",
        "nodiar.polished",
        "nodiar.polished.raw",
        "polished",
        "polished.raw",
        "words",
        "diar",
        "assigned",
        "speaker_map",
        "imported",
        "segments.meta",
        "diarization.meta",
    }
)


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


def wav_duration(path: Path) -> float:
    """Return WAV duration in seconds, or 0.0 on error."""
    import soundfile as sf

    try:
        return sf.info(str(path)).duration
    except (OSError, RuntimeError):
        return 0.0


def free_vram(model) -> None:
    """Release VRAM after model use."""
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def save_segments_json(
    path: Path,
    segments: list[dict],
    label: str,
    provenance: dict | None = None,
) -> Path:
    """Write a segment list to a JSON file with standard formatting.

    Args:
        path       : output file path
        segments   : list of segment dicts (already cleaned)
        label      : human-readable label for the log message (e.g. "Polished transcript")
        provenance : optional version metadata dict with keys: step, type, model,
                     params, manual_edit.  When provided, the segments
                     are also archived in the .versions directory.

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.success(f"{label} saved — {len(segments)} segments → {path.name}")

    if provenance and "base" in provenance:
        from podcodex.core.versions import maybe_archive

        maybe_archive(Path(provenance["base"]), segments, provenance, path.name)

    return path


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


def merge_consecutive_segments(
    segments: list[dict], max_gap: float = DEFAULT_MAX_GAP
) -> list[dict]:
    """
    Merge consecutive segments from the same speaker into single entries.
    Segments are only merged if the gap between them is <= max_gap seconds,
    preventing merges across music breaks or long silences.

    Args:
        segments : raw diarized segments
        max_gap  : maximum silence gap (seconds) to merge across (default 10s);
                   0 disables merging

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
            # With timestamps: merge only if gap <= max_gap
            # Without timestamps: always merge consecutive same-speaker
            if has_times and "start" in prev:
                gap = entry["start"] - prev["end"]
                if gap <= max_gap:
                    prev["end"] = entry["end"]
                    prev["text"] += " " + entry["text"]
                else:
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
        f"({n_breaks} breaks, max_gap={max_gap}s)"
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
    """Run segments through a local Ollama model."""
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
    """Run segments through an OpenAI-compatible API."""
    import os

    from openai import OpenAI

    if provider and provider in API_PROVIDERS:
        base_url, env_var, default_model = API_PROVIDERS[provider]
        api_base_url = api_base_url or base_url
        model = model or default_model
        api_key = api_key or os.environ.get(env_var)

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
