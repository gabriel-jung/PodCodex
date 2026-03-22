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

    @classmethod
    def from_stem(
        cls,
        stem: str,
        output_dir: str | Path,
        nodiar: bool = False,
    ) -> Self:
        """Create paths for a transcript-only episode (no audio file).

        Resolves all paths from ``output_dir / stem`` without requiring an
        audio file to exist.
        """
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        base = root / stem
        # Use a dummy audio_path — synthesis methods will check for existence
        return cls(audio_path=root / f"{stem}.audio", base=base, nodiar=nodiar)

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
    except Exception:
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
) -> Path:
    """Write a segment list to a JSON file with standard formatting.

    Args:
        path     : output file path
        segments : list of segment dicts (already cleaned)
        label    : human-readable label for the log message (e.g. "Polished transcript")

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.success(f"{label} saved — {len(segments)} segments → {path.name}")
    return path


def batch_segments_by_duration(
    segments: list[dict], batch_minutes: float = 15.0
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
        header = (
            f"[{seg['start']:.3f}s - {seg['end']:.3f}s] {seg.get('speaker', 'UNKNOWN')}"
        )
        text = seg.get(text_field) or "[empty]"
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def merge_consecutive_segments(
    segments: list[dict], max_gap: float = 10.0
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
        entry = {
            "speaker": speaker,
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "text": str(seg.get("text", "")).strip(),
        }
        if (
            result
            and result[-1]["speaker"] == entry["speaker"]
            and entry["start"] - result[-1]["end"] <= max_gap
        ):
            result[-1]["end"] = entry["end"]
            result[-1]["text"] += " " + entry["text"]
        else:
            if result and entry["start"] - result[-1]["end"] > max_gap:
                result.append(
                    {
                        "speaker": "[BREAK]",
                        "start": result[-1]["end"],
                        "end": entry["start"],
                        "text": "",
                    }
                )
            result.append(entry)
    n_breaks = sum(1 for s in result if s["speaker"] == "[BREAK]")
    logger.debug(
        f"merge_consecutive_segments: {n_input} → {len(result)} segments "
        f"({n_breaks} breaks, max_gap={max_gap}s)"
    )
    return result


# ──────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────


def call_and_parse(
    batch: list[dict],
    system_prompt: str,
    call_fn,
    instruction: str = "Process",
    min_length_ratio: float = 0.7,
) -> list[dict]:
    """Call the LLM for one batch and parse the response.

    Builds the user message, calls ``call_fn(messages) -> str``, strips
    markdown/thinking tags, and merges corrections back into the original
    segments.  Segments truncated below *min_length_ratio* are kept as-is.
    """
    user_content = "\n\n".join(f"[{i}] {seg['text']}" for i, seg in enumerate(batch))
    user_content += f"\n\n{instruction} all {len(batch)} numbered segments above."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw = call_fn(messages)
    logger.debug(f"LLM response: {len(raw)} chars")
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        by_index = {item.get("index", i): item for i, item in enumerate(parsed)}
        logger.debug(f"Parsed {len(parsed)} items from LLM response")
    except Exception as e:
        logger.warning(f"Parse error: {e} — batch will keep original text")
        logger.debug(f"Raw response (first 500 chars): {raw[:500]}")
        by_index = {}

    results = []
    changed = 0
    for i, seg in enumerate(batch):
        item = by_index.get(i, {})
        original_text = seg["text"]
        corrected_text = item.get("text", original_text)

        if (
            min_length_ratio
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
        results.append(entry)

    logger.debug(f"Batch: {changed}/{len(batch)} segments modified")
    return results


def run_ollama(
    segments: list[dict],
    system_prompt: str,
    model: str,
    batch_size: int,
    instruction: str = "Process",
    min_length_ratio: float = 0.7,
    label: str = "",
) -> list[dict]:
    """Run segments through a local Ollama model."""
    from ollama import Client

    client = Client()
    results = []
    n_batches = -(-len(segments) // batch_size)

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        logger.info(
            f"{label} batch {i // batch_size + 1}/{n_batches} via Ollama ({model})"
        )

        def call_fn(messages):
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": 0},
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

    return results


def run_api(
    segments: list[dict],
    system_prompt: str,
    model: str,
    api_base_url: str,
    api_key: str | None,
    batch_size: int,
    provider: str | None = None,
    instruction: str = "Process",
    min_length_ratio: float = 0.7,
    label: str = "",
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
    n_batches = -(-len(segments) // batch_size)

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        logger.info(
            f"{label} batch {i // batch_size + 1}/{n_batches} via API ({model})"
        )

        def call_fn(messages):
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0
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

    return results


def validate_manual(
    corrections: list[dict], original_segments: list[dict]
) -> list[dict]:
    """Merge LLM-returned corrections with original source segments.

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

    if len(corrections) != len(original_segments):
        logger.warning(
            f"Correction count mismatch: {len(corrections)} corrections "
            f"vs {len(original_segments)} original segments"
        )

    by_index = {item.get("index", i): item for i, item in enumerate(corrections)}

    results = []
    for i, seg in enumerate(original_segments):
        item = by_index.get(i, {})
        corrected = item.get("text", seg["text"])
        if not corrected:
            logger.warning(f"Segment [{i}] has no corrected text — keeping original")
            corrected = seg["text"]
        entry = {**seg, "text": corrected}
        entry.pop("index", None)
        results.append(entry)

    logger.info(f"Manual corrections validated — {len(results)} segments")
    return results
