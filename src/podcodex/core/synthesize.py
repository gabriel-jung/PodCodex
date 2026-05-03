"""
podcodex.core.synthesize — Voice synthesis pipeline using Qwen3-TTS.

Steps:
    1. extract_voice_samples() — extract audio clips per speaker for voice cloning
    2. generate_segments()     — generate TTS audio for each translated segment
    3. assemble_episode()      — merge all segments into a final podcast audio file

Files produced in output_dir:
    voice_samples/{speaker}.wav         — reference clips for voice cloning
    tts_segments/{index:04d}_{speaker}.wav  — generated audio per segment
    tts_segments/manifest.json          — generation metadata for incremental re-runs
    {stem}.synthesized.wav              — final merged podcast
"""

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from loguru import logger

from podcodex.core._ffmpeg import ffmpeg_exe
from podcodex.core._utils import (
    SAMPLE_RATE,
    UNKNOWN_SPEAKERS,
    AudioPaths,
    group_by_speaker,
    wav_duration,
)


# ──────────────────────────────────────────────
# Generation manifest — tracks what produced each segment
# ──────────────────────────────────────────────


def _text_hash(text: str) -> str:
    """Return a truncated SHA-256 hash of segment text for change detection."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _sample_key(
    voice_samples: dict[str, list[dict]],
    speaker: str,
    sample_index: dict[str, int] | int = 0,
) -> str:
    """Return the filename of the voice sample selected for a speaker.

    Args:
        voice_samples: mapping of speaker to their extracted sample dicts.
        speaker: speaker label to look up.
        sample_index: which sample to use — int (global) or dict per speaker.

    Returns:
        Filename string of the selected sample, or ``""`` if no samples exist.
    """
    samples = voice_samples.get(speaker, [])
    if not samples:
        return ""
    idx = (
        sample_index.get(speaker, 0) if isinstance(sample_index, dict) else sample_index
    )
    idx = min(idx, len(samples) - 1)
    return Path(samples[idx]["file"]).name


def load_manifest(segments_dir: Path) -> dict:
    """Load the generation manifest from disk.

    Args:
        segments_dir: directory containing ``manifest.json``.

    Returns:
        Parsed manifest dict, or an empty structure
        ``{"model": None, "language": None, "segments": {}}`` if the file
        is missing or corrupt.
    """
    manifest_path = segments_dir / "manifest.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt manifest.json — will regenerate all segments")
    return {"model": None, "language": None, "segments": {}}


def save_manifest(segments_dir: Path, manifest: dict) -> None:
    """Write the generation manifest to disk.

    Args:
        segments_dir: directory where ``manifest.json`` will be written.
        manifest: manifest dict containing model, language, and per-segment entries.
    """
    from podcodex.core._utils import atomic_write

    def _write(p: Path) -> None:
        with p.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

    atomic_write(segments_dir / "manifest.json", _write, suffix=".json")


def segment_is_current(
    manifest: dict,
    filename: str,
    text: str,
    speaker: str,
    voice_sample_name: str,
    model_size: str,
    language: str,
) -> bool:
    """Check if a previously generated segment is still valid.

    A segment is valid only if ALL of these match:

    - The WAV file exists (checked by caller)
    - Run-level settings (model, language) match
    - Segment text hasn't changed (hash match)
    - Same voice sample was used for this speaker

    Args:
        manifest: loaded manifest dict from :func:`load_manifest`.
        filename: WAV filename key in the manifest (e.g. ``"0001_Alice.wav"``).
        text: current segment text to compare against stored hash.
        speaker: speaker label (used only for logging context).
        voice_sample_name: filename of the voice sample that would be used now.
        model_size: TTS model size (``"0.6B"`` or ``"1.7B"``).
        language: target language string.

    Returns:
        ``True`` if the existing segment can be reused, ``False`` otherwise.
    """
    if manifest.get("model") != model_size or manifest.get("language") != language:
        return False
    entry = manifest.get("segments", {}).get(filename)
    if not entry:
        return False
    return (
        entry.get("text_hash") == _text_hash(text)
        and entry.get("voice_sample") == voice_sample_name
    )


# ──────────────────────────────────────────────
# Hallucination detection
# ──────────────────────────────────────────────

# Patterns Whisper commonly generates on music/silence instead of real speech.
_HALLUCINATION_RE = re.compile(
    r"(?i)"
    r"sous-titrag"  # "Sous-titrage FR", "Sous-titrage MFP", etc.
    r"|subtitl"  # English equivalents
    r"|merci d'avoir"  # filler sign-off
    r"|transcription\s+réalis"
    r"|www\."  # URLs hallucinated on silence
)


def is_hallucination(text: str) -> bool:
    """Return True if *text* looks like a Whisper hallucination rather than real speech.

    Catches the most common artifacts produced on music or silence:
    repeated punctuation (``... ...``), very short strings, and
    known French/English subtitle watermarks.

    Args:
        text: segment text to evaluate.

    Returns:
        ``True`` if the text matches known hallucination patterns, ``False``
        for empty strings or genuine speech.
    """
    t = text.strip()
    if not t:
        return False
    # Only punctuation / dots / ellipses
    if re.fullmatch(r"[\s.…\-_,;:!?/|]+", t):
        return True
    # Very short (single word or less, likely a stutter artifact)
    if len(t) <= 3:
        return True
    # Known watermark / filler patterns
    if _HALLUCINATION_RE.search(t):
        return True
    return False


# ──────────────────────────────────────────────
# STEP 1 — Voice sample extraction
# ──────────────────────────────────────────────


def _select_candidates(
    segs: list[dict],
    speaker: str,
    *,
    min_duration: float | None = None,
    max_duration: float | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Pick the best voice-cloning candidates for one speaker.

    Filters by duration range and hallucination detection, then returns
    up to *top_k* segments sorted by duration descending.

    Args:
        segs: all segments for this speaker, each with a ``duration`` key.
        speaker: speaker label (used for log messages).
        min_duration: minimum clip duration in seconds (``None`` to skip).
        max_duration: maximum clip duration in seconds (``None`` to skip).
        top_k: maximum number of candidates to return.

    Returns:
        Up to *top_k* candidate dicts sorted by duration descending,
        or an empty list if no usable candidates remain.
    """
    candidates = segs

    # Duration filter (fall back to all segments if nothing matches)
    if min_duration is not None or max_duration is not None:
        filtered = candidates
        if min_duration is not None:
            filtered = [s for s in filtered if s["duration"] >= min_duration]
        if max_duration is not None:
            filtered = [s for s in filtered if s["duration"] <= max_duration]
        if filtered:
            candidates = filtered
        else:
            logger.warning(
                f"No segments in duration range for {speaker} — using all segments"
            )

    # Hallucination filter
    clean = [s for s in candidates if not is_hallucination(s.get("text", ""))]
    dropped = len(candidates) - len(clean)
    if dropped:
        logger.warning(f"Dropped {dropped} hallucinated-text segment(s) for {speaker}")
    if not clean:
        logger.warning(
            f"All candidates for {speaker} are music/hallucination — skipping"
        )
        return []

    return sorted(clean, key=lambda s: s["duration"], reverse=True)[:top_k]


def _extract_clip(audio_path: Path, seg: dict, output_path: Path) -> dict:
    """Extract a single audio clip via ffmpeg, resampled to 16 kHz mono WAV.

    Args:
        audio_path: source audio file.
        seg: segment dict with ``start``, ``end``, ``duration``, and ``text`` keys.
        output_path: destination path for the extracted WAV clip.

    Returns:
        Dict with ``file``, ``start``, ``end``, ``duration``, and ``text`` fields.

    Raises:
        subprocess.CalledProcessError: if ffmpeg exits with a non-zero status.
    """
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            str(seg["start"]),
            "-to",
            str(seg["end"]),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return {
        "file": output_path,
        "start": seg["start"],
        "end": seg["end"],
        "duration": seg["duration"],
        "text": seg["text"],
    }


def extract_voice_samples(
    audio_path: Path | str,
    segments: list[dict],
    output_dir: str | Path | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """
    Extract audio clips per speaker for voice cloning.

    For each speaker, selects up to top_k segments sorted by duration descending.
    Optionally filtered by min_duration and max_duration.
    Clips are saved as 16kHz mono WAV in output_dir/voice_samples/.

    Args:
        audio_path   : source audio file
        segments     : output of merge_consecutive_segments()
        output_dir   : directory relative to audio_path for outputs
        min_duration : minimum clip duration in seconds (optional)
        max_duration : maximum clip duration in seconds (optional)
        top_k        : max number of candidates per speaker

    Returns:
        {speaker: [{"file", "start", "end", "duration", "text"}, ...]}
        sorted by duration descending
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    logger.info(
        f"Extracting voice samples from {p.audio_path.name} — {len(segments)} segments, top_k={top_k}"
    )
    samples_dir = p.voice_samples_dir

    # Group by speaker, add duration, select candidates, build extraction plan
    by_speaker = {
        speaker: [{**seg, "duration": seg["end"] - seg["start"]} for seg in segs]
        for speaker, segs in group_by_speaker(segments).items()
    }
    logger.debug(f"Found {len(by_speaker)} distinct speaker labels")

    plan: list[tuple[str, dict, Path]] = []
    for speaker, segs in by_speaker.items():
        if not speaker or speaker in UNKNOWN_SPEAKERS:
            logger.warning(
                f"Skipping speaker {speaker!r} — not a real speaker ({len(segs)} segments)"
            )
            continue
        candidates = _select_candidates(
            segs,
            speaker,
            min_duration=min_duration,
            max_duration=max_duration,
            top_k=top_k,
        )
        for i, seg in enumerate(candidates):
            plan.append((speaker, seg, samples_dir / f"{speaker}_{i:02d}.wav"))

    # Run ffmpeg extractions in parallel (I/O-bound)
    results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(len(plan) or 1, 8)) as executor:
        futures = {
            executor.submit(_extract_clip, p.audio_path, seg, out): speaker
            for speaker, seg, out in plan
        }
        for future in as_completed(futures):
            speaker = futures[future]
            entry = future.result()
            logger.debug(f"{speaker} — {entry['duration']:.1f}s → {entry['file'].name}")
            results.setdefault(speaker, []).append(entry)

    # Restore order (sorted by duration descending)
    for speaker in results:
        results[speaker].sort(key=lambda e: e["duration"], reverse=True)

    total = sum(len(v) for v in results.values())
    logger.success(
        f"Voice samples extracted — {total} clips for {len(results)} speakers → {samples_dir.name}/"
    )
    return results


def extract_selected_samples(
    audio_path: Path | str,
    selections: list[dict],
    output_dir: str | Path | None = None,
) -> dict[str, list[dict]]:
    """Extract specific user-chosen segments as voice samples.

    Args:
        audio_path  : source audio file
        selections  : list of {speaker, start, end, text} dicts
        output_dir  : directory relative to audio_path for outputs

    Returns:
        {speaker: [{"file", "start", "end", "duration", "text"}, ...]}
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    samples_dir = p.voice_samples_dir

    # Build extraction plan grouped by speaker
    by_speaker: dict[str, list[dict]] = {}
    for sel in selections:
        speaker = sel["speaker"]
        seg = {**sel, "duration": sel["end"] - sel["start"]}
        by_speaker.setdefault(speaker, []).append(seg)

    plan: list[tuple[str, dict, Path]] = []
    for speaker, segs in by_speaker.items():
        for i, seg in enumerate(segs):
            plan.append((speaker, seg, samples_dir / f"{speaker}_{i:02d}.wav"))

    # Clear old samples for these speakers
    for speaker in by_speaker:
        for old in samples_dir.glob(f"{speaker}_*.wav"):
            old.unlink()

    results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(len(plan) or 1, 8)) as executor:
        futures = {
            executor.submit(_extract_clip, p.audio_path, seg, out): speaker
            for speaker, seg, out in plan
        }
        for future in as_completed(futures):
            speaker = futures[future]
            entry = future.result()
            results.setdefault(speaker, []).append(entry)

    for speaker in results:
        results[speaker].sort(key=lambda e: e["duration"], reverse=True)

    total = sum(len(v) for v in results.values())
    logger.success(
        f"Extracted {total} selected voice samples for {len(results)} speakers"
    )
    return results


# ──────────────────────────────────────────────
# STEP 2 — Segment generation
# ──────────────────────────────────────────────


def load_tts_model(model_size: str = "1.7B"):
    """
    Load Qwen3-TTS model.

    Args:
        model_size : "0.6B" or "1.7B"

    Returns:
        Loaded Qwen3TTSModel instance
    """
    import contextlib
    import io

    import torch

    # qwen_tts.core.tokenizer_25hz.vq.whisper_encoder prints a multi-line
    # "flash-attn is not installed" banner at import time. It is harmless
    # (the encoder falls back to plain PyTorch attention) and we pin
    # attn_implementation=sdpa below to avoid the flash path entirely.
    with contextlib.redirect_stdout(io.StringIO()):
        from qwen_tts import Qwen3TTSModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from podcodex.core._hf_logging import timed_load
    from podcodex.core.cache import get_hf_cache_dir

    with timed_load(f"Qwen3-TTS {model_size} on {device}"):
        model = Qwen3TTSModel.from_pretrained(
            f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base",
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            cache_dir=str(get_hf_cache_dir()),
        )
    return model


def build_clone_prompts(
    model: Any,
    voice_samples: dict[str, list[dict]],
    sample_index: dict[str, int] | int = 0,
) -> dict[str, object]:
    """
    Precompute voice clone prompts for each speaker.

    Args:
        model        : loaded Qwen3TTSModel from load_tts_model()
        voice_samples: output of extract_voice_samples()
        sample_index : which sample to use per speaker —
                       int (global) or dict {speaker: index}

    Returns:
        {speaker: voice_clone_prompt}
    """
    clone_prompts = {}
    for speaker, samples in voice_samples.items():
        idx = (
            sample_index.get(speaker, 0)
            if isinstance(sample_index, dict)
            else sample_index
        )
        idx = min(idx, len(samples) - 1)
        sample = samples[idx]
        clone_prompts[speaker] = model.create_voice_clone_prompt(
            ref_audio=str(sample["file"]),
            ref_text=sample["text"],
            x_vector_only_mode=True,
        )
        logger.debug(
            f"Voice prompt ready for {speaker} (sample {idx} — {sample['duration']:.1f}s)"
        )
    logger.info(f"Clone prompts built for {len(clone_prompts)} speakers")
    return clone_prompts


def _split_text(text: str, max_parts: int) -> list[str]:
    """Split text into at most *max_parts*, breaking at natural boundaries.

    Strategy:

    1. Split at sentence endings (``.`` ``!`` ``?``)
    2. If that yields fewer parts than needed, also split at commas
    3. If we now have more parts than needed, greedily group them into
       balanced chunks by character count

    Args:
        text: input text to split.
        max_parts: maximum number of parts to produce.

    Returns:
        List of at most *max_parts* strings. If there are fewer natural
        breakpoints than requested, returns what is available without
        forcing artificial mid-word splits.
    """
    text = text.strip()
    if not text or max_parts <= 1:
        return [text] if text else []

    # 1. Split at sentence boundaries
    parts = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]

    # 2. If not enough parts, also split at commas
    if len(parts) < max_parts:
        finer: list[str] = []
        for s in parts:
            finer.extend(p for p in re.split(r"(?<=,)\s+", s) if p)
        parts = finer

    # Enough natural splits or fewer — done
    if len(parts) <= max_parts:
        return parts

    # 3. Too many small parts — group into balanced chunks
    target_len = sum(len(p) for p in parts) / max_parts
    groups: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for i, part in enumerate(parts):
        buf.append(part)
        buf_len += len(part)

        parts_left = len(parts) - i - 1
        groups_left = max_parts - len(groups) - 1
        if (
            buf_len >= target_len
            and len(groups) < max_parts - 1
            and parts_left >= groups_left
        ):
            groups.append(" ".join(buf))
            buf, buf_len = [], 0

    if buf:
        groups.append(" ".join(buf))

    return groups


def generate_segment(
    model: Any,
    seg: dict,
    clone_prompts: dict[str, object],
    output_path: Path,
    language: str = "English",
    instruct: str | None = None,
    max_chunk_duration: float = 20.0,
    on_chunk: Callable[[int, int], None] | None = None,
) -> dict | None:
    """
    Generate TTS audio for a single segment.

    Segments shorter than max_chunk_duration (in source-audio seconds) are
    synthesized in a single call.  Longer segments are split into
    ceil(duration / max_chunk_duration) balanced parts at sentence boundaries,
    synthesized separately, then concatenated — this avoids quality degradation
    and slow generation on long inputs.

    Args:
        model              : loaded Qwen3TTSModel from load_tts_model()
        seg                : single segment dict with text, speaker, start, end
        clone_prompts      : output of build_clone_prompts()
        output_path        : path to save the generated WAV file
        language           : target language for TTS (must match translation target_lang)
        instruct           : optional style/intonation instruction passed directly to
                             Qwen3-TTS (e.g. "Speak slowly, whisper, enthusiastic").
                             If None or empty string, no instruct is sent.
        max_chunk_duration : source-audio seconds above which a segment is split.
                             Segments at or below this duration are synthesized whole.
        on_chunk           : optional callback(chunk_idx, n_chunks) called after each
                             chunk is generated — useful for progress reporting in UIs

    Returns:
        Segment dict with added "audio_file" and "sample_rate" fields, or None if skipped
    """
    speaker = seg["speaker"]
    text = seg.get("text", "")

    if not text:
        logger.warning(f"Segment has no text — skipping [{output_path.stem}]")
        return None

    if speaker not in clone_prompts:
        logger.warning(f"No voice prompt for {speaker} — skipping [{output_path.stem}]")
        return None

    duration = seg.get("end", 0) - seg.get("start", 0)
    n_chunks = (
        1
        if duration <= max_chunk_duration
        else math.ceil(duration / max_chunk_duration)
    )
    chunks = _split_text(text, n_chunks)
    n_chunks = len(chunks)  # actual count after splitting (may be < requested)
    if n_chunks > 1:
        logger.info(
            f"Segment split into {n_chunks} chunks ({duration:.1f}s source / {len(text)} chars)"
        )

    audio_parts = []
    sr = None
    for i, chunk in enumerate(chunks):
        wavs, chunk_sr = model.generate_voice_clone(
            text=chunk,
            language=language,
            voice_clone_prompt=clone_prompts[speaker],
            instruct=instruct or None,
        )
        audio_parts.append(wavs[0])
        sr = chunk_sr
        if on_chunk:
            on_chunk(i + 1, n_chunks)

    audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
    sf.write(str(output_path), audio, sr)
    gen_duration = len(audio) / sr
    logger.debug(
        f"Generated {output_path.name} — {gen_duration:.1f}s audio from {duration:.1f}s source"
    )
    return {**seg, "audio_file": output_path, "sample_rate": sr}


def generate_segments(
    audio_path: Path | str,
    segments: list[dict],
    voice_samples: dict[str, list[dict]],
    output_dir: str | Path | None = None,
    model_size: str = "1.7B",
    language: str = "English",
    sample_index: dict[str, int] | int = 0,
    max_chunk_duration: float = 20.0,
    force: bool = False,
    only_speakers: list[str] | None = None,
) -> list[dict]:
    """
    Generate TTS audio for all translated segments using Qwen3-TTS voice cloning.

    Supports incremental generation: previously generated segments are skipped
    if their text and voice sample haven't changed (tracked via manifest.json).

    Convenience wrapper around load_tts_model + build_clone_prompts + generate_segment.

    Args:
        audio_path         : source audio file (used to resolve output_dir)
        segments           : translated segment dicts
        voice_samples      : output of extract_voice_samples()
        output_dir         : directory relative to audio_path for outputs
        model_size         : "0.6B" or "1.7B"
        language           : target language for TTS — must match translation target_lang
        sample_index       : which voice sample to use per speaker
        max_chunk_duration : source-audio seconds above which a segment is split
        force              : if True, regenerate all segments ignoring manifest
        only_speakers      : if set, only regenerate segments for these speakers

    Returns:
        List of segments with added "audio_file" and "sample_rate" fields
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    logger.info(
        f"Generating TTS for {len(segments)} segments — model={model_size}, language={language}"
    )
    segments_dir = p.tts_segments_dir

    # Load manifest for incremental generation
    manifest = (
        load_manifest(segments_dir)
        if not force
        else {"model": None, "language": None, "segments": {}}
    )

    model = load_tts_model(model_size=model_size)
    clone_prompts = build_clone_prompts(model, voice_samples, sample_index=sample_index)

    generated = []
    reused = 0
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "UNK")
        text = seg.get("text", "").strip()
        filename = f"{i:04d}_{speaker}.wav"
        output_path = segments_dir / filename

        # Skip if only regenerating specific speakers
        if only_speakers and speaker not in only_speakers:
            if output_path.exists():
                generated.append(
                    {**seg, "audio_file": output_path, "sample_rate": SAMPLE_RATE}
                )
            continue

        # Check manifest — skip if segment is still valid
        sample_name = _sample_key(voice_samples, speaker, sample_index)
        if (
            not force
            and output_path.exists()
            and segment_is_current(
                manifest, filename, text, speaker, sample_name, model_size, language
            )
        ):
            generated.append(
                {**seg, "audio_file": output_path, "sample_rate": SAMPLE_RATE}
            )
            reused += 1
            continue

        result = generate_segment(
            model,
            seg,
            clone_prompts,
            output_path,
            language=language,
            max_chunk_duration=max_chunk_duration,
        )
        if result:
            generated.append(result)
            # Update manifest entry
            manifest["segments"][filename] = {
                "speaker": speaker,
                "voice_sample": sample_name,
                "text_hash": _text_hash(text),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            logger.debug(f"[{i + 1}/{len(segments)}] {speaker}: {text[:60]}…")

    # Update run-level manifest fields and save
    manifest["model"] = model_size
    manifest["language"] = language
    save_manifest(segments_dir, manifest)

    new_count = len(generated) - reused
    logger.success(
        f"TTS generation done — {new_count} generated, {reused} reused"
        + (
            f", {len(segments) - len(generated)} skipped"
            if len(generated) < len(segments)
            else ""
        )
    )
    return generated


# ──────────────────────────────────────────────
# STEP 3 — Assembly
# ──────────────────────────────────────────────


def assemble_episode(
    generated: list[dict],
    audio_path: Path | str,
    output_dir: str | Path | None = None,
    strategy: Literal["silence", "original_timing"] = "original_timing",
    silence_duration: float = 0.5,
) -> Path:
    """
    Assemble generated TTS segments into a final episode audio file.

    Strategies:
        silence          : concatenate segments with a fixed silence between each
        original_timing  : respect original timestamps, insert exact silences to
                           preserve the rhythm of the original podcast

    Args:
        generated        : output of generate_segments()
        audio_path       : source audio file (used to resolve output path)
        output_dir       : directory relative to audio_path for outputs
        strategy         : assembly strategy
        silence_duration : silence in seconds between segments (strategy="silence" only)

    Returns:
        Path to the final .wav file
    """

    logger.info(f"Assembling {len(generated)} segments — strategy={strategy}")
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    out_path = p.synthesized

    if not generated:
        raise ValueError("No generated segments to assemble.")

    sr = generated[0]["sample_rate"]
    chunks = []

    if strategy == "silence":
        silence = np.zeros(int(silence_duration * sr), dtype=np.float32)
        for i, seg in enumerate(generated):
            audio, _ = sf.read(str(seg["audio_file"]), dtype="float32")
            chunks.append(audio)
            if i < len(generated) - 1:
                chunks.append(silence)

    elif strategy == "original_timing":
        cursor = 0.0
        for seg in generated:
            gap = seg["start"] - cursor
            if gap > 0:
                chunks.append(np.zeros(int(gap * sr), dtype=np.float32))
            audio, _ = sf.read(str(seg["audio_file"]), dtype="float32")
            chunks.append(audio)
            cursor = seg["start"] + len(audio) / sr

    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose 'silence' or 'original_timing'."
        )

    episode = np.concatenate(chunks)
    sf.write(str(out_path), episode, sr)
    duration = len(episode) / sr
    logger.success(f"Episode assembled — {duration:.1f}s → {out_path.name}")
    return out_path


# ──────────────────────────────────────────────
# Disk loaders (voice samples & generated segments)
# ──────────────────────────────────────────────


def load_voice_samples(
    output_dir: str | Path,
    speakers: list[str],
    speaker_map: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """Load previously extracted voice samples from disk.

    Args:
        output_dir   : episode output directory containing ``voice_samples/``
        speakers     : ordered list of speaker names to look for
        speaker_map  : optional {SPEAKER_XX: human_name} map for fallback matching

    Returns:
        {speaker: [{"file": Path, "duration": float, "text": ""}, ...]}
    """
    from podcodex.core._utils import VOICE_SAMPLES_DIR

    samples_dir = Path(output_dir) / VOICE_SAMPLES_DIR
    if not samples_dir.exists():
        logger.debug(f"No voice_samples/ directory in {output_dir}")
        return {}

    reverse_map = {v: k for k, v in (speaker_map or {}).items()}

    result: dict[str, list[dict]] = {}
    for speaker in speakers:
        files = sorted(
            f for f in samples_dir.glob(f"{speaker}_*.wav") if "_custom_" not in f.name
        )
        if not files:
            speaker_id = reverse_map.get(speaker)
            if speaker_id:
                files = sorted(
                    f
                    for f in samples_dir.glob(f"{speaker_id}_*.wav")
                    if "_custom_" not in f.name
                )
        if files:
            result[speaker] = [
                {"file": f, "duration": wav_duration(f), "text": ""} for f in files
            ]
    total = sum(len(v) for v in result.values())
    logger.debug(
        f"Loaded {total} voice samples for {len(result)}/{len(speakers)} speakers"
    )
    return result


def load_generated_segments(
    output_dir: str | Path,
    segments: list[dict],
) -> list[dict]:
    """Load previously generated TTS segments from disk.

    Args:
        output_dir : episode output directory containing ``tts_segments/``
        segments   : segment list (used to match filenames and merge metadata)

    Returns:
        List of segment dicts with ``audio_file`` and ``sample_rate`` fields
        for segments that have been generated.  Missing segments are omitted
        (previously this returned [] if any were missing).
    """
    from podcodex.core._utils import TTS_SEGMENTS_DIR

    segments_dir = Path(output_dir) / TTS_SEGMENTS_DIR
    if not segments_dir.exists():
        logger.debug(f"No tts_segments/ directory in {output_dir}")
        return []

    manifest = load_manifest(segments_dir)

    result = []
    missing = 0
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "UNK")
        filename = f"{i:04d}_{speaker}.wav"
        wav_path = segments_dir / filename
        if not wav_path.exists():
            missing += 1
            continue
        try:
            info = sf.info(str(wav_path))
            entry = manifest.get("segments", {}).get(filename, {})
            result.append(
                {
                    **seg,
                    "audio_file": wav_path,
                    "sample_rate": info.samplerate,
                    "voice_sample": entry.get("voice_sample", ""),
                    "generated_at": entry.get("generated_at", ""),
                }
            )
        except (OSError, RuntimeError):
            missing += 1
            continue
    logger.debug(
        f"Loaded {len(result)} generated segments from disk"
        + (f" ({missing} missing)" if missing else "")
    )
    return result
