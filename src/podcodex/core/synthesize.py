"""
podcodex.core.synthesize — Voice synthesis pipeline using Qwen3-TTS.

Steps:
    1. extract_voice_samples() — extract audio clips per speaker for voice cloning
    2. generate_segments()     — generate TTS audio for each translated segment
    3. merge_segments()        — merge all segments into a final podcast audio file

Files produced in output_dir:
    voice_samples/{speaker}.wav         — reference clips for voice cloning
    tts_segments/{index:04d}_{speaker}.wav  — generated audio per segment
    {stem}.synthesized.wav              — final merged podcast
"""

import math
import re
import numpy as np
import subprocess
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional

from loguru import logger

from podcodex.core._paths import episode_output_dir


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


def extract_voice_samples(
    audio_path: Path | str,
    segments: list[dict],
    output_dir: str | Path | None = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """
    Extract audio clips per speaker for voice cloning.

    For each speaker, selects up to top_k segments sorted by duration descending.
    Optionally filtered by min_duration and max_duration.
    Clips are saved as 16kHz mono WAV in output_dir/voice_samples/.

    Args:
        audio_path   : source audio file
        segments     : output of simplify_transcript()
        output_dir   : directory relative to audio_path for outputs
        min_duration : minimum clip duration in seconds (optional)
        max_duration : maximum clip duration in seconds (optional)
        top_k        : max number of candidates per speaker

    Returns:
        {speaker: [{"file", "start", "end", "duration", "text"}, ...]}
        sorted by duration descending
    """
    audio_path = Path(audio_path)
    samples_dir = episode_output_dir(audio_path, output_dir) / "voice_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Group segments by speaker with their duration
    by_speaker: dict[str, list[dict]] = {}
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        duration = seg["end"] - seg["start"]
        by_speaker.setdefault(speaker, []).append({**seg, "duration": duration})

    _INVALID_SPEAKERS = {None, "", "UNKNOWN", "UNK", "None", "none"}

    # Build the full extraction plan: (speaker, index, seg, output_path)
    plan: list[tuple[str, int, dict, Path]] = []
    speaker_candidate_counts: dict[str, int] = {}
    for speaker, segs in by_speaker.items():
        if speaker in _INVALID_SPEAKERS:
            logger.warning(
                f"Skipping speaker {speaker!r} — likely music or noise region ({len(segs)} segments)"
            )
            continue
        candidates = segs
        if min_duration is not None:
            candidates = [s for s in candidates if s["duration"] >= min_duration]
        if max_duration is not None:
            candidates = [s for s in candidates if s["duration"] <= max_duration]

        if not candidates:
            logger.warning(
                f"No segments in duration range for {speaker} — using all segments"
            )
            candidates = segs

        # Drop candidates whose text is a known Whisper hallucination artifact
        before = len(candidates)
        candidates = [s for s in candidates if not is_hallucination(s.get("text", ""))]
        dropped = before - len(candidates)
        if dropped:
            logger.warning(
                f"Dropped {dropped} hallucinated-text segment(s) for {speaker}"
            )

        if not candidates:
            logger.warning(
                f"All candidates for {speaker} are music/hallucination — skipping"
            )
            continue

        candidates = sorted(candidates, key=lambda s: s["duration"], reverse=True)[
            :top_k
        ]
        speaker_candidate_counts[speaker] = len(candidates)
        for i, seg in enumerate(candidates):
            plan.append((speaker, i, seg, samples_dir / f"{speaker}_{i:02d}.wav"))

    def _extract_clip(speaker: str, i: int, seg: dict, output_path: Path) -> dict:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ss",
                str(seg["start"]),
                "-to",
                str(seg["end"]),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        logger.info(
            f"{speaker} [{i + 1}/{speaker_candidate_counts[speaker]}] — {seg['duration']:.1f}s → {output_path.name}"
        )
        return {
            "speaker": speaker,
            "idx": i,
            "file": output_path,
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["duration"],
            "text": seg["text"],
        }

    # Run all ffmpeg extractions in parallel (I/O-bound, safe to thread)
    results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(len(plan), 8)) as executor:
        futures = {
            executor.submit(_extract_clip, sp, i, seg, out): (sp, i)
            for sp, i, seg, out in plan
        }
        for future in as_completed(futures):
            entry = future.result()  # raises on ffmpeg failure
            speaker = entry.pop("speaker")
            entry.pop("idx")
            results.setdefault(speaker, []).append(entry)

    # Restore per-speaker order (sorted by duration descending, matching plan order)
    for speaker in results:
        results[speaker].sort(key=lambda e: e["duration"], reverse=True)

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
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )
    logger.info(f"Qwen3-TTS {model_size} loaded")
    return model


def build_clone_prompts(
    model,
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
        logger.info(
            f"Voice prompt ready for {speaker} (sample {idx} — {sample['duration']:.1f}s)"
        )
    return clone_prompts


def _split_text(text: str, n_chunks: int) -> list[str]:
    """
    Split text into at most n_chunks parts, breaking at sentence boundaries
    (. ! ?) and falling back to commas when more pieces are needed.

    Parts are balanced by character count (a reasonable proxy for TTS duration).
    When there are fewer natural breakpoints than n_chunks, returns as many
    pieces as are available rather than forcing artificial splits.
    """
    import re

    text = text.strip()
    if not text or n_chunks <= 1:
        return [text] if text else []

    # Primary split: sentence boundaries
    pieces = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]

    # If we still need more breakpoints, also split on commas
    if len(pieces) < n_chunks:
        expanded = []
        for s in pieces:
            expanded.extend(p for p in re.split(r"(?<=,)\s+", s) if p)
        pieces = expanded

    # Fewer pieces than requested chunks — return what we have
    if len(pieces) <= n_chunks:
        return pieces

    # Greedy grouping: accumulate pieces until we reach the per-chunk target,
    # then flush, ensuring we don't run out of pieces for the remaining chunks.
    total = sum(len(p) for p in pieces)
    target = total / n_chunks

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for i, piece in enumerate(pieces):
        current.append(piece)
        current_len += len(piece)
        remaining_pieces = len(pieces) - i - 1
        remaining_chunks_needed = n_chunks - len(chunks) - 1
        if (
            current_len >= target
            and len(chunks) < n_chunks - 1
            and remaining_pieces >= remaining_chunks_needed
        ):
            chunks.append(" ".join(current))
            current, current_len = [], 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def generate_segment(
    model,
    seg: dict,
    clone_prompts: dict[str, object],
    output_path: Path,
    language: str = "English",
    instruct: str | None = None,
    max_chunk_duration: float = 20.0,
    on_chunk: object | None = None,
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
        seg                : single segment dict with text_trad, speaker, start, end
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
    text = seg.get("text_trad", "")

    if not text:
        logger.warning(f"Segment has no text_trad — skipping [{output_path.stem}]")
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
            instruct=instruct if instruct else None,
        )
        audio_parts.append(wavs[0])
        sr = chunk_sr
        if on_chunk:
            on_chunk(i + 1, n_chunks)

    audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
    sf.write(str(output_path), audio, sr)
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
) -> list[dict]:
    """
    Generate TTS audio for all translated segments using Qwen3-TTS voice cloning.

    Convenience wrapper around load_tts_model + build_clone_prompts + generate_segment.
    For segment-by-segment control (e.g. in Streamlit), use those functions directly.

    Args:
        audio_path         : source audio file (used to resolve output_dir)
        segments           : output of load_translation()
        voice_samples      : output of extract_voice_samples()
        output_dir         : directory relative to audio_path for outputs
        model_size         : "0.6B" or "1.7B"
        language           : target language for TTS — must match translation target_lang
        sample_index       : which voice sample to use per speaker
        max_chunk_duration : source-audio seconds above which a segment is split

    Returns:
        List of segments with added "audio_file" and "sample_rate" fields
    """
    audio_path = Path(audio_path)
    segments_dir = episode_output_dir(audio_path, output_dir) / "tts_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    model = load_tts_model(model_size=model_size)
    clone_prompts = build_clone_prompts(model, voice_samples, sample_index=sample_index)

    generated = []
    for i, seg in enumerate(segments):
        output_path = segments_dir / f"{i:04d}_{seg['speaker']}.wav"
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
            logger.info(
                f"[{i + 1}/{len(segments)}] {seg['speaker']}: {seg.get('text_trad', '')[:60]}..."
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

    audio_path = Path(audio_path)
    out_path = (
        episode_output_dir(audio_path, output_dir)
        / f"{audio_path.stem}.synthesized.wav"
    )

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
