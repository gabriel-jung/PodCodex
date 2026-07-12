"""
podcodex.core.synthesize — Voice synthesis pipeline using Qwen3-TTS.

Steps:
    1. extract_selected_samples() — extract user-chosen clips for voice cloning
    2. generate_segment()      — generate TTS audio per translated segment
                                 (driven incrementally by synthesize_job.run_generate)
    3. assemble_episode()      — merge all segments into a final podcast audio file

Files produced in output_dir:
    voice_samples/{speaker}.wav            — reference clips for voice cloning
    tts_segments/{start_ms}_{end_ms}.wav   — generated audio per segment
    tts_segments/manifest.json             — generation metadata for incremental re-runs
    synthesize/{version_id}.wav            — final merged podcast (versioned)
"""

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from loguru import logger

from podcodex.core._ffmpeg import ffmpeg_exe
from podcodex.core._utils import (
    SAMPLE_RATE,
    AudioPaths,
    tts_segment_filename,
    wav_duration,
)
from podcodex.core.constants import AssembleStrategy


# Qwen3-TTS validates input languages against this lowercase set and rejects
# everything else (incl. ISO 639-1 codes like "en"/"fr"). Convert frontend or
# RSS feed values to the expected English noun before handing off.
_QWEN_LANG_ALIASES: dict[str, str] = {
    "en": "English",
    "eng": "English",
    "english": "English",
    "fr": "French",
    "fra": "French",
    "fre": "French",
    "french": "French",
    "français": "French",
    "francais": "French",
    "de": "German",
    "deu": "German",
    "ger": "German",
    "german": "German",
    "deutsch": "German",
    "es": "Spanish",
    "spa": "Spanish",
    "spanish": "Spanish",
    "español": "Spanish",
    "espanol": "Spanish",
    "it": "Italian",
    "ita": "Italian",
    "italian": "Italian",
    "italiano": "Italian",
    "pt": "Portuguese",
    "por": "Portuguese",
    "portuguese": "Portuguese",
    "português": "Portuguese",
    "portugues": "Portuguese",
    "ru": "Russian",
    "rus": "Russian",
    "russian": "Russian",
    "русский": "Russian",
    "ja": "Japanese",
    "jpn": "Japanese",
    "japanese": "Japanese",
    "日本語": "Japanese",
    "ko": "Korean",
    "kor": "Korean",
    "korean": "Korean",
    "한국어": "Korean",
    "zh": "Chinese",
    "zho": "Chinese",
    "chi": "Chinese",
    "chinese": "Chinese",
    "中文": "Chinese",
    "auto": "auto",
}


def _normalize_qwen_language(lang: str) -> str:
    """Map ISO codes / native names / casings to Qwen3-TTS-accepted form.

    Raises a friendly ValueError instead of qwen_tts's opaque list dump when
    the language isn't supported.
    """
    if not lang:
        return "auto"
    key = lang.strip().lower()
    if key in _QWEN_LANG_ALIASES:
        return _QWEN_LANG_ALIASES[key]
    raise ValueError(
        f"Voice synthesis doesn't support language {lang!r}. "
        f"Supported: {sorted(set(v for v in _QWEN_LANG_ALIASES.values() if v != 'auto'))}."
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
# STEP 1 — Voice sample extraction
# ──────────────────────────────────────────────


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
    from podcodex.core._utils import fill_narrator_speaker

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    samples_dir = p.ensure_voice_samples_dir()

    # Empty / placeholder labels (subtitle-imported segments without `<v>`
    # tags carry speaker="") must collapse to NARRATOR_SPEAKER so the on-disk
    # filename matches what ``load_voice_samples`` later globs for in the UI.
    by_speaker: dict[str, list[dict]] = {}
    for sel in fill_narrator_speaker(selections):
        speaker = sel["speaker"]
        seg = {**sel, "duration": sel["end"] - sel["start"]}
        by_speaker.setdefault(speaker, []).append(seg)

    plan: list[tuple[str, dict, Path]] = []
    for speaker, segs in by_speaker.items():
        for i, seg in enumerate(segs):
            plan.append((speaker, seg, samples_dir / f"{speaker}_{i:02d}.wav"))

    # Clear old samples for these speakers. Preserve uploaded files
    # (suffixed with ``_custom_``) so a Re-extract doesn't wipe the user's
    # manual uploads — same filter ``load_voice_samples`` applies.
    for speaker in by_speaker:
        for old in samples_dir.glob(f"{speaker}_*.wav"):
            if "_custom_" in old.name:
                continue
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

    # qwen_tts.core.tokenizer_25hz.vq.whisper_encoder prints a multi-line
    # "flash-attn is not installed" banner at import time. It is harmless
    # (the encoder falls back to plain PyTorch attention) and we pin
    # attn_implementation=sdpa below to avoid the flash path entirely.
    with contextlib.redirect_stdout(io.StringIO()):
        from qwen_tts import Qwen3TTSModel

    from podcodex.core._hf_logging import timed_load
    from podcodex.core.cache import get_hf_cache_dir
    from podcodex.core.device import device_str, torch_dtype

    get_hf_cache_dir()  # ensure HF_HOME is set; qwen_tts internals use HF_HUB_CACHE
    device = device_str()
    dtype = torch_dtype()

    _patch_sdpa_mask_for_mimi_vmap_bug()

    with timed_load(f"Qwen3-TTS {model_size} on {device} ({dtype})"):
        model = Qwen3TTSModel.from_pretrained(
            f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base",
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
    return model


_SDPA_MASK_PATCHED = False


def _patch_sdpa_mask_for_mimi_vmap_bug() -> None:
    """Replace transformers' vmap-based mask builder with a broadcast one.

    transformers 4.57.3's ``sdpa_mask_recent_torch`` (and ``eager_mask``,
    which internally delegates to it) builds the 4D causal mask by
    composing per-cell ``mask_function`` calls under ``torch.vmap``
    (``masking_utils.py:392``). MiMi's encoder feeds in a
    ``packed_sequence_mask`` whose inner mask_function indexes a 2D tensor
    by scalar tensor indices and compares the results. Under vmap that
    path triggers ``.item()`` internally, which vmap can't trace, raising
    ``RuntimeError: vmap: ... .item() ...`` on CPU.

    Swap ``_vmap_for_bhqkv`` for a no-vmap implementation that broadcasts
    the four index aranges to a single ``(B, H, Q, KV)`` shape and calls
    the mask_function exactly once. All shipping mask_functions (causal,
    padding, packed_sequence, sliding/chunked window, offsets, and_masks,
    or_masks) are already pure tensor ops that broadcast cleanly, so the
    result is identical to the vmap'd version. The small memory bump
    (materialising the full 4D index grid) is negligible at MiMi's 12 Hz
    frame rate.

    Called only from ``load_tts_model`` (the synth subprocess entry) on
    purpose: ``bootstrap.py:_install_transformers_torch_check_patch`` runs
    in every subprocess and pins sdpa to ``sdpa_mask_recent_torch`` to
    dodge a *different* vmap NameError in Pplx's ``or_masks`` path.
    Replacing ``_vmap_for_bhqkv`` globally would still produce correct
    masks for that path (broadcasting handles ``or_masks`` the same way),
    but we keep the scope narrow to avoid affecting unrelated subprocesses
    until needed. Idempotent for the lifetime of the Python (sub)process.
    """
    global _SDPA_MASK_PATCHED
    if _SDPA_MASK_PATCHED:
        return
    import transformers.masking_utils as _mu

    def _no_vmap_for_bhqkv(mask_function: Any, bh_indices: bool = True) -> Any:
        def wrapped(batch_arange, head_arange, q_arange, kv_arange):
            if bh_indices:
                b = batch_arange[:, None, None, None]
                h = head_arange[None, :, None, None]
                q = q_arange[None, None, :, None]
                kv = kv_arange[None, None, None, :]
            else:
                b = batch_arange  # caller passes None when bh_indices=False
                h = head_arange
                q = q_arange[:, None]
                kv = kv_arange[None, :]
            return mask_function(b, h, q, kv)

        return wrapped

    _mu._vmap_for_bhqkv = _no_vmap_for_bhqkv
    _SDPA_MASK_PATCHED = True


def build_clone_prompts(
    model: Any,
    voice_samples: dict[str, list[dict]],
    sample_index: dict[str, int] | int = 0,
) -> dict[str, object]:
    """
    Precompute voice clone prompts for each speaker.

    Args:
        model        : loaded Qwen3TTSModel from load_tts_model()
        voice_samples: output of extract_selected_samples() / load_voice_samples()
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
    cancelled: Callable[[], bool] | None = None,
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
    qwen_language = _normalize_qwen_language(language)
    for i, chunk in enumerate(chunks):
        # Cancel between chunks: a 30-60s segment can dominate wall-clock cost.
        if cancelled and cancelled():
            return None
        wavs, chunk_sr = model.generate_voice_clone(
            text=chunk,
            language=qwen_language,
            voice_clone_prompt=clone_prompts[speaker],
            instruct=instruct or None,
        )
        audio_parts.append(wavs[0])
        sr = chunk_sr
        if on_chunk:
            on_chunk(i + 1, n_chunks)
    if cancelled and cancelled():
        return None

    audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
    sf.write(str(output_path), audio, sr)
    gen_duration = len(audio) / sr
    logger.debug(
        f"Generated {output_path.name} — {gen_duration:.1f}s audio from {duration:.1f}s source"
    )
    return {**seg, "audio_file": output_path, "sample_rate": sr}


# ──────────────────────────────────────────────
# STEP 3 — Assembly
# ──────────────────────────────────────────────


def assemble_episode(
    generated: list[dict],
    output_path: Path,
    strategy: AssembleStrategy = "silence",
    silence_duration: float = 0.5,
) -> Path:
    """
    Assemble generated TTS segments into a final episode audio file.

    Strategies:
        silence          : concatenate segments with a fixed silence between each
        original_timing  : respect original timestamps, insert exact silences to
                           preserve the rhythm of the original podcast

    Args:
        generated        : segment dicts with "audio_file" set (from generate_segment)
        output_path      : destination .wav file (parent dir must exist)
        strategy         : assembly strategy
        silence_duration : silence in seconds between segments (strategy="silence" only)

    Returns:
        Path to the final .wav file
    """

    logger.info(f"Assembling {len(generated)} segments — strategy={strategy}")
    out_path = output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not generated:
        raise ValueError("No generated segments to assemble.")

    sr = generated[0]["sample_rate"]
    chunks = []

    if strategy == "silence":
        # Speaker-aware pause: short within a turn, longer at speaker changes.
        # Approximates natural conversational rhythm without the cumulative
        # bloat a single fixed gap produces.
        within_pause = max(silence_duration * 0.4, 0.05)
        across_pause = silence_duration
        within_silence = np.zeros(int(within_pause * sr), dtype=np.float32)
        across_silence = np.zeros(int(across_pause * sr), dtype=np.float32)
        for i, seg in enumerate(generated):
            audio, _ = sf.read(str(seg["audio_file"]), dtype="float32")
            chunks.append(audio)
            if i < len(generated) - 1:
                next_speaker = generated[i + 1].get("speaker") or ""
                same_speaker = (seg.get("speaker") or "") == next_speaker
                chunks.append(within_silence if same_speaker else across_silence)

    elif strategy == "original_timing":
        # Anchor at the first selected segment's start so a narrowed
        # selection (e.g. only segments 12-14 of an episode) doesn't open
        # with a long blank lead-in equal to the first start time. Within
        # the selection, inter-segment gaps still reflect the original
        # podcast's rhythm.
        cursor = generated[0]["start"]
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
        files = sorted(samples_dir.glob(f"{speaker}_*.wav"))
        if not files:
            speaker_id = reverse_map.get(speaker)
            if speaker_id:
                files = sorted(samples_dir.glob(f"{speaker_id}_*.wav"))
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
    for seg in segments:
        filename = tts_segment_filename(seg)
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
