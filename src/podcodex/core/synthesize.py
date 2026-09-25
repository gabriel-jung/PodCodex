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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from loguru import logger

from podcodex.core._ffmpeg import run_ffmpeg
from podcodex.core._utils import (
    SAMPLE_RATE,
    AudioPaths,
    atomic_write,
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


def _selected_sample(
    voice_samples: dict[str, list[dict]],
    speaker: str,
    sample_index: dict[str, int] | int = 0,
) -> dict | None:
    """The voice sample used as a speaker's clone reference, or None.

    The one place the choice is made, so the manifest records the sample
    the clone prompt was actually built from.

    Args:
        voice_samples: mapping of speaker to their sample dicts (see
            ``load_voice_samples`` for the order).
        speaker: speaker label to look up.
        sample_index: which sample to use: int (global) or dict per speaker.
    """
    samples = voice_samples.get(speaker, [])
    if not samples:
        return None
    idx = (
        sample_index.get(speaker, 0) if isinstance(sample_index, dict) else sample_index
    )
    return samples[min(idx, len(samples) - 1)]


def _sample_key(
    voice_samples: dict[str, list[dict]],
    speaker: str,
    sample_index: dict[str, int] | int = 0,
) -> str:
    """Filename of the speaker's clone reference, or ``""`` if none."""
    sample = _selected_sample(voice_samples, speaker, sample_index)
    return Path(sample["file"]).name if sample else ""


def empty_manifest() -> dict:
    """A manifest with no generated segments."""
    return {"segments": {}}


def load_manifest(segments_dir: Path) -> dict:
    """Load the generation manifest from disk.

    Args:
        segments_dir: directory containing ``manifest.json``.

    Returns:
        Parsed manifest dict, or :func:`empty_manifest` if the file is
        missing or corrupt.
    """
    manifest_path = segments_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt manifest.json, will regenerate all segments")
            return empty_manifest()
        # Manifests written before model and language were stored per
        # segment: the run-level values described every entry then, so they
        # are copied onto the entries once. Reading them as a fallback later
        # would let the next run's stamp pass old audio off as current.
        for entry in manifest.get("segments", {}).values():
            entry.setdefault("model", manifest.get("model"))
            entry.setdefault("language", manifest.get("language"))
        return manifest
    return empty_manifest()


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

    # Not fsynced: after a power loss the worst case is regenerating segments.
    atomic_write(segments_dir / "manifest.json", _write, suffix=".json", durable=False)


def record_segment(
    manifest: dict,
    filename: str,
    *,
    speaker: str,
    text: str,
    voice_sample_name: str,
    model_size: str,
    language: str,
) -> None:
    """Record what produced one generated segment, for :func:`segment_is_current`.

    Model and language are stored per segment: a run that stops early, or
    one limited to some speakers, leaves other segments made by a previous
    model, and a run-level stamp would pass those off as current.
    """
    from datetime import datetime, timezone

    manifest.setdefault("segments", {})[filename] = {
        "speaker": speaker,
        "voice_sample": voice_sample_name,
        "text_hash": _text_hash(text),
        "model": model_size,
        "language": language,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def segment_is_current(
    manifest: dict,
    filename: str,
    text: str,
    voice_sample_name: str,
    model_size: str,
    language: str,
) -> bool:
    """Check if a previously generated segment is still valid.

    A segment is valid only if ALL of these match:

    - The WAV file exists (checked by caller)
    - The model and language that produced it
    - Segment text hasn't changed (hash match)
    - Same voice sample was used for this speaker

    Args:
        manifest: loaded manifest dict from :func:`load_manifest`.
        filename: WAV filename key in the manifest (e.g. ``"0001_Alice.wav"``).
        text: current segment text to compare against stored hash.
        voice_sample_name: filename of the voice sample that would be used now.
        model_size: TTS model size (``"0.6B"`` or ``"1.7B"``).
        language: target language string.

    Returns:
        ``True`` if the existing segment can be reused, ``False`` otherwise.
    """
    entry = manifest.get("segments", {}).get(filename)
    if not entry:
        return False
    return (
        entry.get("model") == model_size
        and entry.get("language") == language
        and entry.get("text_hash") == _text_hash(text)
        and entry.get("voice_sample") == voice_sample_name
    )


# ──────────────────────────────────────────────
# STEP 1 — Voice sample extraction
# ──────────────────────────────────────────────


def _extract_clip(audio_path: Path, seg: dict, output_path: Path) -> dict:
    """Extract a single audio clip via ffmpeg, resampled to 16 kHz mono WAV.

    ``-ss`` before ``-i`` seeks the input instead of decoding from the start
    of the file up to the clip; the output is re-encoded, so the cut stays
    sample-accurate.

    Args:
        audio_path: source audio file.
        seg: segment dict with ``start``, ``end``, ``duration``, and ``text`` keys.
        output_path: destination path for the extracted WAV clip.

    Returns:
        Dict with ``file``, ``start``, ``end``, ``duration``, and ``text`` fields.

    Raises:
        RuntimeError: see ``run_ffmpeg``.
    """
    run_ffmpeg(
        [
            "-y",
            "-ss",
            str(seg["start"]),
            "-i",
            str(audio_path),
            "-t",
            str(max(seg["end"] - seg["start"], 0)),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(output_path),
        ],
        what=(f"extract {seg['start']:.1f}-{seg['end']:.1f}s from {audio_path.name}"),
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
    from podcodex.core._utils import (
        fill_narrator_speaker,
        speaker_file_slug,
        temp_sibling,
    )

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

    # The label only reaches disk through ``speaker_file_slug``: an imported
    # subtitle can name a speaker "../../x", which would otherwise write the
    # clips outside voice_samples/ and let the cleanup glob below follow the
    # ".." segments into ancestor directories. The raw label stays the dict key.
    # Clips are extracted under temporary names (temp_sibling: reaped by
    # startup recovery) and swapped in only once every one succeeded, so a
    # failed re-extract leaves the previous samples in place.
    plan: list[tuple[str, dict, Path, Path]] = []
    for speaker, segs in by_speaker.items():
        slug = speaker_file_slug(speaker)
        for i, seg in enumerate(segs):
            final = samples_dir / f"{slug}_{i:02d}.wav"
            plan.append((speaker, seg, temp_sibling(final), final))

    extracted: list[tuple[str, dict, Path]] = []
    try:
        with ThreadPoolExecutor(max_workers=min(len(plan) or 1, 8)) as executor:
            futures = {
                executor.submit(_extract_clip, p.audio_path, seg, tmp): (
                    speaker,
                    final,
                )
                for speaker, seg, tmp, final in plan
            }
            for future in as_completed(futures):
                speaker, final = futures[future]
                extracted.append((speaker, future.result(), final))
    except BaseException:
        for _speaker, _seg, tmp, _final in plan:
            tmp.unlink(missing_ok=True)
        raise

    # Clear old samples for these speakers. Preserve uploaded files
    # (suffixed with ``_custom_``) so a Re-extract doesn't wipe the user's
    # manual uploads.
    for speaker in by_speaker:
        for old in samples_dir.glob(f"{speaker_file_slug(speaker)}_*.wav"):
            if "_custom_" in old.name:
                continue
            old.unlink()

    results: dict[str, list[dict]] = {}
    for speaker, entry, final in extracted:
        Path(entry["file"]).replace(final)
        results.setdefault(speaker, []).append({**entry, "file": final})

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
    from podcodex.core.device import device_str, torch_dtype

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
    values match the vmap'd version once broadcast; the shape can be
    smaller (``(1, 1, Q, KV)`` for a causal mask where vmap gives
    ``(B, H, Q, KV)``), which sdpa broadcasts too. The small memory bump
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

    if not hasattr(_mu, "_vmap_for_bhqkv"):
        # A plain assignment would silently add an attribute nothing calls,
        # and the vmap error would come back mid-generation. Fail at load.
        raise RuntimeError(
            "transformers.masking_utils._vmap_for_bhqkv is gone (transformers "
            f"{getattr(__import__('transformers'), '__version__', '?')}); the "
            "MiMi mask patch needs updating, see ML_RUNTIME.md"
        )

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
    for speaker in voice_samples:
        sample = _selected_sample(voice_samples, speaker, sample_index)
        if sample is None:
            continue
        clone_prompts[speaker] = model.create_voice_clone_prompt(
            ref_audio=str(sample["file"]),
            ref_text=sample["text"],
            x_vector_only_mode=True,
        )
        logger.debug(
            f"Voice prompt ready for {speaker} "
            f"({Path(sample['file']).name}, {sample['duration']:.1f}s)"
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
    # Atomic: a kill mid-write must not leave a torn WAV that a manifest
    # entry from an earlier run still vouches for. Not fsynced: a segment is
    # cheap to regenerate, and this runs once per segment.
    atomic_write(
        output_path,
        lambda tmp: sf.write(str(tmp), audio, sr, format="WAV"),
        suffix=".wav",
        durable=False,
    )
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
    if strategy not in ("silence", "original_timing"):
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose 'silence' or 'original_timing'."
        )

    def _write(tmp: Path) -> None:
        # Streamed: segments are written as they are read, so the episode is
        # never held in memory twice (a 3 h episode is about 1 GB per copy).
        with sf.SoundFile(
            str(tmp), mode="w", samplerate=sr, channels=1, format="WAV"
        ) as out:
            if strategy == "silence":
                # Speaker-aware pause: short within a turn, longer at speaker
                # changes. Approximates natural conversational rhythm without
                # the cumulative bloat a single fixed gap produces.
                within = np.zeros(
                    int(max(silence_duration * 0.4, 0.05) * sr), dtype=np.float32
                )
                across = np.zeros(int(silence_duration * sr), dtype=np.float32)
                for i, seg in enumerate(generated):
                    out.write(sf.read(str(seg["audio_file"]), dtype="float32")[0])
                    if i < len(generated) - 1:
                        next_speaker = generated[i + 1].get("speaker") or ""
                        same = (seg.get("speaker") or "") == next_speaker
                        out.write(within if same else across)
            else:
                # Anchor at the first selected segment's start so a narrowed
                # selection (e.g. only segments 12-14 of an episode) doesn't
                # open with a long blank lead-in equal to the first start
                # time. Within the selection, inter-segment gaps still
                # reflect the original podcast's rhythm.
                cursor = generated[0]["start"]
                for seg in generated:
                    gap = seg["start"] - cursor
                    if gap > 0:
                        out.write(np.zeros(int(gap * sr), dtype=np.float32))
                    audio, _ = sf.read(str(seg["audio_file"]), dtype="float32")
                    out.write(audio)
                    cursor = seg["start"] + len(audio) / sr

    # Atomic: the output is a version path, and a partial WAV there reads as
    # a finished synthesis to the status reconcile and to backfill.
    atomic_write(out_path, _write, suffix=".wav")
    duration = wav_duration(out_path)
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
        {speaker: [{"file": Path, "duration": float, "text": ""}, ...]}, each
        list ordered uploads first (newest first), then extracted clips.
    """
    from podcodex.core._utils import VOICE_SAMPLES_DIR, speaker_file_slug

    samples_dir = Path(output_dir) / VOICE_SAMPLES_DIR
    if not samples_dir.exists():
        logger.debug(f"No voice_samples/ directory in {output_dir}")
        return {}

    reverse_map = {v: k for k, v in (speaker_map or {}).items()}

    def ordered(slug: str) -> list[Path]:
        # The clone reference is the first entry. A sample the user uploaded
        # (newest first) is a deliberate choice and outranks the extracted
        # clips, which sort after it by index.
        files = list(samples_dir.glob(f"{slug}_*.wav"))
        custom = sorted(
            (f for f in files if "_custom_" in f.name),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        return custom + sorted(f for f in files if "_custom_" not in f.name)

    result: dict[str, list[dict]] = {}
    for speaker in speakers:
        files = ordered(speaker_file_slug(speaker))
        if not files:
            speaker_id = reverse_map.get(speaker)
            if speaker_id:
                files = ordered(speaker_file_slug(speaker_id))
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
        List of segment dicts with ``audio_file``, ``sample_rate`` and
        ``duration`` (of the generated audio, not the source span) fields
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
                    "duration": info.duration,
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
