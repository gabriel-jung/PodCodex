"""
podcodex.core.translate — Translation pipeline for podcast transcripts.

Translates source text to a target language without correcting the source.
(For source correction only, use podcodex.core.polish.)

Modes:
    - manual  : user provides the translated JSON directly (e.g. via a LLM UI)
    - ollama  : local LLM via Ollama
    - api     : external API (OpenAI, Anthropic, etc.)

Output files:
    .versions/{lang}/{id}.json  — versioned translated segments (primary store)
    {stem}.{lang_norm}.raw.json — legacy copy of pipeline output
    {stem}.{lang_norm}.json     — legacy copy of validated output
"""

from collections.abc import Callable
from pathlib import Path

from loguru import logger

from podcodex.core._utils import (
    DEFAULT_BATCH_MINUTES,
    DEFAULT_MAX_GAP,
    INTERNAL_SUFFIXES,
    AudioPaths,
    batch_segments_by_duration,
    build_llm_prompt,
    format_segments,
    merge_consecutive_segments,
    normalize_lang,
    read_json,
    run_api,
    run_ollama,
    save_segments_json,
    validate_manual,
)
from podcodex.core.pipeline_db import mark_step
from podcodex.core.versions import _get_db, load_latest, save_version


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────


def _build_prompt(
    context: str = "", source_lang: str = "French", target_lang: str = "English"
) -> str:
    """Build the system prompt for transcript translation.

    Args:
        context     : podcast context (names, topics) to preserve correct spellings
        source_lang : source language (e.g. "French")
        target_lang : target language (e.g. "English")
    """
    return build_llm_prompt(
        role=f"You are translating a transcript from a podcast in {source_lang}.",
        task=f"""\
Your task: translation only.
- Translate into natural, conversational {target_lang}
- Preserve the oral tone and style of the podcast
- Do not translate proper nouns (people, films, places)
- Translate the full text — never truncate or summarize""",
        output="""\
Output format:
Return a JSON array with one entry per segment, containing only the index and translated text.
Reply ONLY with valid JSON, no surrounding text, no markdown.
Format: [{"index": 0, "text": "translated text..."}]""",
        context=context,
    )


# ──────────────────────────────────────────────
# Manual
# ──────────────────────────────────────────────


def build_manual_prompt(
    segments: list[dict],
    context: str = "",
    source_lang: str = "French",
    target_lang: str = "English",
) -> str:
    """Generate a prompt to paste into a LLM UI for manual translation.

    Uses :func:`~podcodex.core._utils.format_segments` — the same
    ``[i] text`` format that ollama/api modes use.
    ``[BREAK]`` segments are excluded from the prompt.
    """
    prompt = _build_prompt(
        context=context, source_lang=source_lang, target_lang=target_lang
    )
    segments_text = format_segments(segments, instruction="Translate")
    return f"{prompt}\n\n{segments_text}"


def build_manual_prompts_batched(
    segments: list[dict],
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    context: str = "",
    source_lang: str = "French",
    target_lang: str = "English",
) -> list[tuple[list[dict], str]]:
    """Split segments into time-based batches and return one prompt per batch.

    Args:
        segments      : transcript segments to batch
        batch_minutes : maximum duration per batch in minutes (default 15)
        context       : podcast context to include in each prompt
        source_lang   : source language (e.g. "French")
        target_lang   : target language (e.g. "English")

    Returns:
        List of (batch_segments, prompt_string) tuples.
    """
    return [
        (
            batch,
            build_manual_prompt(
                batch,
                context=context,
                source_lang=source_lang,
                target_lang=target_lang,
            ),
        )
        for batch in batch_segments_by_duration(segments, batch_minutes)
    ]


# ──────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────


def translate_segments(
    segments: list[dict],
    mode: str = "ollama",
    context: str = "",
    source_lang: str = "French",
    target_lang: str = "English",
    model: str = "",
    api_base_url: str = "",
    api_key: str | None = None,
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    original_segments: list[dict] | None = None,
    merge: bool = True,
    max_gap: float = DEFAULT_MAX_GAP,
    provider: str | None = None,
    on_batch: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """
    Translate transcript segments to the target language.

    Args:
        segments          : source segments to translate (for ollama/api mode),
                            or LLM output with translated text (for manual mode)
        mode              : "manual", "ollama", or "api"
        context           : podcast context to guide the LLM
        source_lang       : source language (e.g. "French")
        target_lang       : target language (e.g. "English")
        model             : LLM model name (auto-detected from provider if empty)
        api_base_url      : base URL for OpenAI-compatible API (auto-detected from provider if empty)
        api_key           : API key (None reads from provider's env variable)
        batch_minutes     : maximum audio duration per LLM batch in minutes (default 15)
        original_segments : only for manual mode — the original source segments
                            (used to merge metadata with the LLM-provided translations)
        merge             : merge consecutive same-speaker segments before processing
                            (default True, ignored in manual mode)
        max_gap           : maximum silence gap in seconds for merging (default 10.0)
        provider          : provider shorthand ("openai", "anthropic", "mistral", "groq")
                            — sets api_base_url, model, and api_key env var automatically
        on_batch          : optional callback(batch_num, total_batches) for progress

    Returns:
        List of segments with translated text field.
    """
    logger.info(
        f"Translating {len(segments)} segments — mode={mode}, {source_lang} → {target_lang}"
    )
    logger.debug(f"Translate params: batch_minutes={batch_minutes}, merge={merge}")
    if context:
        logger.debug(f"Context: {context[:100]}{'…' if len(context) > 100 else ''}")

    if mode == "manual":
        orig = original_segments if original_segments is not None else segments
        return validate_manual(segments, orig)

    if merge:
        segments = merge_consecutive_segments(segments, max_gap=max_gap)
        logger.info(f"After merge: {len(segments)} segments")

    system_prompt = _build_prompt(
        context, source_lang=source_lang, target_lang=target_lang
    )

    if mode == "ollama":
        result = run_ollama(
            segments,
            system_prompt,
            model=model or "qwen3:4b",
            batch_minutes=batch_minutes,
            instruction="Translate",
            label="Translate",
            on_batch=on_batch,
        )
    elif mode == "api":
        result = run_api(
            segments,
            system_prompt,
            model=model,
            api_base_url=api_base_url,
            api_key=api_key,
            batch_minutes=batch_minutes,
            provider=provider,
            instruction="Translate",
            label="Translate",
            on_batch=on_batch,
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Choose from 'manual', 'ollama', 'api'."
        )

    logger.success(f"Translation done — {len(result)} segments")
    return result


# ──────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────


def save_translation_raw(
    audio_path: Path | str,
    segments: list[dict],
    lang: str,
    output_dir: str | Path | None = None,
    provenance: dict | None = None,
) -> Path:
    """Save pipeline-generated translated segments.

    Creates a new version in ``.versions/{lang}/`` and a legacy file copy.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    path = p.translation_raw(lang)
    save_segments_json(path, segments, f"Translation ({lang})")
    lang_norm = normalize_lang(lang)
    save_version(p.base, lang_norm, segments, provenance)
    prov_update = {lang_norm: provenance} if provenance else {}
    translations = list_translations(audio_path, output_dir=output_dir)
    mark_step(
        p.show_dir, p.base.name, translations=translations, provenance=prov_update
    )
    return path


def save_translation(
    audio_path: Path | str,
    segments: list[dict],
    lang: str,
    output_dir: str | Path | None = None,
    provenance: dict | None = None,
) -> Path:
    """Save validated/edited translated segments.

    Creates a new version in ``.versions/{lang}/`` and a legacy file copy.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    path = p.translation(lang)
    save_segments_json(path, segments, f"Translation ({lang})")
    lang_norm = normalize_lang(lang)
    save_version(p.base, lang_norm, segments, provenance)
    prov_update = {lang_norm: provenance} if provenance else {}
    translations = list_translations(audio_path, output_dir=output_dir)
    mark_step(
        p.show_dir, p.base.name, translations=translations, provenance=prov_update
    )
    return path


def load_translation(
    audio_path: Path | str,
    lang: str,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Load translated segments — latest version, falls back to legacy files."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    lang_norm = normalize_lang(lang)
    segments = load_latest(p.base, lang_norm)
    if segments is not None:
        return segments
    return read_json(p.translation_best(lang))


def list_translations(
    audio_path: Path | str,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Return sorted list of available translation language names for this episode.

    Tries the version DB first, falls back to filesystem scan.
    """
    audio_path = Path(audio_path)
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    # Try DB first
    try:
        db = _get_db(p.base)
        steps = db.list_steps(p.base.name)
        # Filter out non-translation steps
        non_translation = {"transcript", "polished", "indexed"}
        langs = [s for s in steps if s not in non_translation]
        if langs:
            return sorted(langs)
    except Exception:
        pass

    # Legacy fallback: scan filesystem
    # New convention: {stem}.translated.{lang}.(raw.)json
    # Legacy convention: {stem}.{lang}.(raw.)json (simple name, no dots)
    root = AudioPaths.output_dir(audio_path, output_dir)
    langs_set: set[str] = set()
    for f in sorted(root.glob(f"{audio_path.stem}.*.json")):
        suffix = f.stem[len(audio_path.stem) + 1 :]
        if suffix.endswith(".raw"):
            suffix = suffix[:-4]
        # New convention: translated.{lang}
        if suffix.startswith("translated."):
            langs_set.add(suffix[len("translated.") :])
            continue
        # Legacy: simple name with no dots, not an internal suffix
        if "." not in suffix and suffix not in INTERNAL_SUFFIXES:
            langs_set.add(suffix)
    return sorted(langs_set)
