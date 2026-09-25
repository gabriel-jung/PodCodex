"""
podcodex.core.translate — Translation pipeline for podcast transcripts.

Translates source text to a target language without correcting the source.
(For source correction only, use podcodex.core.correct.)

Modes:
    - manual  : user provides the translated JSON directly (e.g. via a LLM UI)
    - ollama  : local LLM via Ollama
    - api     : external API (OpenAI, Anthropic, etc.)

Output:
    {lang}/{id}.json  — versioned translated segments (via version DB)
"""

from collections.abc import Callable
from pathlib import Path

from loguru import logger

from podcodex.core._utils import DEFAULT_BATCH_MINUTES, AudioPaths, normalize_lang
from podcodex.core.llm import (
    build_batched_manual_prompts,
    build_llm_prompt,
    format_segments,
    output_format_rules,
    run_llm_step,
)
from podcodex.core.pipeline_db import mark_step
from podcodex.core.versions import save_version, translation_steps


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────


def _build_prompt(
    context: str = "", source_lang: str = "English", target_lang: str = "French"
) -> str:
    """Build the system prompt for transcript translation."""
    return build_llm_prompt(
        role=f"You are translating a transcript from a podcast in {source_lang}.",
        task=f"""\
Your task: translation only.
- Translate into natural, conversational {target_lang}
- Preserve the oral tone and style of the podcast
- Do not translate proper nouns (people, films, places)
- Translate the full text — never truncate or summarize""",
        output=output_format_rules("untranslatable"),
        context=context,
    )


# ──────────────────────────────────────────────
# Manual
# ──────────────────────────────────────────────


def build_manual_prompt(
    segments: list[dict],
    context: str = "",
    source_lang: str = "English",
    target_lang: str = "French",
    start_index: int = 0,
) -> str:
    """Generate a prompt to paste into a LLM UI for manual translation."""
    prompt = _build_prompt(
        context=context, source_lang=source_lang, target_lang=target_lang
    )
    segments_text = format_segments(
        segments, instruction="Translate", start_index=start_index
    )
    return f"{prompt}\n\n{segments_text}"


def build_manual_prompts_batched(
    segments: list[dict],
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    context: str = "",
    source_lang: str = "English",
    target_lang: str = "French",
    batch_count: int | None = None,
) -> list[tuple[list[dict], str]]:
    """Split segments into time-based batches and return one prompt per batch."""
    return build_batched_manual_prompts(
        segments,
        lambda batch, start: build_manual_prompt(
            batch,
            context=context,
            source_lang=source_lang,
            target_lang=target_lang,
            start_index=start,
        ),
        batch_minutes,
        batch_count,
    )


# ──────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────


def translate_segments(
    segments: list[dict],
    mode: str = "ollama",
    context: str = "",
    source_lang: str = "English",
    target_lang: str = "French",
    model: str = "",
    api_base_url: str = "",
    api_key: str | None = None,
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    original_segments: list[dict] | None = None,
    merge: bool = True,
    max_gap: float = 10.0,
    provider: str | None = None,
    on_batch: Callable[[int, int], None] | None = None,
    audio_path: str | None = None,
    output_dir: str | None = None,
) -> list[dict]:
    """Translate transcript segments to the target language.

    When *audio_path*/*output_dir* identify an episode, each batch's LLM
    outcome (ollama/api modes) is recorded to ``llm_failures.json`` so a
    silently-rejected batch can be reviewed afterwards.
    """
    logger.info(
        f"Translating {len(segments)} segments — mode={mode}, {source_lang} → {target_lang}"
    )
    system_prompt = _build_prompt(
        context, source_lang=source_lang, target_lang=target_lang
    )
    result = run_llm_step(
        normalize_lang(target_lang),
        segments,
        system_prompt,
        audio_path=audio_path,
        output_dir=output_dir,
        mode=mode,
        model=model,
        api_base_url=api_base_url,
        api_key=api_key,
        batch_minutes=batch_minutes,
        provider=provider,
        instruction="Translate",
        label="Translate",
        original_segments=original_segments,
        merge=merge,
        max_gap=max_gap,
        on_batch=on_batch,
        # The length guard is for correction. A translation legitimately
        # changes length: English to Chinese is often a third of the
        # characters, and the guard saved those as the English source.
        min_length_ratio=0,
        # A small model sometimes answers a batch with the source verbatim.
        flag_unchanged=True,
    )
    logger.success(f"Translation done, {len(result)} segments")
    return result


# ──────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────


def save_translation(
    audio_path: Path | str | None,
    segments: list[dict],
    lang: str,
    output_dir: str | Path | None = None,
    provenance: dict | None = None,
) -> str:
    """Save translated segments (version DB + pipeline DB). Returns the version id."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    lang_norm = normalize_lang(lang)
    version_id = save_version(p.base, lang_norm, segments, provenance)
    prov_update = {lang_norm: provenance} if provenance else {}
    translations = list_translations(audio_path, output_dir=output_dir)
    mark_step(
        p.show_dir, p.base.name, translations=translations, provenance=prov_update
    )
    return version_id


def list_translations(
    audio_path: Path | str,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Return sorted list of available translation language names for this episode."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    return translation_steps(p.base)
