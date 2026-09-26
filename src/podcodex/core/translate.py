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
from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    from podcodex.core.llm_resolver import LLMRun
    from podcodex.core.source import SourceVersion


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
    records_out: list[dict] | None = None,
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
        records_out=records_out,
        # The length guard is for correction. A translation legitimately
        # changes length: English to Chinese is often a third of the
        # characters, and the guard saved those as the English source.
        min_length_ratio=0,
        # A small model sometimes answers a batch with the source verbatim.
        flag_unchanged=True,
    )
    logger.success(f"Translation done, {len(result)} segments")
    return result


def translate_and_save(
    source: "SourceVersion",
    llm: "LLMRun",
    *,
    audio_path: str,
    output_dir: str | None = None,
    target_lang: str,
    context: str = "",
    source_lang: str = "English",
    batch_minutes: float = DEFAULT_BATCH_MINUTES,
    provider_profile: str | None = None,
    key_name: str | None = None,
    on_batch: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], str]:
    """One auto translate run: translate *source*, save it, record its failures.

    The single run path for the translate route and the batch runner.
    Returns ``(segments, version_id)``.
    """
    from podcodex.core.provenance import save_llm_run

    records: list[dict] = []
    translated = translate_segments(
        source.segments,
        **llm.pipeline_kwargs(),
        context=context,
        source_lang=source_lang,
        target_lang=target_lang,
        batch_minutes=batch_minutes,
        original_segments=source.segments,
        merge=False,  # sources are already merged on load/upload
        on_batch=on_batch,
        audio_path=audio_path,
        output_dir=output_dir,
        records_out=records,
    )
    version_id = save_llm_run(
        normalize_lang(target_lang),
        lambda prov: save_translation(
            audio_path, translated, target_lang, output_dir=output_dir, provenance=prov
        ),
        source=source,
        llm=llm,
        audio_path=audio_path,
        output_dir=output_dir,
        records=records,
        provider_profile=provider_profile,
        key_name=key_name,
        source_lang=source_lang,
        target_lang=target_lang,
        batch_minutes=batch_minutes,
    )
    return translated, version_id


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
