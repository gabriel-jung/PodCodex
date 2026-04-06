"""
podcodex.core.translate — Translation pipeline for podcast transcripts.

Translates source text to a target language without correcting the source.
(For source correction only, use podcodex.core.polish.)

Modes:
    - manual  : user provides the translated JSON directly (e.g. via a LLM UI)
    - ollama  : local LLM via Ollama
    - api     : external API (OpenAI, Anthropic, etc.)

Output:
    .versions/{lang}/{id}.json  — versioned translated segments (via version DB)
"""

from collections.abc import Callable
from pathlib import Path

from loguru import logger

from podcodex.core._utils import (
    DEFAULT_BATCH_MINUTES,
    AudioPaths,
    batch_segments_by_duration,
    build_llm_prompt,
    format_segments,
    normalize_lang,
    run_llm_pipeline,
)
from podcodex.core.pipeline_db import mark_step
from podcodex.core.versions import _get_db, save_version


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────


def _build_prompt(
    context: str = "", source_lang: str = "French", target_lang: str = "English"
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
    """Generate a prompt to paste into a LLM UI for manual translation."""
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
    """Split segments into time-based batches and return one prompt per batch."""
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
    max_gap: float = 10.0,
    provider: str | None = None,
    on_batch: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Translate transcript segments to the target language."""
    logger.info(
        f"Translating {len(segments)} segments — mode={mode}, {source_lang} → {target_lang}"
    )
    system_prompt = _build_prompt(
        context, source_lang=source_lang, target_lang=target_lang
    )
    result = run_llm_pipeline(
        segments,
        system_prompt,
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
    )
    logger.success(f"Translation done — {len(result)} segments")
    return result


# ──────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────


def save_translation(
    audio_path: Path | str,
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


save_translation_raw = save_translation


def list_translations(
    audio_path: Path | str,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Return sorted list of available translation language names for this episode."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    try:
        db = _get_db(p.base)
        steps = db.list_steps(p.base.name)
    except Exception:
        return []
    non_translation = {"transcript", "polished", "indexed"}
    return sorted(s for s in steps if s not in non_translation)
