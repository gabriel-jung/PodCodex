"""
podcodex.core.translate — Translation pipeline for podcast transcripts.

Translates source text to a target language without correcting the source.
(For source correction only, use podcodex.core.polish.)

Modes:
    - manual  : user provides the translated JSON directly (e.g. via a LLM UI)
    - ollama  : local LLM via Ollama
    - api     : external API (OpenAI, Anthropic, etc.)

Output files:
    {stem}.{lang_norm}.json  — translated segments, e.g. {stem}.english.json
"""

from pathlib import Path

from loguru import logger

from podcodex.core._utils import (
    BREAK_SPEAKER,
    DEFAULT_BATCH_MINUTES,
    DEFAULT_MAX_GAP,
    INTERNAL_SUFFIXES,
    AudioPaths,
    batch_segments_by_duration,
    build_llm_prompt,
    merge_consecutive_segments,
    read_json,
    run_api,
    run_ollama,
    save_segments_json,
    validate_manual,
)


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
- Translate the full text — never truncate or summarize
- Segments with speaker "{BREAK_SPEAKER}" are music or jingle breaks — copy them to the output exactly as-is, do not translate their text""",
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

    Combines the system prompt from ``_build_prompt`` with the
    numbered segments so the user can paste a single block into any LLM UI.
    """
    prompt = _build_prompt(
        context=context, source_lang=source_lang, target_lang=target_lang
    )
    segments_text = "\n\n".join(
        f"[{i}] {seg['text']}" for i, seg in enumerate(segments)
    )
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
    batch_size: int = 10,
    original_segments: list[dict] | None = None,
    merge: bool = True,
    max_gap: float = DEFAULT_MAX_GAP,
    provider: str | None = None,
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
        batch_size        : number of segments per LLM call
        original_segments : only for manual mode — the original source segments
                            (used to merge metadata with the LLM-provided translations)
        merge             : merge consecutive same-speaker segments before processing
                            (default True, ignored in manual mode)
        max_gap           : maximum silence gap in seconds for merging (default 10.0)
        provider          : provider shorthand ("openai", "anthropic", "mistral", "groq")
                            — sets api_base_url, model, and api_key env var automatically

    Returns:
        List of segments with translated text field.
    """
    logger.info(
        f"Translating {len(segments)} segments — mode={mode}, {source_lang} → {target_lang}"
    )
    logger.debug(f"Translate params: batch_size={batch_size}, merge={merge}")
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
            batch_size=batch_size,
            instruction="Translate",
            label="Translate",
        )
    elif mode == "api":
        result = run_api(
            segments,
            system_prompt,
            model=model,
            api_base_url=api_base_url,
            api_key=api_key,
            batch_size=batch_size,
            provider=provider,
            instruction="Translate",
            label="Translate",
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
    nodiar: bool = False,
) -> Path:
    """Save pipeline-generated translation to {stem}.{lang_norm}.raw.json.

    Use this for LLM/pipeline output. The user can then review and promote
    to the validated {stem}.{lang_norm}.json.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir, nodiar=nodiar)
    return save_segments_json(
        p.translation_raw(lang), segments, f"Translation ({lang})"
    )


def save_translation(
    audio_path: Path | str,
    segments: list[dict],
    lang: str,
    output_dir: str | Path | None = None,
    nodiar: bool = False,
) -> Path:
    """Save validated/edited translation to {stem}.{lang_norm}.json."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir, nodiar=nodiar)
    return save_segments_json(p.translation(lang), segments, f"Translation ({lang})")


def load_translation(
    audio_path: Path | str,
    lang: str,
    output_dir: str | Path | None = None,
    nodiar: bool = False,
) -> list[dict]:
    """Load translated segments. Prefers validated .json, falls back to .raw.json."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir, nodiar=nodiar)
    return read_json(p.translation_best(lang))


def load_translation_raw(
    audio_path: Path | str,
    lang: str,
    output_dir: str | Path | None = None,
    nodiar: bool = False,
) -> list[dict]:
    """Load specifically from .{lang}.raw.json (pipeline output)."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir, nodiar=nodiar)
    return read_json(p.translation_raw(lang))


def load_translation_validated(
    audio_path: Path | str,
    lang: str,
    output_dir: str | Path | None = None,
    nodiar: bool = False,
) -> list[dict]:
    """Load specifically from .{lang}.json (user-validated)."""
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir, nodiar=nodiar)
    return read_json(p.translation(lang))


def list_translations(
    audio_path: Path | str,
    output_dir: str | Path | None = None,
    nodiar: bool = False,
) -> list[str]:
    """Return sorted list of available translation language names for this episode.

    Scans for {stem}.*.json files in output_dir, excluding internal suffixes.
    Merges raw and validated: english.json and english.raw.json both contribute
    "english" once. Returns normalised language names, e.g. ["english", "spanish"].

    When *nodiar* is True, only returns languages from nodiar translation files.
    When False (default), only returns languages from diarized translation files.
    """
    audio_path = Path(audio_path)
    root = AudioPaths.output_dir(audio_path, output_dir)
    prefix = "nodiar." if nodiar else ""
    langs: set[str] = set()
    for f in sorted(root.glob(f"{audio_path.stem}.{prefix}*.json")):
        suffix = f.stem[len(audio_path.stem) + 1 :]
        if suffix in INTERNAL_SUFFIXES:
            continue
        if suffix.endswith(".raw"):
            base = suffix[:-4]  # strip ".raw"
            if base not in INTERNAL_SUFFIXES:
                langs.add(base.removeprefix("nodiar."))
        else:
            langs.add(suffix.removeprefix("nodiar."))
    return sorted(langs)
